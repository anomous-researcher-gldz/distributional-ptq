"""Monkey-patch TesseraQ's IntegerQuantizer to add DBAF + PCSA-tf hooks.

TesseraQ (llmc/compression/quantization/quant.py) routes every fake-quant op
through two top-level methods on the IntegerQuantizer base:

    IntegerQuantizer.fake_quant_act_dynamic(act, args)        -> quantized act
    IntegerQuantizer.fake_quant_weight_dynamic(weight, args)  -> quantized w

These are called from:
    BaseBlockwiseQuantization.a_qdq          (activation path)
    BaseBlockwiseQuantization.w_qdq          (weight path)
    TesseraQ.* block-recon inner loop        (both)

We wrap both at the IntegerQuantizer class level so every TesseraQ-using
quantizer instance inherits the folded + routed behaviour with no upstream
code edits. The base FloatQuantizer is left alone (irrelevant for INT4).

DBAF fold + unfold are imported from ahcptq (shared across all hosts).
PCSA-tf state is fit once on per-block hidden-state descriptors collected
during the AWQ-init pass that precedes TesseraQ recon; at TesseraQ time the
routed scale is applied to the activation tensor BEFORE the standard
fake_quant_act_dynamic path.

Usage (from a script that runs TesseraQ):

    import sys
    sys.path.insert(0, '/home/ubuntu/unifying-ptq')
    sys.path.insert(0, '/home/ubuntu/unifying-ptq/TesseraQ')
    import tesseraq_dbaf_pcsa_patch as p

    # DBAF only — no calibration needed
    p.install_dbaf_patches(dbaf_alpha=0.95)

    # PCSA-tf — fit once after AWQ scales are loaded but before TesseraQ recon
    # (descriptors are mean-pooled hidden states per layer).
    p.fit_pcsa_tf_on_calib_data(descriptors, activations, K=8)
    p.install_pcsa_tf()

    # Set the per-prompt descriptor before each forward (TesseraQ recon
    # processes a calibration batch; descriptor is the layer-mean of that batch).
    p.set_descriptor(desc)
"""
from __future__ import annotations
import sys

# Make our DBAF + PCSA-tf utilities importable from anywhere
sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")

import torch
from ahcptq.quantization.fake_quant import (
    is_like_normal_plus_3sigma_outliers, fold_outliers, unfold_outliers,
)
from flatquant.baselines.pcsa_tf import (
    fit_pcsa_tf, route_pcsa_tf,
)


_PCSA_STATE: dict | None = None
_DBAF_ALPHA: float = 0.95
_DBAF_INSTALLED: bool = False
_PCSA_INSTALLED: bool = False
_CURRENT_DESCRIPTOR: torch.Tensor | None = None


def set_descriptor(desc: torch.Tensor):
    """Set the per-prompt descriptor used by PCSA-tf routing.

    Called by the TesseraQ block-recon loop just before iterating the
    inner reconstruction batch.
    """
    global _CURRENT_DESCRIPTOR
    _CURRENT_DESCRIPTOR = desc


def fit_pcsa_tf_on_calib_data(descs: torch.Tensor, acts: torch.Tensor, K: int = 8):
    """Fit PCSA-tf state once from collected calibration descriptors + acts.

    Args:
        descs: [N, hidden_dim] per-batch layer-mean hidden-state descriptors.
        acts:  [N, ...] activation tensors collected across calibration prompts.
        K:     Number of PCSA anchors (typical 4-8).
    """
    global _PCSA_STATE
    _PCSA_STATE = fit_pcsa_tf(descs, acts, K=K)
    print(f"[pcsa_tf] fitted K={K} anchors; scales={_PCSA_STATE['scales'].tolist()}",
          flush=True)


def install_dbaf_patches(dbaf_alpha: float = 0.95):
    """Wrap IntegerQuantizer.fake_quant_{act,weight}_dynamic with DBAF gate.

    DBAF only fires on tensors matching the dense-core + sparse-outlier signature
    (kurt 3-30, |skew| <= 0.7, frac3sigma in [1e-4, 2e-2]). Other distributions
    pass through unchanged, so we don't disturb post-softmax or post-GELU layers
    where TesseraQ's block-recon already adapts.

    Safe to call multiple times; second call is a no-op.
    """
    global _DBAF_ALPHA, _DBAF_INSTALLED
    if _DBAF_INSTALLED:
        print("[dbaf_patch] already installed; skipping", flush=True)
        return
    _DBAF_ALPHA = dbaf_alpha

    from llmc.compression.quantization.quant import IntegerQuantizer

    orig_act = IntegerQuantizer.fake_quant_act_dynamic
    orig_w = IntegerQuantizer.fake_quant_weight_dynamic

    def wrapped_act(self, act, args=None):
        args = args if args is not None else {}
        # Gate on the dense-core + sparse-outlier signature
        try:
            fires = bool(is_like_normal_plus_3sigma_outliers(act.detach()))
        except Exception:
            fires = False
        if not fires:
            return orig_act(self, act, args)
        folded, fold_meta = fold_outliers(act, alpha=_DBAF_ALPHA)
        q_folded = orig_act(self, folded, args)
        return unfold_outliers(q_folded, fold_meta)

    def wrapped_w(self, weight, args=None):
        args = args if args is not None else {}
        try:
            fires = bool(is_like_normal_plus_3sigma_outliers(weight.detach()))
        except Exception:
            fires = False
        if not fires:
            return orig_w(self, weight, args)
        folded, fold_meta = fold_outliers(weight, alpha=_DBAF_ALPHA)
        q_folded = orig_w(self, folded, args)
        return unfold_outliers(q_folded, fold_meta)

    IntegerQuantizer.fake_quant_act_dynamic = wrapped_act
    IntegerQuantizer.fake_quant_weight_dynamic = wrapped_w
    _DBAF_INSTALLED = True
    print(f"[dbaf_patch] installed on IntegerQuantizer (alpha={dbaf_alpha})",
          flush=True)


def install_pcsa_tf():
    """Wrap IntegerQuantizer.fake_quant_act_dynamic with PCSA-tf scale routing.

    Stacks on top of DBAF if already installed — the order is:
        x  -- DBAF gate --> folded(x)  -- PCSA-tf routing --> rescaled(folded)
            -- IntegerQuantizer quant --> q
            -- DBAF unfold --> final
    """
    global _PCSA_INSTALLED
    if _PCSA_INSTALLED:
        print("[pcsa_tf] already installed; skipping", flush=True)
        return
    if _PCSA_STATE is None:
        raise RuntimeError("PCSA-tf state not fit. Call fit_pcsa_tf_on_calib_data first.")

    from llmc.compression.quantization.quant import IntegerQuantizer
    orig_act = IntegerQuantizer.fake_quant_act_dynamic

    def wrapped_pcsa(self, act, args=None):
        args = args if args is not None else {}
        # Route via current descriptor (one anchor per prompt)
        if _CURRENT_DESCRIPTOR is None:
            return orig_act(self, act, args)
        try:
            rescaled = route_pcsa_tf(act, _CURRENT_DESCRIPTOR, _PCSA_STATE)
        except Exception:
            return orig_act(self, act, args)
        return orig_act(self, rescaled, args)

    IntegerQuantizer.fake_quant_act_dynamic = wrapped_pcsa
    _PCSA_INSTALLED = True
    print("[pcsa_tf] installed on IntegerQuantizer.fake_quant_act_dynamic", flush=True)


# Convenience function: install both at once (DBAF before PCSA-tf)
def install_both(dbaf_alpha: float = 0.95):
    install_dbaf_patches(dbaf_alpha=dbaf_alpha)
    if _PCSA_STATE is not None:
        install_pcsa_tf()
    else:
        print("[install_both] DBAF installed; PCSA-tf skipped (state not fit)",
              flush=True)
