"""Monkey-patch 2DQuant's FakeQuantizerWeight / FakeQuantizerAct to add DBAF + PCSA-tf hooks.

2DQuant (basicsr/archs/quant_arch.py) uses two separate quantizer classes:
  - FakeQuantizerWeight  — calibrated with DOBI, used in QuantLinear.weight_quantizer
                            and QuantConv2d.weight_quantizer
  - FakeQuantizerAct     — same DOBI calibration but also supports dynamic / running_stat
                            modes; used in QuantLinear.act_quantizer, QuantConv2d.act_quantizer,
                            and standalone attention quantizers (mymodule_q/k/v/a).

Both share the same fake-quant arithmetic:
    s = (ub - lb) / (2^n - 1)
    return s * round((clip(x, lb, ub) - lb) / s) + lb

We wrap the calibrated path of each class:
  DBAF  — fires when is_like_normal_plus_3sigma_outliers(x) returns True;
           fold_outliers / unfold_outliers bracket the fake-quant arithmetic.
  PCSA-tf — for FakeQuantizerAct only: pre-clamp x to the per-image-cluster anchor
             range before computing the scale, tightening quantisation to the cluster band.

Usage (from a script that has 2DQuant on sys.path):
  import sys; sys.path.insert(0, '/home/ubuntu/unifying-ptq/2DQuant')
  import twodquant_dbaf_pcsa_patch as p
  p.install_dbaf_patches(dbaf_alpha=0.95)
  # ... run DOBI calibration forward pass (net_Q(lq)) ...
  p.fit_pcsa_tf_on_calib_data(descs, acts, K=8)
  p.install_pcsa_tf()

For each subsequent forward pass, set the per-image descriptor before calling net_Q:
  p.set_descriptor(desc)   # desc: [embed_dim] tensor (mean-pool of conv_first output)

Copy this file into 2DQuant/ at use-time:
  cp /home/ubuntu/unifying-ptq/patches/twodquant_dbaf_pcsa_patch.py \\
     /home/ubuntu/unifying-ptq/2DQuant/twodquant_dbaf_pcsa_patch.py
"""
from __future__ import annotations
import sys

# Make our DBAF + PCSA-tf utilities importable
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
_CURRENT_DESCRIPTOR: torch.Tensor | None = None  # set externally per forward pass


def set_descriptor(desc: torch.Tensor):
    """Called by the calibration/inference loop before each image's forward pass."""
    global _CURRENT_DESCRIPTOR
    _CURRENT_DESCRIPTOR = desc


def fit_pcsa_tf_on_calib_data(descs: torch.Tensor, acts: torch.Tensor, K: int = 8):
    """Fit the PCSA-tf state once from collected calibration descriptors + acts.

    Args:
        descs: [N, embed_dim] per-image descriptors (mean-pool of conv_first output).
        acts:  [N, ...] activation tensors collected across calibration images.
        K:     Number of PCSA clusters.
    """
    global _PCSA_STATE
    _PCSA_STATE = fit_pcsa_tf(descs, acts, K=K)
    print(f"[pcsa_tf] fitted K={K} anchors; scales={_PCSA_STATE['scales'].tolist()}",
          flush=True)


def install_dbaf_patches(dbaf_alpha: float = 0.95):
    """Wrap FakeQuantizerWeight.forward and FakeQuantizerAct.forward with DBAF gate.

    The patch only fires on the *calibrated* path (self.calibrated == True). The
    initial DOBI calibration pass (self.calibrated == False) is left untouched so
    that DOBI can still measure the true range before DBAF alters the tensor.

    Safe to call multiple times; second call is a no-op.
    """
    global _DBAF_ALPHA, _DBAF_INSTALLED
    if _DBAF_INSTALLED:
        print("[dbaf_patch] already installed; skipping", flush=True)
        return
    _DBAF_ALPHA = dbaf_alpha

    from basicsr.archs.quant_arch import FakeQuantizerWeight, FakeQuantizerAct

    # ---- FakeQuantizerWeight ----
    orig_weight_forward = FakeQuantizerWeight.forward

    def wrapped_weight_forward(self, x: torch.Tensor):
        # Let DOBI calibration run unmodified
        if not self.calibrated:
            return orig_weight_forward(self, x)
        gate = is_like_normal_plus_3sigma_outliers(x)
        if gate["is_like_c"]:
            T = float(3.0 * gate["stats"]["std"])
            x_fold, tag = fold_outliers(x, T, _DBAF_ALPHA)
            q = orig_weight_forward(self, x_fold)
            return unfold_outliers(q, tag, T, _DBAF_ALPHA)
        return orig_weight_forward(self, x)

    FakeQuantizerWeight.forward = wrapped_weight_forward

    # ---- FakeQuantizerAct ----
    orig_act_forward = FakeQuantizerAct.forward

    def wrapped_act_forward(self, x: torch.Tensor):
        # Let DOBI calibration run unmodified
        if not self.calibrated:
            return orig_act_forward(self, x)
        gate = is_like_normal_plus_3sigma_outliers(x)
        if gate["is_like_c"]:
            T = float(3.0 * gate["stats"]["std"])
            x_fold, tag = fold_outliers(x, T, _DBAF_ALPHA)
            q = orig_act_forward(self, x_fold)
            return unfold_outliers(q, tag, T, _DBAF_ALPHA)
        return orig_act_forward(self, x)

    FakeQuantizerAct.forward = wrapped_act_forward

    _DBAF_INSTALLED = True
    print(f"[dbaf_patch] wrapped FakeQuantizerWeight.forward and FakeQuantizerAct.forward "
          f"(alpha={dbaf_alpha})", flush=True)


def install_pcsa_tf():
    """Activate per-image scale routing on FakeQuantizerAct instances.

    Must be called AFTER fit_pcsa_tf_on_calib_data(...) has populated the state.
    Adds a second wrapper on top of FakeQuantizerAct.forward (which may already be
    DBAF-wrapped) that pre-clamps x to the per-cluster anchor range before the
    fake-quant arithmetic, routing via the current image descriptor.

    Unlike OmniQuant, 2DQuant has separate classes for weight and activation
    quantizers; we only wrap FakeQuantizerAct so weight quantization is unaffected.
    """
    global _PCSA_INSTALLED
    if _PCSA_INSTALLED:
        print("[pcsa_patch] already installed; skipping", flush=True)
        return
    if _PCSA_STATE is None:
        raise RuntimeError("fit_pcsa_tf_on_calib_data must be called before install_pcsa_tf")

    from basicsr.archs.quant_arch import FakeQuantizerAct
    pre_pcsa_forward = FakeQuantizerAct.forward  # may be DBAF-wrapped already

    def routed_act_forward(self, x: torch.Tensor):
        # Only re-scale when a current descriptor exists and PCSA state is ready,
        # and skip during DOBI calibration (calibrated == False).
        if self.calibrated and (_CURRENT_DESCRIPTOR is not None) and (_PCSA_STATE is not None):
            anchor_id = route_pcsa_tf(_CURRENT_DESCRIPTOR, _PCSA_STATE)
            anchor_scale = float(_PCSA_STATE["scales"][anchor_id[0]].item())
            # Pre-clamp x to the cluster anchor range so the subsequent affine
            # quantization (DOBI-fitted or dynamic) operates in the cluster band.
            x = x.clamp(min=-anchor_scale, max=anchor_scale)
        return pre_pcsa_forward(self, x)

    FakeQuantizerAct.forward = routed_act_forward
    _PCSA_INSTALLED = True
    print("[pcsa_patch] wrapped FakeQuantizerAct.forward for PCSA-tf routing", flush=True)
