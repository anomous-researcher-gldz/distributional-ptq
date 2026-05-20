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


def install_dbaf_patches(dbaf_alpha: float = 0.95, act_alpha: float | None = None,
                         act_no_gate: bool = True):
    """Wrap IntegerQuantizer.fake_quant_{act,weight}_dynamic with DBAF.

    Weight path: gated DBAF (kurt 3-30, |skew| <= 0.7, frac3sigma in [1e-4, 2e-2]).
    Weights on unrotated LLMs match this shape, so the gate keeps DBAF off
    post-softmax / post-GELU layers where TesseraQ's recon already adapts.

    Activation path: if act_no_gate=True (default), DBAF fires unconditionally
    on every activation tensor. Unrotated Llama-3-8B deep-layer activations
    have kurt 50-500+ and frac3sigma >> 2% (massive-activation phenomenon),
    which the gated criterion rejects.

    act_alpha override order: explicit arg > env TESSERAQ_DBAF_ACT_ALPHA > 0.25.
    The 0.25 default comes from an RTN+DBAF LLaMA-3-8B W4A4 alpha-sweep
    (alpha-sweep/llama3-8b/summary.json): wt2 PPL drops from 250.60 at
    alpha=0.75 to 16.31 at alpha=0.25, a 15x improvement.
    """
    import os
    if act_alpha is None:
        act_alpha = float(os.environ.get("TESSERAQ_DBAF_ACT_ALPHA", "0.25"))
    global _DBAF_ALPHA, _DBAF_INSTALLED
    if _DBAF_INSTALLED:
        print("[dbaf_patch] already installed; skipping", flush=True)
        return
    _DBAF_ALPHA = dbaf_alpha
    _act_alpha = act_alpha
    print(f"[dbaf_patch] dbaf_alpha(weights)={dbaf_alpha}  act_alpha={_act_alpha}",
          flush=True)

    from llmc.compression.quantization.quant import IntegerQuantizer

    orig_act = IntegerQuantizer.fake_quant_act_dynamic
    orig_w = IntegerQuantizer.fake_quant_weight_dynamic

    def wrapped_act(self, act, args=None):
        args = args if args is not None else {}
        if act_no_gate:
            sigma = float(act.detach().std())
            if sigma <= 0.0:
                return orig_act(self, act, args)
            T = 3.0 * sigma
            folded, tag = fold_outliers(act, T, _act_alpha)
            q_folded = orig_act(self, folded, args)
            return unfold_outliers(q_folded, tag, T, _act_alpha)
        try:
            stats = is_like_normal_plus_3sigma_outliers(act.detach())
            fires = bool(stats.get("is_like_c", False))
        except Exception:
            fires = False
        if not fires:
            return orig_act(self, act, args)
        T = float(3.0 * stats["stats"]["std"])
        folded, tag = fold_outliers(act, T, _act_alpha)
        q_folded = orig_act(self, folded, args)
        return unfold_outliers(q_folded, tag, T, _act_alpha)

    def wrapped_w(self, weight, args=None):
        args = args if args is not None else {}
        try:
            stats = is_like_normal_plus_3sigma_outliers(weight.detach())
            fires = bool(stats.get("is_like_c", False))
        except Exception:
            fires = False
        if not fires:
            return orig_w(self, weight, args)
        T = float(3.0 * stats["stats"]["std"])
        folded, tag = fold_outliers(weight, T, _DBAF_ALPHA)
        q_folded = orig_w(self, folded, args)
        return unfold_outliers(q_folded, tag, T, _DBAF_ALPHA)

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


# ---------------------------------------------------------------------------
# Trained PCSA inside TesseraQ block-recon (HM Phase E)
# ---------------------------------------------------------------------------
_PCSA_TRAINED_INSTALLED: bool = False


def _fit_per_linear_pcsa(self, block, K: int = 8, max_prompts: int = 64) -> dict:
    """One block forward with pre-hooks on each FakeQuantLinear, then per-linear
    fit via fit_pcsa_tf (uses the 99th-percentile fix already in pcsa_tf.py).

    Memory model: full block fwd on 512 calib prompts with float32 CPU copies
    of 7 Linear inputs is ~240 GB. We subsample to `max_prompts` *batches* by
    only keeping the first N hook firings per Linear (the hook order tracks the
    block_forward iteration order). K-means with K=8 is fully determined by
    ~64 descriptors; per-anchor 99%-tile scales are stable at that sample size.

    Returns dict name -> {"anchors": [K,D], "scales": [K]} on CPU float.
    """
    import torch
    import gc
    from flatquant.baselines.pcsa_tf import fit_pcsa_tf
    from llmc.compression.quantization.module_utils import FakeQuantLinear

    inputs: dict[str, list] = {}
    hooks = []

    def _make_pre_hook(name: str):
        def _h(module, args):
            bucket = inputs.setdefault(name, [])
            if len(bucket) >= max_prompts:
                return
            x = args[0]
            # CPU first, fp16 to halve memory; fit_pcsa_tf does .float() itself.
            bucket.append(x.detach().to(torch.float16).cpu())
        return _h

    for n, m in block.named_modules():
        if isinstance(m, FakeQuantLinear):
            hooks.append(m.register_forward_pre_hook(_make_pre_hook(n)))

    # Only forward through the first max_prompts prompts: hooks short-circuit
    # after that anyway, so processing all 512 is pure waste and (at block 31)
    # can OOM during _fit's CPU concat + kthvalue spike.
    n_prompts = min(max_prompts, len(self.input['data']))
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"[pcsa_trained] _fit pre-fwd block {self.block_idx} "
              f"cuda mem: {mem_before:.2f} GiB, n_prompts={n_prompts}",
              flush=True)
    _ = self.block_forward(block, input_data=self.input['data'][:n_prompts])
    for h in hooks:
        h.remove()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated() / 1024**3
        print(f"[pcsa_trained] _fit post-fwd block {self.block_idx} "
              f"cuda mem: {mem_after:.2f} GiB", flush=True)

    state: dict = {}
    # Process per-linear and drop intermediate tensors immediately to keep
    # CPU peak bounded (down_proj at block 31 spikes kthvalue temp ~7 GB).
    names = list(inputs.keys())
    for name in names:
        acts_list = inputs.pop(name)
        acts = torch.cat(acts_list, dim=0).float()  # [N_total, T, D] or [N,D]
        del acts_list
        if acts.dim() == 3:
            descs = acts.mean(dim=1)
        else:
            descs = acts
        state[name] = fit_pcsa_tf(descs, acts, K=K)
        del acts, descs
        gc.collect()
    print(f"[pcsa_trained] fitted per-linear PCSA for {len(state)} Linears "
          f"in block {self.block_idx} (max_prompts={max_prompts})", flush=True)
    return state


def install_pcsa_trained_patch(K: int = 8):
    """Make per-block PCSA scales trainable inside TesseraQ block-recon.

    Anchors stay frozen (descriptor-space k-means clusters); only the K-vector
    `scales` per FakeQuantLinear is added to Adam's `params_s` list and updated
    jointly with the weight-rounding params. PCSA-tf is applied BEFORE the
    standard `a_qdq` activation quantizer.

    Idempotent.
    """
    global _PCSA_TRAINED_INSTALLED
    if _PCSA_TRAINED_INSTALLED:
        print("[pcsa_trained] already installed; skipping", flush=True)
        return

    import torch
    import torch.nn as nn
    from flatquant.baselines.pcsa_tf import apply_pcsa_tf_to_activation_ste
    from llmc.compression.quantization.tesseraq import TesseraQ
    from llmc.compression.quantization.module_utils import FakeQuantLinear

    _orig_reg   = TesseraQ.register_rounding_parameters
    _orig_get   = TesseraQ.get_rounding_parameters
    _orig_block = TesseraQ.block_transform

    def _make_a_qdq_wrapper(m, orig_a_qdq):
        def _wrapped(x, self_mod):
            if hasattr(m, 'pcsa_scales') and hasattr(m, 'pcsa_anchors'):
                if x.dim() == 3:
                    desc = x.mean(dim=1).float()
                else:
                    desc = x.float()
                x = apply_pcsa_tf_to_activation_ste(
                    x, desc, m.pcsa_anchors, m.pcsa_scales, bits=4,
                )
            return orig_a_qdq(x, self_mod)
        return _wrapped

    def _new_block_transform(self, block, input_feat, block_kwargs):
        # We need to fit per-linear PCSA AFTER FakeQuantLinear replacement
        # (inside _new_reg below), so just stash K and defer.
        self._pcsa_K = K
        import gc
        if torch.cuda.is_available():
            mem_pre = torch.cuda.memory_allocated() / 1024**3
            print(f"[pcsa_trained] pre-block {getattr(self,'block_idx','?')} "
                  f"cuda mem: {mem_pre:.2f} GiB", flush=True)
        result = _orig_block(self, block, input_feat, block_kwargs)
        # Post-block cleanup: freeze this block's pcsa_scales (Parameter -> buffer)
        # so the STE forward in subsequent blocks' calibration prep can't
        # accumulate a backward graph that pins ~100 MB/layer of activation.
        from llmc.compression.quantization.module_utils import FakeQuantLinear
        frozen = 0
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear) and hasattr(m, 'pcsa_scales'):
                if isinstance(m.pcsa_scales, nn.Parameter):
                    data = m.pcsa_scales.data.detach().clone()
                    del m.pcsa_scales
                    m.register_buffer('pcsa_scales', data, persistent=False)
                    frozen += 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_post = torch.cuda.memory_allocated() / 1024**3
            print(f"[pcsa_trained] post-block {getattr(self,'block_idx','?')} "
                  f"cuda mem: {mem_post:.2f} GiB (froze {frozen} linears)",
                  flush=True)
        return result

    def _new_reg(self, block):
        _orig_reg(self, block)
        K_ = getattr(self, '_pcsa_K', K)
        try:
            state = _fit_per_linear_pcsa(self, block, K=K_, max_prompts=32)
        except Exception as exc:
            print(f"[pcsa_trained] WARNING: per-linear fit failed: {exc}", flush=True)
            return
        for n, m in block.named_modules():
            if not isinstance(m, FakeQuantLinear) or n not in state:
                continue
            anchors = state[n]['anchors'].cuda().float()
            scales  = state[n]['scales'].cuda().float()
            m.register_buffer('pcsa_anchors', anchors, persistent=False)
            m.pcsa_scales = nn.Parameter(scales.clone())
            m.register_buffer('pcsa_scales_init', scales.clone(), persistent=False)
            m.a_qdq = _make_a_qdq_wrapper(m, m.a_qdq)

    def _new_get(self, block):
        params_r, params_s = _orig_get(self, block)
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear) and hasattr(m, 'pcsa_scales'):
                params_s.append(m.pcsa_scales)
        return params_r, params_s

    TesseraQ.block_transform              = _new_block_transform
    TesseraQ.register_rounding_parameters = _new_reg
    TesseraQ.get_rounding_parameters      = _new_get
    _PCSA_TRAINED_INSTALLED = True
    print(f"[pcsa_trained] installed (K={K}); anchors frozen, scales trainable",
          flush=True)


def clamp_pcsa_scales(block, floor_frac: float = 1e-3):
    """Call after each optimizer.step() to keep `pcsa_scales` above `floor_frac *
    pcsa_scales_init` so division stays well-conditioned. Safe no-op if patch
    not installed."""
    from llmc.compression.quantization.module_utils import FakeQuantLinear
    for n, m in block.named_modules():
        if isinstance(m, FakeQuantLinear) and hasattr(m, 'pcsa_scales'):
            with torch.no_grad():
                floor = m.pcsa_scales_init * floor_frac
                m.pcsa_scales.data = torch.maximum(m.pcsa_scales.data, floor)
