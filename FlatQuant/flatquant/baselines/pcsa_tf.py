"""Training-free PCSA: k-means on calibration prompt descriptors + per-anchor
max-abs activation scales. No gradient training. Composes on any host method
that has activation tensors.

API:
  fit_pcsa_tf(descs, acts, K) -> state dict {anchors, scales}
  route_pcsa_tf(desc, state) -> anchor_id tensor
  apply_pcsa_tf_to_activation(x, desc, state, bits) -> fake-quantized x
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def _kmeans(x: torch.Tensor, k: int, n_iter: int = 25) -> torch.Tensor:
    """k-means on [N, D] -> [k, D] centroids. CPU-friendly, no gradients."""
    x = x.detach()
    N, D = x.shape
    # init from random rows; clamp k if we have fewer descriptors than clusters
    k = min(k, N)
    idx = torch.randperm(N)[:k]
    centroids = x[idx].clone()
    for _ in range(n_iter):
        # assign each row to nearest centroid (cosine on L2-normalized vectors)
        xn = F.normalize(x, dim=-1)
        cn = F.normalize(centroids, dim=-1)
        sims = xn @ cn.T
        assign = sims.argmax(dim=-1)
        # update centroids = mean of assigned rows
        new_cents = torch.zeros_like(centroids)
        for j in range(k):
            mask = (assign == j)
            if mask.any():
                new_cents[j] = x[mask].mean(dim=0)
            else:
                new_cents[j] = centroids[j]
        if torch.allclose(new_cents, centroids, atol=1e-6):
            break
        centroids = new_cents
    return centroids


@torch.no_grad()
def fit_pcsa_tf(
    descs: torch.Tensor,
    acts: torch.Tensor,
    K: int = 8,
) -> dict:
    """Fit K anchors via k-means on `descs`, then per-anchor max-abs scale on `acts`.

    Args:
      descs: [N, D_desc] prompt-level descriptors (e.g., mean-pooled hidden states)
      acts:  [N, T, D_act] or [N, D_act] activations per prompt; if 3D, max over T
      K: number of anchors

    Returns dict {"anchors": [K, D_desc], "scales": [K]} (both no grad).
    """
    descs = descs.detach().float()
    acts = acts.detach().float()
    # Robust per-prompt scale: 99th percentile of |acts|, not max.
    # Max-abs is dominated by outliers in LLM deep-layer residual streams
    # (e.g., scales[31]~290 with median ~1.0); using it as the quant range
    # forces step = 290/7 = 41, which zeroes most token values.
    if acts.dim() == 3:
        flat = acts.abs().reshape(acts.shape[0], -1)  # [N, T*D]
    elif acts.dim() == 2:
        flat = acts.abs()  # [N, D]
    else:
        raise ValueError(f"acts must be [N,T,D] or [N,D], got {acts.shape}")
    # torch.quantile() crashes when a row exceeds ~16M elements (LLaMA-3-8B
    # down_proj at seq_len=2048: 29M). Use kthvalue, which handles arbitrary
    # row sizes, to get the 99th-percentile per row.
    M = flat.shape[1]
    k99 = max(1, int(0.99 * M))
    per_prompt_scale = flat.kthvalue(k99, dim=1).values  # [N]
    anchors = _kmeans(descs, K)
    K_actual = anchors.shape[0]  # may be < K if N < K (clamped in _kmeans)
    sims = F.normalize(descs, dim=-1) @ F.normalize(anchors, dim=-1).T
    assign = sims.argmax(dim=-1)  # [N]
    scales = torch.zeros(K_actual)
    for j in range(K_actual):
        mask = (assign == j)
        scales[j] = per_prompt_scale[mask].max() if mask.any() else per_prompt_scale.max()
    return {"anchors": anchors, "scales": scales}


@torch.no_grad()
def route_pcsa_tf(desc: torch.Tensor, state: dict) -> torch.Tensor:
    """desc: [B, D]; returns [B] anchor indices."""
    sims = F.normalize(desc, dim=-1) @ F.normalize(state["anchors"], dim=-1).T
    return sims.argmax(dim=-1)


def apply_pcsa_tf_to_activation_ste(
    x: torch.Tensor,
    desc: torch.Tensor,
    anchors: torch.Tensor,
    scales: torch.Tensor,
    bits: int = 4,
) -> torch.Tensor:
    """Differentiable variant of apply_pcsa_tf_to_activation for TesseraQ block-recon.

    Identical semantics to apply_pcsa_tf_to_activation but `scales` is a trainable
    nn.Parameter and `anchors` is a frozen buffer. Uses straight-through round so
    gradients flow back into `scales`. The clamp + round combination supplies
    gradient through clipped values: for |x/s| > qmax, dy/ds = +/-qmax, which
    correctly pushes `scales` up when an outlier needs to be preserved.

    Matches TesseraQ's STE: round_func(x) = (x.round() - x).detach() + x
    (see TesseraQ/llmc/compression/quantization/quant.py:23).
    """
    qmax = 2 ** (bits - 1) - 1
    # Routing (no grad needed — anchor selection is discrete)
    with torch.no_grad():
        ids = (F.normalize(desc, dim=-1) @ F.normalize(anchors, dim=-1).T).argmax(dim=-1)
    scale_per_prompt = scales[ids].clamp(min=1e-9)
    extra_dims = x.dim() - 1
    s = (scale_per_prompt / qmax).view(-1, *([1] * extra_dims))
    y = x / s
    q = (y.round() - y).detach() + y  # STE
    q = q.clamp(-qmax, qmax)
    return q * s


@torch.no_grad()
def apply_pcsa_tf_to_activation(
    x: torch.Tensor,
    desc: torch.Tensor,
    state: dict,
    bits: int = 4,
    use_dbaf: bool = False,
    dbaf_alpha: float = 0.75,
    dbaf_T_sigma: float = 3.0,
) -> torch.Tensor:
    """Per-prompt symmetric INT[bits] fake-quantization using anchor-routed scale.

    Symmetric int{bits}: integer codes span [-(2^(bits-1)-1), +(2^(bits-1)-1)],
    so step = S / (2^(bits-1) - 1). For bits=4 the step is S/7 (NOT S/15).

    If use_dbaf=True, applies dual-band affine fold along the per-token dim
    BEFORE the per-prompt quant, then unfolds AFTER. The fold formula:
        folded = sgn(x)*T + alpha*(x - sgn(x)*T)   for |x| > T
        folded = x                                  otherwise
    with T = dbaf_T_sigma * per-token std. This compresses outliers without
    disturbing bulk values; on outlier-heavy LLM residual streams it lets
    PCSA-tf scale on a dense post-fold distribution rather than fighting
    the bulk/outlier 100x+ ratio.
    """
    qmax = 2 ** (bits - 1) - 1
    if use_dbaf:
        sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-9)
        T = dbaf_T_sigma * sigma
        sgn = torch.sign(x)
        mask = x.abs() > T
        x_in = torch.where(mask, sgn * T + dbaf_alpha * (x - sgn * T), x)
    else:
        x_in = x
        T = None
        sgn = None
        mask = None
    anchor_ids = route_pcsa_tf(desc, state)
    scale_per_prompt = state["scales"][anchor_ids]
    extra_dims = x_in.dim() - 1
    scale = scale_per_prompt.view(-1, *([1] * extra_dims)) / qmax
    scale = scale.clamp(min=1e-9)
    q = torch.round(x_in / scale).clamp(-qmax, qmax)
    x_q = (q * scale).to(x.dtype)
    if use_dbaf:
        x_q = torch.where(mask, sgn * T + (1.0 / dbaf_alpha) * (x_q - sgn * T), x_q)
    return x_q
