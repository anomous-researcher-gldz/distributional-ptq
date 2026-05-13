"""Training-free KV-PCSA: k-means on calibration prompt descriptors + per-anchor
max-abs K and V cache scales. No gradient training. LLM-specific.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from flatquant.baselines.pcsa_tf import _kmeans, route_pcsa_tf


@torch.no_grad()
def fit_kv_pcsa_tf(
    descs: torch.Tensor,
    k_caches: list[torch.Tensor],
    v_caches: list[torch.Tensor],
    K: int = 4,
) -> dict:
    """Fit K anchors via k-means on `descs`; per-anchor max-abs over K and V.

    descs: [N, D] prompt descriptors
    k_caches/v_caches: list of N tensors, each shape [num_heads, seq, head_dim]
                       or any shape (we max-abs all dims)
    K: number of anchors
    """
    descs = descs.detach().float()
    anchors = _kmeans(descs, K)
    sims = F.normalize(descs, dim=-1) @ F.normalize(anchors, dim=-1).T
    assign = sims.argmax(dim=-1)  # [N]
    k_scales = torch.zeros(K)
    v_scales = torch.zeros(K)
    k_max = torch.tensor([kc.abs().max() for kc in k_caches])
    v_max = torch.tensor([vc.abs().max() for vc in v_caches])
    for j in range(K):
        mask = (assign == j)
        k_scales[j] = k_max[mask].max() if mask.any() else k_max.max()
        v_scales[j] = v_max[mask].max() if mask.any() else v_max.max()
    return {"anchors": anchors, "k_scales": k_scales, "v_scales": v_scales}


@torch.no_grad()
def _quantize_with_scale(x: torch.Tensor, scale_per_prompt: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** bits - 1
    extra_dims = x.dim() - 1
    scale = scale_per_prompt.view(-1, *([1] * extra_dims)) / qmax
    scale = scale.clamp(min=1e-9)
    q = torch.round(x / scale).clamp(-qmax // 2, qmax // 2)
    return (q * scale).to(x.dtype)


@torch.no_grad()
def quantize_k_with_kv_pcsa_tf(k: torch.Tensor, desc: torch.Tensor, state: dict, bits: int = 4) -> torch.Tensor:
    """Per-prompt symmetric INT[bits] fake-quant on K cache using anchor-routed scale."""
    anchor_ids = route_pcsa_tf(desc, state)
    return _quantize_with_scale(k, state["k_scales"][anchor_ids], bits)


@torch.no_grad()
def quantize_v_with_kv_pcsa_tf(v: torch.Tensor, desc: torch.Tensor, state: dict, bits: int = 4) -> torch.Tensor:
    """Per-prompt symmetric INT[bits] fake-quant on V cache using anchor-routed scale."""
    anchor_ids = route_pcsa_tf(desc, state)
    return _quantize_with_scale(v, state["v_scales"][anchor_ids], bits)
