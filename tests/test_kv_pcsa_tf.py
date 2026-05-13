"""Unit tests for training-free KV-PCSA primitive."""
import pytest
import torch
import sys
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
from flatquant.baselines.kv_pcsa_tf import (
    fit_kv_pcsa_tf, quantize_k_with_kv_pcsa_tf, quantize_v_with_kv_pcsa_tf,
)


def test_fit_returns_k_and_v_scales_per_anchor():
    # 12 calibration prompts, 32-dim descriptors, 4 anchors,
    # each prompt has K cache [num_heads=4, seq=16, head_dim=8]
    N, K_anchors = 12, 4
    descs = torch.randn(N, 32)
    k_caches = [torch.randn(4, 16, 8) for _ in range(N)]
    v_caches = [torch.randn(4, 16, 8) for _ in range(N)]
    state = fit_kv_pcsa_tf(descs, k_caches, v_caches, K=K_anchors)
    assert state["anchors"].shape == (K_anchors, 32)
    assert state["k_scales"].shape == (K_anchors,)
    assert state["v_scales"].shape == (K_anchors,)


def test_quantize_k_uses_anchor_scale():
    state = {
        "anchors": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "k_scales": torch.tensor([5.0, 0.5]),
        "v_scales": torch.tensor([3.0, 0.3]),
    }
    k = torch.full((1, 2, 4, 4), 2.0)
    desc = torch.tensor([[1.0, 0.01]])
    out = quantize_k_with_kv_pcsa_tf(k, desc, state, bits=4)
    # Anchor 0 scale 5.0 / qmax 15 ≈ 0.33 per code; 2.0 / 0.33 ≈ 6 codes
    # Should reconstruct close to 2.0
    assert (out - k).abs().mean() < 1.0


def test_quantize_v_uses_v_scale_not_k_scale():
    state = {
        "anchors": torch.tensor([[1.0, 0.0]]),
        "k_scales": torch.tensor([5.0]),
        "v_scales": torch.tensor([50.0]),  # very loose v scale
    }
    v = torch.full((1, 1, 4, 4), 20.0)  # within v scale, outside k scale
    desc = torch.tensor([[1.0, 0.0]])
    out = quantize_v_with_kv_pcsa_tf(v, desc, state, bits=4)
    # With scale 50.0/15 ≈ 3.33, 20.0 / 3.33 ≈ 6 codes — should reconstruct close to 20
    assert (out - v).abs().mean() < 5.0
