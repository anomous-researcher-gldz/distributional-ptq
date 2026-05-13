"""Unit tests for training-free PCSA primitive."""
import pytest
import torch
import sys
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
from flatquant.baselines.pcsa_tf import (
    fit_pcsa_tf, route_pcsa_tf, apply_pcsa_tf_to_activation,
)


def test_fit_returns_anchors_and_scales():
    # 16 calibration prompts, 64-dim descriptors, 4 anchors
    descs = torch.randn(16, 64)
    acts = torch.randn(16, 4, 8)  # 4 tokens, 8 channels per prompt
    state = fit_pcsa_tf(descs, acts, K=4)
    assert "anchors" in state
    assert state["anchors"].shape == (4, 64)
    assert "scales" in state
    assert state["scales"].shape == (4,)


def test_route_assigns_correct_anchor():
    descs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    acts = torch.ones(4, 1, 4)
    state = fit_pcsa_tf(descs, acts, K=4)
    # Query a descriptor exactly matching anchor 0
    q = state["anchors"][0:1]
    idx = route_pcsa_tf(q, state)
    assert idx.item() == 0


def test_apply_pcsa_uses_anchor_scale():
    # Build a state where anchor 0 has scale=10, anchor 1 has scale=0.1
    state = {
        "anchors": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "scales": torch.tensor([10.0, 0.1]),
    }
    x = torch.full((1, 4, 2), 5.0)
    desc = torch.tensor([[1.0, 0.01]])  # routes to anchor 0
    out = apply_pcsa_tf_to_activation(x, desc, state, bits=4)
    # With scale 10 and INT4 asym, 5.0 fits easily
    assert torch.allclose(out, x, atol=1.0)


def test_pcsa_tf_no_gradient():
    descs = torch.randn(8, 32, requires_grad=True)
    acts = torch.randn(8, 4, 16, requires_grad=True)
    state = fit_pcsa_tf(descs, acts, K=2)
    # Anchors and scales should NOT carry gradients
    assert state["anchors"].requires_grad is False
    assert state["scales"].requires_grad is False
