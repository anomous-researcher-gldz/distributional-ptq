"""Unit tests for KVPCSAQuantizer."""
import torch
import pytest


@pytest.fixture
def kv():
    from flatquant.kv_pcsa import KVPCSAQuantizer
    q = KVPCSAQuantizer(num_anchors=4, descriptor_dim=128, k_bits=4, v_bits=4)
    return q.cuda()


def test_set_prompt_picks_anchor(kv):
    idx = kv.set_prompt(torch.randn(1, 128).cuda())
    assert 0 <= idx < 4


def test_quantize_k_round_trip(kv):
    kv.set_prompt(torch.randn(1, 128).cuda())
    # Update scale so quantization has signal
    k = torch.randn(1, 8, 64, 32, dtype=torch.float16, device="cuda") * 2.0
    kv.update_scales(k, k)
    kq = kv.quantize_k(k)
    assert kq.shape == k.shape
    assert (k - kq).abs().mean() < 0.5


def test_anchor_routing_deterministic(kv):
    desc = torch.randn(1, 128).cuda()
    i1 = kv.set_prompt(desc)
    i2 = kv.set_prompt(desc)
    assert i1 == i2


def test_calibrate_updates_anchors_ema(kv):
    a0 = kv.anchors.detach().clone()
    descs = torch.randn(100, 128).cuda()
    for d in descs:
        kv.calibrate_step(d.unsqueeze(0))
    a1 = kv.anchors.detach().clone()
    assert (a0 - a1).abs().mean() > 1e-4, "anchors didn't move during calibration"


def test_descriptor_from_hidden_states_3d(kv):
    """When given [B, T, D] hidden states, set_prompt should mean-pool to [B, D]."""
    h = torch.randn(2, 16, 128).cuda()
    idx = kv.set_prompt(h)
    assert 0 <= idx < 4
