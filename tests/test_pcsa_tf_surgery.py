"""PCSA-tf magnitude preservation under symmetric INT4 fake-quant.

Catches the qmax = 2**bits - 1 vs. 2**(bits-1) - 1 bug that compressed
activations to ~0.47x of original, yielding PPL 300K-1M after 32-layer
propagation.
"""
import sys
import torch

sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")


def test_pcsa_tf_preserves_magnitude_within_one_step():
    """For x at the per-prompt max-abs (i.e. saturation point), the fake-
    quantized output should be within one quantization step of x.

    For INT4 symmetric with S = max(|x|), the step is S/7 (not S/15).
    A correctly-implemented quantizer maps x=S -> q=7 -> dequant 7*(S/7) = S.
    The buggy implementation maps x=S -> q=7 -> dequant 7*(S/15) = 0.47*S.
    """
    from flatquant.baselines.pcsa_tf import apply_pcsa_tf_to_activation

    torch.manual_seed(0)
    B, T, D = 2, 8, 16
    x = torch.randn(B, T, D)
    # Per-prompt max-abs as the scale anchor (single anchor for simplicity)
    scale_per_prompt = x.abs().amax(dim=(1, 2))  # [B]
    state = {
        "anchors": torch.eye(B),         # B-by-B descriptor space, each prompt is its own anchor
        "scales":  scale_per_prompt,     # one anchor per prompt
    }
    desc = torch.eye(B)  # each row identifies a different anchor

    y = apply_pcsa_tf_to_activation(x, desc, state, bits=4)
    # Step size = S/7 per prompt
    step = scale_per_prompt / 7.0
    err = (y - x).abs()
    # Each element's |y - x| should be at most one step (S/7) plus float slop.
    for b in range(B):
        max_err = err[b].max()
        tol = step[b] * 1.1  # 10% slack for float rounding
        assert max_err <= tol, (
            f"prompt {b}: max |y - x| = {max_err:.4f} exceeds 1 step "
            f"{tol:.4f} (this is the qmax bug)"
        )


def test_pcsa_tf_routes_per_prompt():
    """Anchor-routing sanity: two prompts with different scales should
    pick different anchors and get different effective step sizes."""
    from flatquant.baselines.pcsa_tf import apply_pcsa_tf_to_activation

    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    x[0] *= 10.0   # prompt 0 has 10x bigger activations
    state = {
        "anchors": torch.eye(2),
        "scales":  torch.tensor([10.0 * x[0].abs().max() / 10.0,
                                  x[1].abs().max()]),
    }
    desc = torch.eye(2)
    y = apply_pcsa_tf_to_activation(x, desc, state, bits=4)
    # The output magnitudes should approximately match each prompt's input,
    # NOT be compressed to a single global scale.
    assert torch.allclose(y[0].abs().max(), x[0].abs().max(), rtol=0.2), \
        "high-magnitude prompt should preserve its magnitude"
    assert torch.allclose(y[1].abs().max(), x[1].abs().max(), rtol=0.2), \
        "low-magnitude prompt should preserve its magnitude"
