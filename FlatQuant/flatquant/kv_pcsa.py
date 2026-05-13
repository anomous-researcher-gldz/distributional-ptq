"""Per-prompt KV-cache scale routing via PCSA.

Compute a prompt descriptor once when the prompt arrives, route to one of
K anchors, and use the per-anchor K/V quantization scale for the full
generation. Calibration uses EMA on descriptors (mirrors original PCSA).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class KVPCSAQuantizer(nn.Module):
    """Per-prompt KV-cache scale routing.

    Maintains K anchor descriptor vectors and per-anchor (K, V) scales.
    At prompt time, the input's mean-pooled hidden state is matched to
    the nearest anchor via cosine similarity, freezing the (K, V) scales
    for the duration of generation.
    """

    def __init__(
        self,
        num_anchors: int = 4,
        descriptor_dim: int = 4096,
        k_bits: int = 4,
        v_bits: int = 4,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.K = num_anchors
        self.D = descriptor_dim
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.momentum = momentum
        anchors = torch.randn(num_anchors, descriptor_dim)
        anchors = anchors / anchors.norm(dim=1, keepdim=True).clamp(min=1e-6)
        self.register_buffer("anchors", anchors)
        self.register_buffer("k_scales", torch.ones(num_anchors))
        self.register_buffer("v_scales", torch.ones(num_anchors))
        self.register_buffer("counts", torch.zeros(num_anchors))
        self.current_anchor = 0

    # ----- routing -----
    def _descriptor(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> normalized [B, D]."""
        d = hidden_states.mean(dim=1)
        return d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)

    def set_prompt(self, descriptor_or_hidden: torch.Tensor) -> int:
        """Called once per prompt. Returns the selected anchor index."""
        if descriptor_or_hidden.dim() == 3:
            d = self._descriptor(descriptor_or_hidden)
        else:
            d = descriptor_or_hidden
            d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)
        sims = d @ self.anchors.T  # [B, K]
        idx = int(sims.argmax(dim=1)[0].item())
        self.current_anchor = idx
        return idx

    # ----- calibration -----
    def calibrate_step(self, descriptor: torch.Tensor):
        d = descriptor / descriptor.norm(dim=1, keepdim=True).clamp(min=1e-6)
        d = d.squeeze(0)
        sims = d @ self.anchors.T
        idx = int(sims.argmax().item())
        self.anchors[idx] = self.momentum * self.anchors[idx] + (1 - self.momentum) * d
        self.anchors[idx] = self.anchors[idx] / self.anchors[idx].norm().clamp(min=1e-6)
        self.counts[idx] += 1

    def update_scales(self, k: torch.Tensor, v: torch.Tensor):
        """Call during calibration after set_prompt to track per-anchor scales."""
        with torch.no_grad():
            new_k = k.abs().max()
            new_v = v.abs().max()
            i = self.current_anchor
            self.k_scales[i] = self.momentum * self.k_scales[i] + (1 - self.momentum) * new_k
            self.v_scales[i] = self.momentum * self.v_scales[i] + (1 - self.momentum) * new_v

    # ----- quantization -----
    def _quant(self, x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
        qmax = 2 ** (bits - 1) - 1
        s = scale / qmax
        if s.item() == 0:
            return x
        q = torch.round(x / s).clamp(-qmax, qmax)
        return (q * s).to(x.dtype)

    def quantize_k(self, k: torch.Tensor) -> torch.Tensor:
        return self._quant(k, self.k_scales[self.current_anchor], self.k_bits)

    def quantize_v(self, v: torch.Tensor) -> torch.Tensor:
        return self._quant(v, self.v_scales[self.current_anchor], self.v_bits)
