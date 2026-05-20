"""Per-token activation quantization wrappers for W4A4 sweep at Phase A.

Plain `_ActQuantWrapper` mirrors SmoothQuant's `_ActDivideWrapper` minus the
smooth divide. `_ActDBAFQuantWrapper` adds dual-band affine folding on the
activation before per-token quant (matching the DBAF semantics used in the
QuaRot/TesseraQ ActQuantizer wraps).
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _quantize_per_token(x: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return (q * scale).to(x.dtype)


def _dbaf_fold_per_token(x: torch.Tensor, alpha: float, T_sigma: float = 3.0):
    sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-9)
    T = T_sigma * sigma
    sgn = torch.sign(x)
    mask = x.abs() > T
    folded = torch.where(mask, sgn * T + alpha * (x - sgn * T), x)
    return folded, T, mask, sgn


def _dbaf_unfold_per_token(x_q: torch.Tensor, T: torch.Tensor, mask: torch.Tensor,
                            sgn: torch.Tensor, alpha: float) -> torch.Tensor:
    unfolded = torch.where(mask, sgn * T + (1.0 / alpha) * (x_q - sgn * T), x_q)
    return unfolded


class _ActQuantWrapper(nn.Module):
    """Per-token symmetric act quant before Linear matmul."""
    def __init__(self, linear: nn.Linear, bits: int):
        super().__init__()
        self.linear = linear
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(_quantize_per_token(x, self.bits))


class _ActDBAFQuantWrapper(nn.Module):
    """DBAF fold -> per-token quant -> unfold, before Linear matmul."""
    def __init__(self, linear: nn.Linear, bits: int, alpha: float = 0.75,
                 T_sigma: float = 3.0):
        super().__init__()
        self.linear = linear
        self.bits = bits
        self.alpha = alpha
        self.T_sigma = T_sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        folded, T, mask, sgn = _dbaf_fold_per_token(x, self.alpha, self.T_sigma)
        q = _quantize_per_token(folded, self.bits)
        return self.linear(_dbaf_unfold_per_token(q, T, mask, sgn, self.alpha))


def apply_w4a4_act_quant(model: nn.Module, bits: int = 4, use_dbaf: bool = False,
                         alpha: float = 0.75, T_sigma: float = 3.0) -> nn.Module:
    """Wrap every nn.Linear (except lm_head) with an act-quant wrapper.

    Idempotent: skips modules already wrapped.
    """
    name_to_module = dict(model.named_modules())
    n_wrapped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if "lm_head" in name:
            continue
        if isinstance(module, (_ActQuantWrapper, _ActDBAFQuantWrapper)):
            continue
        if use_dbaf:
            new_mod = _ActDBAFQuantWrapper(module, bits=bits, alpha=alpha,
                                           T_sigma=T_sigma)
        else:
            new_mod = _ActQuantWrapper(module, bits=bits)
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = name_to_module[parent_name] if parent_name else model
        setattr(parent, child_name, new_mod)
        n_wrapped += 1
    print(f"[act_quant] wrapped {n_wrapped} Linears (bits={bits}, dbaf={use_dbaf})",
          flush=True)
    return model
