"""Closed-form alpha* (Eq. alpha-star), shared by the cross-architecture sweeps.

alpha* = ( T * p_out / ((M - T) * p_in) )^(1/3)

with, per output row (DBAF applies T_l = 3 sigma per row):
    T     = 3 * std(row)            <- std of the SIGNED row, i.e. a true 3 sigma
    M     = quantile(|row|, 0.999)
    p_out = Pr(|row| > T),  p_in = 1 - p_out

Aggregated by median across sampled rows and Linear weights.

NOTE ON T: T must be computed from the signed row. Taking .abs() before .std()
yields std(|x|) ~ 0.6 sigma for a near-symmetric weight distribution, i.e. an
effective threshold of ~1.8 sigma rather than 3 sigma, which inflates p_out,
shrinks (M - T), and biases alpha* upward by roughly 1.5x. The same definition is
used by ahcptq/quantization/fake_quant.py::compute_T.
"""
import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def alpha_star_rows(w, n_rows=6, eps=1e-12):
    """alpha* values for evenly sampled rows of a 2D weight tensor."""
    vals = []
    w = w.float()
    for r in range(0, w.shape[0], max(1, w.shape[0] // n_rows)):
        row = w[r]
        T = 3.0 * row.std()                    # signed std -> true 3 sigma
        a = row.abs()
        M = torch.quantile(a, 0.999)
        p_out = (a > T).float().mean()
        p_in = 1.0 - p_out
        if p_out > 0 and M > T:
            v = torch.pow(T * p_out / ((M - T) * p_in + eps), 1 / 3.0)
            if torch.isfinite(v):
                vals.append(v.item())
    return vals


@torch.no_grad()
def alpha_star_model(model, skip=(), n_rows=6):
    """Median alpha* over the Linear weights of a model.

    Raises if no tensor yields a finite alpha* -- a silent numeric fallback would
    put an unbacked constant into a released result file.
    """
    vals = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or mod.weight.dim() != 2:
            continue
        if any(s in name for s in skip):
            continue
        vals += alpha_star_rows(mod.weight.data, n_rows=n_rows)
    if not vals:
        raise RuntimeError("alpha_star_model: no finite alpha* over any Linear weight")
    return float(np.median(vals)), len(vals)
