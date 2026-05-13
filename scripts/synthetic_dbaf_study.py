"""Synthetic outlier study to isolate DBAF's causal mechanism.

Generate Gaussian tensors with controlled outlier injection.
Vary outlier fraction (0%, 0.1%, 1%, 5%, 10%).
Vary outlier magnitude (5*sigma, 10*sigma, 20*sigma).
For each (fraction, magnitude):
  - Compute MSE of RTN W4 quantization (baseline)
  - Compute MSE of RTN+DBAF W4 quantization (with folding)
  - Plot DBAF gain vs outlier fraction

Expected: DBAF gain monotonically increases with outlier fraction.
At 0% outliers, DBAF gain ≈ 0 (negative control).

Output: results/S4-dbaf-weak/synthetic/study.json + figures/synthetic_outlier_gain.pdf
"""
from __future__ import annotations
import json
import pathlib
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf


def make_tensor(n: int = 4096, m: int = 4096, outlier_fraction: float = 0.0, outlier_magnitude: float = 10.0, seed: int = 0) -> torch.Tensor:
    """Generate [n, m] Gaussian tensor with controlled outliers."""
    rng = torch.Generator().manual_seed(seed)
    w = torch.randn(n, m, generator=rng)
    sigma = w.std().item()
    if outlier_fraction > 0:
        n_outliers = int(w.numel() * outlier_fraction)
        flat = w.flatten()
        idx = torch.randperm(flat.numel(), generator=rng)[:n_outliers]
        signs = (torch.rand(n_outliers, generator=rng) > 0.5).float() * 2 - 1
        flat[idx] = signs * outlier_magnitude * sigma
        w = flat.view(n, m)
    return w


def mse(w: torch.Tensor, w_hat: torch.Tensor) -> float:
    return ((w - w_hat) ** 2).mean().item()


def run_one(w: torch.Tensor, bits: int, use_dbaf: bool, alpha: float = 0.75) -> float:
    if use_dbaf:
        w_q = _quantize_per_channel_with_dbaf(w, bits, alpha=alpha)
    else:
        w_q = _quantize_tensor_uniform(w, bits, per_channel=True)
    return mse(w, w_q)


def main():
    fractions = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    magnitudes = [5.0, 10.0, 20.0]
    bits = 4
    alpha = 0.75
    results = []
    for mag in magnitudes:
        for frac in fractions:
            w = make_tensor(outlier_fraction=frac, outlier_magnitude=mag, seed=0)
            mse_baseline = run_one(w, bits, use_dbaf=False)
            mse_dbaf = run_one(w, bits, use_dbaf=True, alpha=alpha)
            gain = (mse_baseline - mse_dbaf) / mse_baseline * 100
            results.append({
                "outlier_fraction": frac,
                "outlier_magnitude_sigma": mag,
                "mse_rtn": mse_baseline,
                "mse_rtn_dbaf": mse_dbaf,
                "pct_mse_reduction": gain,
            })
            print(f"frac={frac:.3f}, mag={mag}σ: RTN MSE={mse_baseline:.6f}, +DBAF MSE={mse_dbaf:.6f}, reduction={gain:.2f}%")

    out_dir = pathlib.Path("/home/ubuntu/unifying-ptq/results/S4-dbaf-weak/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "study.json").write_text(json.dumps(results, indent=2))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    for mag in magnitudes:
        xs = [r["outlier_fraction"] for r in results if r["outlier_magnitude_sigma"] == mag]
        ys = [r["pct_mse_reduction"] for r in results if r["outlier_magnitude_sigma"] == mag]
        ax.plot(xs, ys, marker="o", label=f"outliers at {int(mag)}σ")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Outlier fraction")
    ax.set_ylabel("MSE reduction from DBAF (%)")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_title("DBAF gain scales with outlier prevalence (W4, synthetic Gaussian)")
    ax.legend()
    plt.tight_layout()
    fig.savefig("/home/ubuntu/paper/emnlp2026/figures/synthetic_outlier_gain.pdf")
    fig.savefig(out_dir / "synthetic_outlier_gain.pdf")
    print(f"Wrote {out_dir/'study.json'} and figure.")


if __name__ == "__main__":
    main()
