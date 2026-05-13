"""Per-layer outlier-vs-DBAF-gain correlation analysis on LLaMA-3-8B.

For each nn.Linear in the model:
  - outlier_fraction = |w|>3sigma_per_row, mean across rows
  - rtn_mse = MSE(RTN(w), w)
  - dbaf_mse = MSE(RTN+DBAF(w), w)
  - gain_pct = (rtn_mse - dbaf_mse) / rtn_mse * 100

Plot outlier_fraction vs gain_pct. Strong positive correlation = DBAF gain
is specifically driven by outlier prevalence, not generic dynamic range
reduction.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf


@torch.no_grad()
def analyze(model_path: str, bits: int = 4, alpha: float = 0.75, T_sigma: float = 3.0):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cuda", low_cpu_mem_usage=True
    )
    rows = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "lm_head" in name:
            continue
        w = mod.weight.data.float()
        # Per-row outlier fraction
        sigma = w.std(dim=1, keepdim=True)
        T = T_sigma * sigma
        outlier_mask = w.abs() > T
        out_frac = outlier_mask.float().mean().item()
        # MSE under both quantization paths
        w_q_rtn = _quantize_tensor_uniform(w, bits, per_channel=True)
        w_q_dbaf = _quantize_per_channel_with_dbaf(w, bits, alpha=alpha)
        rtn_mse = ((w - w_q_rtn) ** 2).mean().item()
        dbaf_mse = ((w - w_q_dbaf) ** 2).mean().item()
        gain = (rtn_mse - dbaf_mse) / max(rtn_mse, 1e-12) * 100
        rows.append({"layer": name, "out_frac": out_frac, "rtn_mse": rtn_mse,
                     "dbaf_mse": dbaf_mse, "gain_pct": gain,
                     "shape": list(w.shape)})
        if len(rows) % 50 == 0:
            print(f"[{len(rows)}] {name}: out_frac={out_frac:.4f}, gain={gain:.2f}%", flush=True)
    return rows


def plot(rows, out_path):
    xs = [r["out_frac"] for r in rows]
    ys = [r["gain_pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(xs, ys, alpha=0.6, s=18, edgecolor="black", linewidth=0.5)
    # Fit a regression
    import numpy as np
    arr_x = np.array(xs); arr_y = np.array(ys)
    if arr_x.std() > 0:
        coef = np.polyfit(arr_x, arr_y, 1)
        line_x = np.linspace(arr_x.min(), arr_x.max(), 100)
        ax.plot(line_x, np.polyval(coef, line_x), 'r--', linewidth=1, label=f"linear fit (slope={coef[0]:.1f})")
    # Correlation
    if arr_x.std() > 0 and arr_y.std() > 0:
        corr = float(np.corrcoef(arr_x, arr_y)[0, 1])
        ax.set_title(f"Per-layer DBAF gain vs outlier fraction (LLaMA-3-8B, W4); Pearson r={corr:.3f}")
    ax.set_xlabel("Per-row outlier fraction (|x|>3σ)")
    ax.set_ylabel("DBAF MSE reduction (%) vs RTN")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/data/modelzoo/meta-llama/Meta-Llama-3-8B")
    p.add_argument("--out-json", default="/home/ubuntu/unifying-ptq/results/S4-dbaf-weak/per_layer_correlation/llama3-8b.json")
    p.add_argument("--out-fig", default="/home/ubuntu/paper/emnlp2026/figures/per_layer_outlier_correlation.pdf")
    args = p.parse_args()

    rows = analyze(args.model_path)
    pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_json).write_text(json.dumps(rows, indent=2))
    pathlib.Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    plot(rows, args.out_fig)
    print(f"Wrote {args.out_json} + figure {args.out_fig}")


if __name__ == "__main__":
    main()
