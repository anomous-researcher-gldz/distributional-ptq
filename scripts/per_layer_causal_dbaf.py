"""Per-layer causal test: intervene on outlier fraction, observe DBAF gain.

If DBAF gain is *caused* by outlier fraction (not just correlated), then
surgically clipping a layer's weight outliers to <3sigma BEFORE the gate
should eliminate the gain on that layer specifically.

Procedure on LLaMA-3-8B W4 weight-only:
  1. For each of 20 randomly-sampled Linear layers:
     a) Baseline RTN  (no DBAF) on that layer → MSE
     b) RTN+DBAF gated on that layer → MSE_dbaf
        DBAF gain = (MSE - MSE_dbaf) / MSE
     c) Pre-clip outliers (|w|>3σ → ±3σ), then re-run b
        Clipped DBAF gain = expected ≈ 0 if causation holds
        (because the gate now rejects the tensor, so no DBAF fires)
  2. Report: mean DBAF gain pre-intervention vs post-intervention across 20 layers.

If gain collapses to ~0 post-intervention, causation established.
"""
from __future__ import annotations
import sys, json, pathlib, argparse, random
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")


@torch.no_grad()
def rtn_per_row(w: torch.Tensor, bits: int = 4) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-9)
    q = torch.round(w / scale).clamp(-qmax, qmax)
    return (q * scale).to(w.dtype)


@torch.no_grad()
def fold_per_row(w: torch.Tensor, T_sigma: float = 3.0, alpha: float = 0.75) -> torch.Tensor:
    sigma = w.std(dim=1, keepdim=True).clamp(min=1e-9)
    T = T_sigma * sigma
    sgn = torch.sign(w)
    mask = w.abs() > T
    folded = sgn * T + alpha * (w - sgn * T)
    return torch.where(mask, folded, w), mask, T, sgn


@torch.no_grad()
def unfold_per_row(w_q: torch.Tensor, mask: torch.Tensor, T: torch.Tensor, sgn: torch.Tensor, alpha: float) -> torch.Tensor:
    delta = (w_q - sgn * T) / alpha
    unfolded = sgn * T + delta
    return torch.where(mask, unfolded, w_q)


def is_gate_pass(w: torch.Tensor) -> bool:
    """Gate check: skew ≤ 0.7, kurt in [3,30], frac>3sigma in [1e-4, 2e-2]."""
    x = w.float().flatten()
    mu = x.mean()
    sd = x.std()
    n = x.numel()
    z = (x - mu) / sd.clamp(min=1e-9)
    skew = ((z ** 3).mean()).abs().item()
    kurt = ((z ** 4).mean()).item() - 3  # excess kurtosis
    kurt_total = kurt + 3
    frac3 = (z.abs() > 3.0).float().mean().item()
    return (skew <= 0.7) and (3.0 <= kurt_total <= 30.0) and (1e-4 <= frac3 <= 2e-2)


@torch.no_grad()
def dbaf_eval(w: torch.Tensor, bits: int = 4, alpha: float = 0.75) -> dict:
    """Returns dict with rtn_mse, dbaf_gated_mse, gain_pct, gate_passed."""
    w_f = w.float()
    norm = (w_f ** 2).mean().clamp(min=1e-12)
    # RTN
    w_rtn = rtn_per_row(w_f, bits=bits)
    rtn_mse = ((w_f - w_rtn) ** 2).mean()
    # Gate
    gate = is_gate_pass(w_f)
    # DBAF (gated): only fold if gate passes
    if gate:
        w_folded, mask, T, sgn = fold_per_row(w_f, alpha=alpha)
        w_q = rtn_per_row(w_folded, bits=bits)
        w_unfolded = unfold_per_row(w_q, mask, T, sgn, alpha)
        dbaf_mse = ((w_f - w_unfolded) ** 2).mean()
    else:
        dbaf_mse = rtn_mse
    gain_pct = 100.0 * (rtn_mse - dbaf_mse) / rtn_mse.clamp(min=1e-12)
    return {
        "rtn_mse": float(rtn_mse.item()),
        "dbaf_mse": float(dbaf_mse.item()),
        "gain_pct": float(gain_pct.item()),
        "gate": bool(gate),
    }


@torch.no_grad()
def clip_outliers(w: torch.Tensor, T_sigma: float = 3.0) -> torch.Tensor:
    """Surgical intervention: clamp |w|>T to ±T."""
    w_f = w.float()
    sigma = w_f.std(dim=1, keepdim=True).clamp(min=1e-9)
    T = T_sigma * sigma
    return torch.clamp(w_f, min=-T, max=T)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_layers", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/home/ubuntu/unifying-ptq/results/per_layer_causal.json")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "/data/modelzoo/meta-llama/Meta-Llama-3-8B",
        torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    # Collect Linear layer weights (keep on CPU; per-layer MSEs are tiny)
    linears = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            linears.append((name, mod.weight.detach().cpu()))

    rng = random.Random(args.seed)
    chosen = rng.sample(linears, k=args.n_layers)
    print(f"[per-layer-causal] sampled {len(chosen)} of {len(linears)} Linears", flush=True)

    rows = []
    for name, w in chosen:
        pre = dbaf_eval(w, alpha=args.alpha)
        w_intervened = clip_outliers(w)
        post = dbaf_eval(w_intervened, alpha=args.alpha)
        rows.append({
            "layer": name,
            "shape": list(w.shape),
            "pre_gate": pre["gate"],
            "pre_rtn_mse": pre["rtn_mse"],
            "pre_dbaf_mse": pre["dbaf_mse"],
            "pre_gain_pct": pre["gain_pct"],
            "post_gate": post["gate"],
            "post_rtn_mse": post["rtn_mse"],
            "post_dbaf_mse": post["dbaf_mse"],
            "post_gain_pct": post["gain_pct"],
        })
        print(f"  {name:60s}  pre: gate={pre['gate']} gain={pre['gain_pct']:5.2f}%  "
              f"post: gate={post['gate']} gain={post['gain_pct']:5.2f}%", flush=True)

    pre_gain = np.mean([r["pre_gain_pct"] for r in rows if r["pre_gate"]])
    post_gain = np.mean([r["post_gain_pct"] for r in rows if r["post_gate"]])
    n_pre_gated = sum(1 for r in rows if r["pre_gate"])
    n_post_gated = sum(1 for r in rows if r["post_gate"])
    summary = {
        "n_layers": len(rows),
        "n_pre_gate_pass": n_pre_gated,
        "n_post_gate_pass": n_post_gated,
        "mean_pre_gain_pct": float(pre_gain) if n_pre_gated else None,
        "mean_post_gain_pct": float(post_gain) if n_post_gated else None,
        "alpha": args.alpha,
    }
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  Pre-intervention:  {n_pre_gated}/{len(rows)} gate-pass, mean DBAF gain = {pre_gain:.3f}%", flush=True)
    print(f"  Post-intervention: {n_post_gated}/{len(rows)} gate-pass, mean DBAF gain = {post_gain if n_post_gated else 0.0:.3f}%", flush=True)

    out = {"summary": summary, "layers": rows}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
