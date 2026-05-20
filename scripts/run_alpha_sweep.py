"""DBAF α-sensitivity sweep on RTN+DBAF Llama-3-8B W4A4.

Sweeps α ∈ {0.25, 0.5, 0.75, 0.95} and records wt2 + c4 PPL per cell.
RTN is the fastest LLM host, so this gives a quick sensitivity curve
that generalizes the choice for other hosts in §4.

Reuses run_training_free_full_table.py helpers for model loading and PPL
eval. Overrides α via the RTN baseline's quantize_model(alpha=...) and
the act_quant.apply_w4a4_act_quant(alpha=...).

Usage:
    python scripts/run_alpha_sweep.py --target llama3-8b \\
        --alphas 0.25,0.5,0.75,0.95 \\
        --out_root /data/outputs/alpha-sweep
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

# Repo paths
_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "FlatQuant"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="DBAF α-sensitivity sweep on RTN+DBAF.")
    p.add_argument("--target", default="llama3-8b",
                   help="Model target (default: llama3-8b).")
    p.add_argument("--alphas", default="0.25,0.5,0.75,0.95",
                   help="Comma-separated α values to sweep.")
    p.add_argument("--out_root", default="/data/outputs/alpha-sweep",
                   help="Output root.")
    p.add_argument("--skip_c4", action="store_true",
                   help="Skip C4 eval (saves ~3 min/cell).")
    return p.parse_args(argv)


def _run_one(target: str, alpha: float, out_path: pathlib.Path, skip_c4: bool):
    import torch
    import gc
    from flatquant.baselines.rtn import quantize_model as rtn_quantize_model
    from flatquant.baselines.act_quant import apply_w4a4_act_quant
    from run_training_free_full_table import (
        _load_llm, _eval_ppl_wikitext2, _eval_ppl_c4,
    )

    if out_path.exists():
        existing = json.loads(out_path.read_text())
        print(f"[alpha-sweep] skip α={alpha} (exists): {existing['metrics']}",
              flush=True)
        return existing

    print(f"[alpha-sweep] === α={alpha} ===", flush=True)
    t0 = time.time()
    model, tok = _load_llm(target)
    model = rtn_quantize_model(model, bits=4, use_dbaf=True, alpha=alpha)
    model = apply_w4a4_act_quant(model, bits=4, use_dbaf=True, alpha=alpha)

    print("[alpha-sweep] evaluating WikiText-2 PPL ...", flush=True)
    wt2_ppl = _eval_ppl_wikitext2(model, tok)
    print(f"[alpha-sweep] α={alpha}  wt2 PPL = {wt2_ppl:.3f}", flush=True)

    c4_ppl = None
    if not skip_c4:
        print("[alpha-sweep] evaluating C4 PPL ...", flush=True)
        try:
            c4_ppl = _eval_ppl_c4(model, tok)
            print(f"[alpha-sweep] α={alpha}  c4 PPL = {c4_ppl:.3f}", flush=True)
        except Exception as exc:
            print(f"[alpha-sweep] WARNING: C4 eval failed ({exc})", flush=True)
            c4_ppl = float("nan")

    result = {
        "target": target,
        "method": "rtn",
        "augments": "dbaf",
        "alpha": alpha,
        "metrics": {"wikitext2_ppl": wt2_ppl, "c4_ppl": c4_ppl},
        "wallclock_seconds": time.time() - t0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    args = _parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    out_root = pathlib.Path(args.out_root) / args.target
    print(f"[alpha-sweep] target={args.target}  alphas={alphas}  out_root={out_root}",
          flush=True)
    results = []
    for alpha in alphas:
        cell_name = f"rtn_dbaf_alpha{alpha:.2f}".replace(".", "p")
        out_path = out_root / cell_name / "eval.json"
        r = _run_one(args.target, alpha, out_path, skip_c4=args.skip_c4)
        results.append(r)

    # Aggregate
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"[alpha-sweep] DONE; summary written to {summary_path}", flush=True)
    print("=" * 60, flush=True)
    print(f"{'α':>6} | {'wt2 PPL':>10} | {'c4 PPL':>10}", flush=True)
    for r in results:
        a = r["alpha"]
        wt2 = r["metrics"]["wikitext2_ppl"]
        c4 = r["metrics"]["c4_ppl"]
        c4_s = f"{c4:.3f}" if c4 is not None and c4 == c4 else "nan"
        print(f"{a:>6.2f} | {wt2:>10.3f} | {c4_s:>10}", flush=True)


if __name__ == "__main__":
    main()
