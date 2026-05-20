"""Phase A α=0.25 redo for RTN, GPTQ, AWQ on LLaMA-3-8B W4A4.

The α-sweep (alpha-sweep/llama3-8b/summary.json) showed RTN+DBAF wt2
PPL drops 15x when α moves from 0.75 to 0.25. This script re-evaluates
the three hosts in the no-rotation/no-training quadrant of the host
matrix at the new α=0.25 setting.

SmoothQuant is skipped: its quantize_model internally calls
_quantize_per_channel_with_dbaf with α=0.75 hardcoded; treating that
as a separate code change.

Usage:
    python scripts/run_phaseA_alpha025.py --out_root /data/outputs/HM-alpha025
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "FlatQuant"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="llama3-8b")
    p.add_argument("--methods", default="rtn,gptq,awq",
                   help="Comma-separated methods to sweep.")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--out_root", default="/data/outputs/HM-alpha025")
    return p.parse_args(argv)


def _run_one(target: str, method: str, alpha: float, out_path: pathlib.Path):
    import gc
    import torch
    from run_training_free_full_table import (
        _load_llm, _eval_ppl_wikitext2, _eval_ppl_c4, _calib_batch_llm,
    )
    from flatquant.baselines.act_quant import apply_w4a4_act_quant

    if out_path.exists():
        existing = json.loads(out_path.read_text())
        print(f"[phaseA025] skip {method} (exists): {existing['metrics']}",
              flush=True)
        return existing

    print(f"[phaseA025] === {method} α={alpha} ===", flush=True)
    t0 = time.time()
    model, tok = _load_llm(target)

    if method == "rtn":
        from flatquant.baselines.rtn import quantize_model
        model = quantize_model(model, bits=4, use_dbaf=True, alpha=alpha)
    elif method == "gptq":
        from flatquant.baselines.gptq import quantize_model
        calib = _calib_batch_llm(tok)
        model = quantize_model(model, bits=4, calibration_data=calib,
                               use_dbaf=True, dbaf_alpha=alpha)
    elif method == "awq":
        from flatquant.baselines.awq import quantize_model
        calib = _calib_batch_llm(tok)
        model = quantize_model(model, bits=4, calibration_data=calib,
                               use_dbaf=True, alpha_dbaf=alpha)
    elif method == "smoothquant":
        from flatquant.baselines.smoothquant import quantize_model
        calib = _calib_batch_llm(tok)
        model = quantize_model(model, bits=4, calibration_data=calib,
                               alpha=0.5, use_dbaf=True, act_bits=4,
                               dbaf_alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    # SmoothQuant has its own _ActDivideWrapper with built-in act-quant +
    # optional DBAF, so skip the generic post-hoc activation wrapper.
    if method != "smoothquant":
        model = apply_w4a4_act_quant(model, bits=4, use_dbaf=True, alpha=alpha)

    print("[phaseA025] eval wt2 ...", flush=True)
    wt2_ppl = _eval_ppl_wikitext2(model, tok)
    print(f"[phaseA025] {method} α={alpha} wt2 = {wt2_ppl:.3f}", flush=True)

    try:
        c4_ppl = _eval_ppl_c4(model, tok)
        print(f"[phaseA025] {method} α={alpha} c4 = {c4_ppl:.3f}", flush=True)
    except Exception as exc:
        print(f"[phaseA025] WARNING: c4 eval failed: {exc}", flush=True)
        c4_ppl = float("nan")

    result = {
        "target": target,
        "method": method,
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
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_root = pathlib.Path(args.out_root) / args.target
    print(f"[phaseA025] target={args.target} methods={methods} α={args.alpha}",
          flush=True)
    results = []
    for method in methods:
        cell = f"{method}_dbaf_alpha{args.alpha:.2f}".replace(".", "p")
        out_path = out_root / cell / "eval.json"
        r = _run_one(args.target, method, args.alpha, out_path)
        results.append(r)

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"[phaseA025] DONE; summary -> {summary_path}", flush=True)
    print("=" * 60, flush=True)
    print(f"{'method':>10} | {'wt2':>10} | {'c4':>10}", flush=True)
    for r in results:
        wt2 = r["metrics"]["wikitext2_ppl"]
        c4 = r["metrics"]["c4_ppl"]
        c4_s = f"{c4:.3f}" if c4 is not None and c4 == c4 else "nan"
        print(f"{r['method']:>10} | {wt2:>10.3f} | {c4_s:>10}", flush=True)


if __name__ == "__main__":
    main()
