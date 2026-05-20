"""RTN + DBAF + PCSA-tf at alpha=0.25 on LLaMA-3-8B W4A4.

Tests whether training-free PCSA-tf, which failed at alpha=0.75 (all 5
variants gate-failed >= 20.2 PPL), is rescued by the much-tighter
alpha=0.25 fold. After DBAF folds outliers down 75% per token, the
per-prompt scale that PCSA-tf computes should be sensible rather than
dominated by 290x outliers (the failure mode at alpha=0.75 with
per-layer max-abs).

This is the cheap test the paper needs before committing to "training-free
PCSA-tf does not work for LLMs". ~5 min on A100.

Wiring:
  1) Load FP model, collect per-layer PCSA-tf state (descriptors=per-block
     hidden-state means).
  2) RTN quantize weights with DBAF alpha=0.25.
  3) Wrap each Linear.forward with the PCSA-tf hook that applies
     `apply_pcsa_tf_to_activation(..., use_dbaf=True, dbaf_alpha=0.25)`
     i.e., DBAF-fold-around-PCSA per call.
  No `apply_w4a4_act_quant` wrap: PCSA-tf hook handles the activation
  quant (it's what does the int4 rounding in this path).

Output: /data/outputs/PCSA-tf-alpha025/llama3-8b/rtn_dbaf_pcsa_tf_a025/eval.json
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
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--pcsa_k", type=int, default=8)
    p.add_argument("--out_root", default="/data/outputs/PCSA-tf-alpha025")
    return p.parse_args(argv)


def main():
    import torch
    import torch.nn as nn
    args = _parse_args()
    from flatquant.baselines.rtn import quantize_model as rtn_quantize_model
    from run_training_free_full_table import (
        _load_llm, _eval_ppl_wikitext2, _eval_ppl_c4,
        _collect_llm_pcsa_state, _apply_pcsa_tf_to_llm,
    )

    out_dir = pathlib.Path(args.out_root) / args.target / \
              f"rtn_dbaf_pcsa_tf_a{args.alpha:.2f}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval.json"

    print(f"[pcsa-test] target={args.target} alpha={args.alpha} K={args.pcsa_k}",
          flush=True)
    t0 = time.time()
    model, tok = _load_llm(args.target)

    # 1) Collect PCSA-tf state on FP model (before any quant)
    print("[pcsa-test] collecting PCSA-tf state on FP model ...", flush=True)
    pcsa_state = _collect_llm_pcsa_state(model, tok)

    # 2) RTN weights with DBAF alpha
    print(f"[pcsa-test] RTN quantize (W4, DBAF alpha={args.alpha}) ...",
          flush=True)
    model = rtn_quantize_model(model, bits=4, use_dbaf=True, alpha=args.alpha)

    # 3) PCSA-tf hooks with DBAF-fold-around at the SAME alpha
    print(f"[pcsa-test] applying PCSA-tf hooks (DBAF alpha={args.alpha}) ...",
          flush=True)
    model = _apply_pcsa_tf_to_llm(model, pcsa_state, use_dbaf=True,
                                  dbaf_alpha=args.alpha)

    # 4) Eval
    print("[pcsa-test] eval wt2 ...", flush=True)
    wt2_ppl = _eval_ppl_wikitext2(model, tok)
    print(f"[pcsa-test] wt2 = {wt2_ppl:.3f}", flush=True)

    try:
        c4_ppl = _eval_ppl_c4(model, tok)
        print(f"[pcsa-test] c4 = {c4_ppl:.3f}", flush=True)
    except Exception as exc:
        print(f"[pcsa-test] WARNING: c4 eval failed: {exc}", flush=True)
        c4_ppl = float("nan")

    result = {
        "target": args.target,
        "method": "rtn",
        "augments": "dbaf+pcsa_tf",
        "alpha": args.alpha,
        "pcsa_k": args.pcsa_k,
        "metrics": {"wikitext2_ppl": wt2_ppl, "c4_ppl": c4_ppl},
        "wallclock_seconds": time.time() - t0,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"[pcsa-test] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
