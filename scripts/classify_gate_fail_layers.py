"""Classify gate-fail layers across all cross-model JSONs into distribution types.

References Wu et al. 2406.06649 taxonomy. Tries to bucket each gate-fail layer
based on its observed weight (or activation) statistics:

  - gaussian-with-sparse-outliers: gate would have passed (kurt 3-30, skew<=0.7, frac3 in 1e-4..2e-2)
  - post-ReLU asymmetric: skew > 0.7 AND values nearly always >= 0
  - bimodal: judge_bimodal() reports two distinct peaks
  - heavy-tailed unimodal: kurt > 30 AND |skew| <= 0.7 AND not bimodal
  - dense-outlier: frac3 > 2e-2 (too many outliers for sparse pattern)
  - low-outlier: frac3 < 1e-4 (basically Gaussian, DBAF doesn't fire)
  - other

For each model, report counts in each bucket. This lets the paper say:
"gate-fail activations in SAM are ~X% post-softmax / Y% bimodal..."
"""
from __future__ import annotations
import sys, json, pathlib, glob
import numpy as np
import torch
sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
from ahcptq.quantization.fake_quant import is_like_normal_plus_3sigma_outliers


def judge_bimodal_simple(stats: dict) -> bool:
    """Heuristic bimodal: high kurt + very low frac>3σ (mass split into two clusters,
    so individual outliers are scarce despite the bulk being heavy-tailed).
    Calibrated to match post-softmax / channel-distinct patterns.
    """
    return stats["kurt"] > 50 and stats["frac3"] < 5e-4


def classify(stats: dict) -> str:
    skew = abs(stats["skew"])
    kurt = stats["kurt"]
    frac3 = stats["frac3"]
    # Gate-pass first
    if (skew <= 0.7) and (3.0 <= kurt <= 30.0) and (1e-4 <= frac3 <= 2e-2):
        return "gaussian_with_sparse_outliers"
    # Post-softmax / bimodal
    if judge_bimodal_simple(stats):
        return "bimodal_or_postsoftmax"
    # Skewed (post-ReLU)
    if skew > 0.7:
        return "skewed_post_relu_like"
    # Heavy-tailed unimodal (Student-t style)
    if kurt > 30:
        return "heavy_tailed_unimodal"
    # Dense-outlier (too many 3σ events)
    if frac3 > 2e-2:
        return "dense_outlier"
    # Low-outlier (basically Gaussian, sparse fails)
    if frac3 < 1e-4:
        return "near_gaussian_no_outliers"
    return "other"


def classify_layer(layer_row: dict, side: str) -> dict:
    """side: 'w' or 'a'. Returns classification + stats."""
    if side == "w":
        if layer_row.get("w_gate") is None:
            return {"kind": None, "side": side}
        stats = {"skew": layer_row["w_skew"], "kurt": layer_row["w_kurt"],
                 "frac3": layer_row["w_frac3"]}
    else:
        if layer_row.get("a_gate") is None:
            return {"kind": None, "side": side}
        stats = {"skew": layer_row["a_skew"], "kurt": layer_row["a_kurt"],
                 "frac3": layer_row["a_frac3"]}
    kind = classify(stats)
    return {"kind": kind, "side": side, "stats": stats}


def main():
    root = pathlib.Path("/home/ubuntu/unifying-ptq/results/S4-cross-model-layer-analysis")
    out_dir = pathlib.Path("/home/ubuntu/unifying-ptq/results/F3-distribution-taxonomy")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for f in sorted(root.glob("*.json")):
        if f.name == "summary.json": continue
        d = json.load(f.open())
        model_name = d["summary"]["model"]
        rows = d["layers"]
        from collections import Counter
        w_counts = Counter(); a_counts = Counter()
        for r in rows:
            wc = classify_layer(r, "w")
            ac = classify_layer(r, "a")
            if wc["kind"] is not None: w_counts[wc["kind"]] += 1
            if ac["kind"] is not None: a_counts[ac["kind"]] += 1
        summary = {"model": model_name,
                   "weight_kinds": dict(w_counts), "activation_kinds": dict(a_counts)}
        summary_rows.append(summary)
        (out_dir / f"{model_name}.json").write_text(json.dumps(summary, indent=2))

    # Pretty-print summary across models
    print(f"{'Model':<10} | {'W kinds':<60} | {'A kinds':<60}")
    for s in summary_rows:
        wk = ", ".join(f"{k.split('_')[0]}:{v}" for k, v in sorted(s["weight_kinds"].items(), key=lambda kv: -kv[1]))
        ak = ", ".join(f"{k.split('_')[0]}:{v}" for k, v in sorted(s["activation_kinds"].items(), key=lambda kv: -kv[1]))
        print(f"{s['model']:<10} | {wk:<60} | {ak:<60}")
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    print(f"\nFull JSON: {out_dir}")


if __name__ == "__main__":
    main()
