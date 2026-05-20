#!/usr/bin/env python3
"""Re-aggregate the Host-Matrix W4A4 Llama-3-8B table from eval.json files.

Stdlib only. Idempotent: safe to re-run as cells land. Missing metrics are
rendered as ``(missing)``; null/None metrics as ``(running)``.
"""
import json
import os
import sys

BASE = "/data/outputs"
ROWS = [
    ("RTN",         "✗", "✗", f"{BASE}/HM-phase-a-w4a4/llama3-8b/rtn_alone/eval.json",       f"{BASE}/HM-phase-a-w4a4/llama3-8b/rtn_dbaf/eval.json"),
    ("GPTQ",        "✗", "✗", f"{BASE}/HM-phase-a-w4a4/llama3-8b/gptq_alone/eval.json",      f"{BASE}/HM-phase-a-w4a4/llama3-8b/gptq_dbaf/eval.json"),
    ("AWQ",         "✗", "✗", f"{BASE}/HM-phase-a-w4a4/llama3-8b/awq_alone/eval.json",       f"{BASE}/HM-phase-a-w4a4/llama3-8b/awq_dbaf/eval.json"),
    ("SmoothQuant", "✗", "✗", f"{BASE}/G8-training-free-full/llama3-8b/smoothquant_alone/eval.json",
                                         f"{BASE}/G8-training-free-full/llama3-8b/smoothquant_dbaf/eval.json"),
    ("QuaRot",      "✓", "✗", f"{BASE}/HM-quarot/llama3-8b/quarot_alone/eval.json",          f"{BASE}/HM-quarot/llama3-8b/quarot_dbaf/eval.json"),
    ("TesseraQ",    "✗", "✓", f"{BASE}/HM-tesseraq/llama3-8b/tesseraq_alone/eval.json",      f"{BASE}/HM-tesseraq/llama3-8b/tesseraq_dbaf/eval.json"),
]


def load_metric(path, key):
    if not os.path.exists(path):
        return None, "missing"
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None, "missing"
    v = d.get("metrics", {}).get(key)
    if v is None:
        return None, "running"
    return float(v), "ok"


def fmt_ppl(v, status):
    if status == "missing":
        return "(missing)"
    if status == "running":
        return "(running)"
    return f"{v:.2f}"


def fmt_delta(a, b, sa, sb):
    if sa != "ok" or sb != "ok" or a == 0:
        return "—"
    pct = (a - b) / a * 100.0
    sign = "−" if pct >= 0 else "+"
    if a < 20:  # saturated regime: 2 sig figs
        return f"{sign}{abs(pct):.2g}%"
    return f"{sign}{abs(pct):.0f}%"


def main():
    header = ["Method", "Rotation", "Trained",
             "Alone wt2", "+DBAF wt2", "Δ wt2",
             "Alone c4", "+DBAF c4", "Δ c4"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for name, rot, trained, p_alone, p_dbaf in ROWS:
        aw, saw = load_metric(p_alone, "wikitext2_ppl")
        dw, sdw = load_metric(p_dbaf,  "wikitext2_ppl")
        ac, sac = load_metric(p_alone, "c4_ppl")
        dc, sdc = load_metric(p_dbaf,  "c4_ppl")
        row = [name, rot, trained,
               fmt_ppl(aw, saw), fmt_ppl(dw, sdw), fmt_delta(aw, dw, saw, sdw),
               fmt_ppl(ac, sac), fmt_ppl(dc, sdc), fmt_delta(ac, dc, sac, sdc)]
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
