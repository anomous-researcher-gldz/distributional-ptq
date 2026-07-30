"""Audit the training-free block of Table 4 (LLaMA-3-8B W4A4) against
results/HM-alpha-sweep-cross-host.json.

Motivation: the SmoothQuant wt2 alpha=0.75 cell printed 5,263.70 while the JSON
recorded 1,477.98; a re-run measured 1,513.97, confirming the JSON. Four further
paper-vs-JSON disagreements remain unchecked, one of which (SmoothQuant C4 at
alpha=0.75, 3,251.95 vs 1,005.35) has the same 3.2x signature.

Recipe copied from scripts/run_phaseA_alpha025.py::_run_one. Environmental
deltas only: model from HF cache, "Salesforce/wikitext" dataset id.

Reproduction noise against the original runs is ~2% (measured on the one cell
already adjudicated), so this settles the large disagreements but cannot
adjudicate sub-2% ones -- those default to the archived JSON.
"""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, "/home/ubuntu/distributional-ptq")
sys.path.insert(0, "/home/ubuntu/distributional-ptq/FlatQuant")

MODEL = "meta-llama/Meta-Llama-3-8B"
NS, SEQ, PPL_N = 4, 2048, 64
OUT = "/home/ubuntu/distributional-ptq/results/HM-table4-audit.json"

PAPER = {  # (host, corpus, alpha) -> value printed in Table 4
    ("rtn", "wt2", 0.75): 250.60, ("rtn", "wt2", 0.25): 16.31,
    ("rtn", "c4", 0.75): 310.08,  ("rtn", "c4", 0.25): 24.49,
    ("gptq", "wt2", 0.75): 348.66, ("gptq", "wt2", 0.25): 32.65,
    ("gptq", "c4", 0.75): 363.25,  ("gptq", "c4", 0.25): 53.43,
    ("awq", "wt2", 0.75): 579.96,  ("awq", "wt2", 0.25): 17.02,
    ("awq", "c4", 0.75): 520.03,   ("awq", "c4", 0.25): 25.84,
    ("smoothquant", "wt2", 0.75): 5263.70, ("smoothquant", "wt2", 0.25): 17.32,
    ("smoothquant", "c4", 0.75): 3251.95,  ("smoothquant", "c4", 0.25): 27.10,
}
JSONV = {
    ("rtn", "wt2", 0.75): 250.6, ("rtn", "wt2", 0.25): 16.31,
    ("rtn", "c4", 0.75): 310.08, ("rtn", "c4", 0.25): 24.49,
    ("gptq", "wt2", 0.75): 348.66, ("gptq", "wt2", 0.25): 32.65,
    ("gptq", "c4", 0.75): 363.25,  ("gptq", "c4", 0.25): 49.06,
    ("awq", "wt2", 0.75): 579.96,  ("awq", "wt2", 0.25): 17.02,
    ("awq", "c4", 0.75): 520.03,   ("awq", "c4", 0.25): 25.97,
    ("smoothquant", "wt2", 0.75): 1477.98, ("smoothquant", "wt2", 0.25): 17.32,
    ("smoothquant", "c4", 0.75): 1005.35,  ("smoothquant", "c4", 0.25): 26.05,
}

_wt2_txt = _c4_txt = None


def corpora():
    global _wt2_txt, _c4_txt
    if _wt2_txt is None:
        _wt2_txt = "\n\n".join(
            load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        c4 = load_dataset("allenai/c4",
                          data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                          split="validation")
        _c4_txt = " ".join(c4[:1100]["text"])
    return _wt2_txt, _c4_txt


def ppl(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    n = min(PPL_N, ids.shape[1] // SEQ)
    nlls = []
    for i in range(n):
        c = ids[:, i * SEQ:(i + 1) * SEQ]
        with torch.no_grad():
            nlls.append(model(c, labels=c).loss.float().item())
    return float(torch.tensor(nlls).mean().exp().item())


def build(host, alpha, tok):
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True).eval()
    calib = None
    if host != "rtn":
        ids = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                           split="train")["text"]), return_tensors="pt").input_ids
        calib = ids[:, : NS * SEQ].view(NS, SEQ).cuda()
    if host == "rtn":
        from flatquant.baselines.rtn import quantize_model
        m = quantize_model(m, bits=4, use_dbaf=True, alpha=alpha)
    elif host == "gptq":
        from flatquant.baselines.gptq import quantize_model
        m = quantize_model(m, bits=4, calibration_data=calib, use_dbaf=True, dbaf_alpha=alpha)
    elif host == "awq":
        from flatquant.baselines.awq import quantize_model
        m = quantize_model(m, bits=4, calibration_data=calib, use_dbaf=True, alpha_dbaf=alpha)
    elif host == "smoothquant":
        from flatquant.baselines.smoothquant import quantize_model
        m = quantize_model(m, bits=4, calibration_data=calib, alpha=0.5,
                           use_dbaf=True, act_bits=4, dbaf_alpha=alpha)
    if host != "smoothquant":
        from flatquant.baselines.act_quant import apply_w4a4_act_quant
        m = apply_w4a4_act_quant(m, bits=4, use_dbaf=True, alpha=alpha)
    return m


tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
wt2_txt, c4_txt = corpora()
rows = []
for host in ["rtn", "gptq", "awq", "smoothquant"]:
    for alpha in [0.25, 0.75]:
        t0 = time.time()
        try:
            m = build(host, alpha, tok)
            got = {"wt2": ppl(m, tok, wt2_txt), "c4": ppl(m, tok, c4_txt)}
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[audit] {host} a={alpha} FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        for corpus in ["wt2", "c4"]:
            p, j, g = PAPER[(host, corpus, alpha)], JSONV[(host, corpus, alpha)], got[corpus]
            rows.append(dict(host=host, corpus=corpus, alpha=alpha, measured=round(g, 2),
                             paper=p, json=j,
                             pct_vs_paper=round(100 * (g - p) / p, 1),
                             pct_vs_json=round(100 * (g - j) / j, 1)))
            print(f"[audit] {host:12s} a={alpha} {corpus:3s} measured={g:9.2f}  "
                  f"paper={p:9.2f} ({rows[-1]['pct_vs_paper']:+.1f}%)  "
                  f"json={j:9.2f} ({rows[-1]['pct_vs_json']:+.1f}%)", flush=True)
        print(f"           ({time.time()-t0:.0f}s)", flush=True)

json.dump({"model": MODEL, "config": "W4A4, DBAF on weights+activations",
           "note": "reproduction noise vs original runs ~2%; sub-2% gaps are not adjudicable",
           "rows": rows}, open(OUT, "w"), indent=1)
print("saved ->", OUT)
