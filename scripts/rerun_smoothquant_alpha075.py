"""Re-run the single disputed cell: SmoothQuant + DBAF at alpha=0.75,
LLaMA-3-8B W4A4, WikiText-2 PPL.

The paper (Table 4 main / Table alpha-cross-host appendix) prints 5,263.70
for this cell; results/HM-alpha-sweep-cross-host.json records 1,477.98.
Every other cell in that appendix table matches the JSON exactly, so exactly
one of the two is wrong.

Recipe is copied verbatim from scripts/run_phaseA_alpha025.py::_run_one for
method == "smoothquant", with the same constants as
scripts/run_training_free_full_table.py (CALIB_NSAMPLES=4, CALIB_SEQLEN=2048,
PPL_SAMPLES=64). Only two things differ, both environmental:
  - model is read from the HF cache rather than /data/modelzoo
  - dataset id is "Salesforce/wikitext" (the bare "wikitext" id no longer
    resolves under datasets>=4)

Also runs alpha=0.25 as a control: that cell is 17.32 in both the paper and
the JSON, so reproducing it confirms the harness is faithful before we trust
its verdict on the 0.75 cell.
"""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, "/home/ubuntu/distributional-ptq")
sys.path.insert(0, "/home/ubuntu/distributional-ptq/FlatQuant")

MODEL = "meta-llama/Meta-Llama-3-8B"
CALIB_NSAMPLES, CALIB_SEQLEN, PPL_SAMPLES = 4, 2048, 64
OUT = "/home/ubuntu/distributional-ptq/results/HM-smoothquant-alpha-recheck.json"


def load():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True).eval()
    return model, tok


def calib_batch(tok):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    return ids[:, : CALIB_NSAMPLES * CALIB_SEQLEN].view(CALIB_NSAMPLES, CALIB_SEQLEN).cuda()


def eval_ppl(model, tok):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids.to(model.device)
    n = min(PPL_SAMPLES, ids.shape[1] // CALIB_SEQLEN)
    nlls = []
    for i in range(n):
        c = ids[:, i * CALIB_SEQLEN:(i + 1) * CALIB_SEQLEN]
        with torch.no_grad():
            nlls.append(model(c, labels=c).loss.float().item())
    return float(torch.tensor(nlls).mean().exp().item())


res = {"model": MODEL, "config": "SmoothQuant(alpha=0.5) + DBAF, W4A4",
       "paper_value_at_0p75": 5263.70, "json_value_at_0p75": 1477.98,
       "control_value_at_0p25": 17.32, "cells": {}}

for dbaf_alpha in [0.25, 0.75]:
    t0 = time.time()
    model, tok = load()
    from flatquant.baselines.smoothquant import quantize_model
    model = quantize_model(model, bits=4, calibration_data=calib_batch(tok),
                           alpha=0.5, use_dbaf=True, act_bits=4,
                           dbaf_alpha=dbaf_alpha)
    ppl = eval_ppl(model, tok)
    res["cells"][str(dbaf_alpha)] = {"wikitext2_ppl": round(ppl, 2),
                                     "seconds": round(time.time() - t0, 1)}
    print(f"[recheck] dbaf_alpha={dbaf_alpha}  wt2 PPL = {ppl:.2f}", flush=True)
    del model
    torch.cuda.empty_cache()

json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
print("saved ->", OUT)
