"""End-task (PPL) DBAF gains under distribution shift in the PAPER'S HEADLINE
W4A4 regime (weight AND activation 4-bit), LLaMA-3-8B.

Unlike shift_endtask_gains.py (weight-only W3), this uses the full W4A4 host that
the rotation-cliff table is built on: W4 per-channel weight RTN + A4 per-token
activation quant. DBAF folds outliers on BOTH the weight and activation paths.
alpha frozen at 0.25 (the paper's W4A4 LLM optimum, WikiText-selected).

Sanity target: on WikiText the RTN W4A4 host should collapse to ~10^3 PPL and
+DBAF should recover it toward ~10^1 (paper Table 2: RTN 970 -> 16.3). We then check
the SAME recovery holds on C4 / code / multilingual / instruction.
"""
import sys, glob, gc, json, torch, torch.nn as nn
import os as _os_repo
_REPO_ROOT = _os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT); sys.path.insert(0, _REPO_ROOT + "/FlatQuant")
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf
from flatquant.baselines.act_quant import apply_w4a4_act_quant
from flatquant.eval_utils import ppl_eval
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
torch.set_grad_enabled(False)
M = "NousResearch/Meta-Llama-3-8B"; WBITS = ABITS = 4; ALPHA = 0.25; WINDOWS = 24; SEQ = 2048
OUT = _REPO_ROOT + "/cross_arch_generalization/results/shift_endtask_w4a4_results.json"
tok = AutoTokenizer.from_pretrained(M)

class Enc:
    def __init__(self, ids): self.input_ids = ids
def enc_of(text):
    return Enc(tok(text, return_tensors="pt").input_ids[:, :WINDOWS * SEQ])

def corpora():
    out = {}
    wt = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    out["wikitext2"] = enc_of("\n\n".join(wt["text"]))
    c4 = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)
    buf = []
    for r in c4:
        buf.append(r["text"])
        if sum(len(x) for x in buf) > 500_000: break
    out["c4"] = enc_of("\n\n".join(buf))
    code = []
    for f in sorted(glob.glob(_REPO_ROOT + "/**/*.py", recursive=True)):
        try: code.append(open(f).read())
        except Exception: pass
        if sum(len(x) for x in code) > 500_000: break
    out["code"] = enc_of("\n\n".join(code))
    ml = []
    for lg in ["de", "fr", "ru", "zh"]:
        try: ds = datasets.load_dataset("Helsinki-NLP/opus-100", f"{lg}-en", split="test")
        except Exception: ds = datasets.load_dataset("Helsinki-NLP/opus-100", f"en-{lg}", split="test")
        got = 0
        for r in ds:
            s = r["translation"][lg]
            if len(s) > 80: ml.append(s); got += 1
            if got >= 400: break
    out["multilingual"] = enc_of("\n\n".join(ml))
    instr = []
    for r in datasets.load_dataset("tatsu-lab/alpaca", split="train"):
        instr.append((r["instruction"] + " " + (r.get("input") or "")).strip())
        if len(instr) >= 700: break
    for r in datasets.load_dataset("databricks/databricks-dolly-15k", split="train"):
        instr.append((r["instruction"] + " " + (r.get("context") or "")).strip())
        if len(instr) >= 1400: break
    out["instruction"] = enc_of("\n\n".join(instr))
    return out

print("building corpora...", flush=True)
C = corpora()
for k, v in C.items():
    print(f"  {k}: {v.input_ids.shape[1]//SEQ} windows", flush=True)

def fresh():
    m = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float16).cuda().eval()
    return m
def wquant(m, dbaf):
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.dim() == 2 and "lm_head" not in n:
            w = mod.weight.data
            wq = (_quantize_per_channel_with_dbaf(w, WBITS, alpha=ALPHA) if dbaf
                  else _quantize_tensor_uniform(w, WBITS, per_channel=True))
            mod.weight.data = wq.to(mod.weight.dtype)

res = {"w_bits": WBITS, "a_bits": ABITS, "alpha": ALPHA, "windows": WINDOWS, "ppl": {}}
def run(tag, build):
    m = fresh(); build(m)
    res["ppl"][tag] = {}
    for name, enc in C.items():
        p = ppl_eval(m, enc); res["ppl"][tag][name] = round(float(p), 3)
        print(f"[{tag:11s}] {name:13s} PPL = {p:.3f}", flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()

run("FP", lambda m: None)
run("RTN-W4A4", lambda m: (wquant(m, False), apply_w4a4_act_quant(m, ABITS, use_dbaf=False)))
run("DBAF-W4A4", lambda m: (wquant(m, True), apply_w4a4_act_quant(m, ABITS, use_dbaf=True, alpha=ALPHA)))

res["gain"] = {}
for name in C:
    rtn = res["ppl"]["RTN-W4A4"][name]; dbaf = res["ppl"]["DBAF-W4A4"][name]
    res["gain"][name] = {"rtn": rtn, "dbaf": dbaf, "reduction_x": round(rtn / dbaf, 1), "improved": dbaf < rtn}
print("\n=== per-corpus DBAF gain (W4A4, alpha=0.25 frozen) ===")
for name, g in res["gain"].items():
    print(f"  {name:13s} RTN {g['rtn']:>11.1f} -> +DBAF {g['dbaf']:>9.1f}  ({g['reduction_x']}x)  improved={g['improved']}")
print(f"\nimproved on all {len(res['gain'])} shifts: {all(g['improved'] for g in res['gain'].values())}")
json.dump(res, open(OUT, "w"), indent=2); print("saved ->", OUT)
