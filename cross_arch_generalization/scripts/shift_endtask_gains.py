"""End-task (PPL) DBAF gains UNDER DISTRIBUTION SHIFT on LLaMA-3-8B.

Reviewer 5dKj asked whether the *gains* generalize under distribution shift, not
just the gate decision. We quantize LLaMA-3-8B with the paper's weight-only
per-channel RTN host at W3 (the aggressive regime where DBAF matters; W4 weight-only
barely degrades) and measure held-out perplexity with vs without DBAF on FIVE eval
distributions: WikiText-2, C4, code, multilingual (de/fr/ru/zh), instruction
(Alpaca/Dolly) -- the same shifts used for the gate-agreement study.

alpha is frozen at 0.25 (selected on WikiText calibration by the one-block
reconstruction sweep; alpha=0.25 beats 0.5 there), NOT retuned per eval set.
Claim under test: RTN collapses on every distribution and DBAF recovers it on every
distribution -- the end-task gain, not merely the dispatch decision, generalizes.
"""
import sys, glob, torch, torch.nn as nn, json
import os as _os_repo
_REPO_ROOT = _os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT); sys.path.insert(0, _REPO_ROOT + "/FlatQuant")
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf
from flatquant.eval_utils import ppl_eval
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
torch.set_grad_enabled(False)
M = "NousResearch/Meta-Llama-3-8B"; BITS = 3; ALPHA = 0.25; WINDOWS = 40; SEQ = 2048
OUT = _REPO_ROOT + "/cross_arch_generalization/results/shift_endtask_gains_results.json"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float16).cuda().eval()
orig = {k: v.clone() for k, v in model.state_dict().items()}

class Enc:  # ppl_eval needs .input_ids
    def __init__(self, ids): self.input_ids = ids
def enc_of(text):
    ids = tok(text, return_tensors="pt").input_ids
    return Enc(ids[:, :WINDOWS * SEQ])

def corpora():
    out = {}
    wt = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    out["wikitext2"] = enc_of("\n\n".join(wt["text"]))
    c4 = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)
    buf = []
    for r in c4:
        buf.append(r["text"])
        if sum(len(x) for x in buf) > 600_000: break
    out["c4"] = enc_of("\n\n".join(buf))
    code = []
    for f in sorted(glob.glob(_REPO_ROOT + "/**/*.py", recursive=True)):
        try: code.append(open(f).read())
        except Exception: pass
        if sum(len(x) for x in code) > 600_000: break
    out["code"] = enc_of("\n\n".join(code))
    ml = []
    for lg in ["de", "fr", "ru", "zh"]:
        cfg = f"{lg}-en"
        try: ds = datasets.load_dataset("Helsinki-NLP/opus-100", cfg, split="test")
        except Exception:
            cfg = f"en-{lg}"; ds = datasets.load_dataset("Helsinki-NLP/opus-100", cfg, split="test")
        got = 0
        for r in ds:
            s = r["translation"][lg]
            if len(s) > 80: ml.append(s); got += 1
            if got >= 400: break
    out["multilingual"] = enc_of("\n\n".join(ml))
    instr = []
    al = datasets.load_dataset("tatsu-lab/alpaca", split="train")
    for r in al:
        instr.append((r["instruction"] + " " + (r.get("input") or "")).strip())
        if len(instr) >= 800: break
    do = datasets.load_dataset("databricks/databricks-dolly-15k", split="train")
    for r in do:
        instr.append((r["instruction"] + " " + (r.get("context") or "")).strip())
        if len(instr) >= 1600: break
    out["instruction"] = enc_of("\n\n".join(instr))
    return out

def quantize(dbaf):
    model.load_state_dict(orig)
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.dim() == 2 and "lm_head" not in n:
            w = m.weight.data
            wq = (_quantize_per_channel_with_dbaf(w, BITS, alpha=ALPHA) if dbaf
                  else _quantize_tensor_uniform(w, BITS, per_channel=True))
            m.weight.data = wq.to(m.weight.dtype)

print("building corpora...", flush=True)
C = corpora()
for k, v in C.items():
    print(f"  {k}: {v.input_ids.shape[1]} tokens ({v.input_ids.shape[1]//SEQ} windows)", flush=True)

res = {"bits": BITS, "alpha": ALPHA, "windows": WINDOWS, "ppl": {}}
for tag, dbaf in [("FP", None), ("RTN", False), ("RTN+DBAF", True)]:
    if tag == "FP": model.load_state_dict(orig)
    else: quantize(dbaf)
    res["ppl"][tag] = {}
    for name, enc in C.items():
        p = ppl_eval(model, enc)
        res["ppl"][tag][name] = round(float(p), 3)
        print(f"[{tag:9s}] {name:13s} PPL = {p:.3f}", flush=True)
model.load_state_dict(orig)

# summarize per-corpus DBAF gain
res["gain"] = {}
for name in C:
    rtn = res["ppl"]["RTN"][name]; dbaf = res["ppl"]["RTN+DBAF"][name]
    res["gain"][name] = {"rtn": rtn, "dbaf": dbaf, "reduction_x": round(rtn / dbaf, 1),
                         "improved": dbaf < rtn}
print("\n=== per-corpus DBAF gain (W3 weight-only, alpha=0.25 frozen) ===")
for name, g in res["gain"].items():
    print(f"  {name:13s} RTN {g['rtn']:>12.1f} -> +DBAF {g['dbaf']:>10.1f}  ({g['reduction_x']}x)  improved={g['improved']}")
print(f"\nimproved on all {len(res['gain'])} shifted distributions: "
      f"{all(g['improved'] for g in res['gain'].values())}")
json.dump(res, open(OUT, "w"), indent=2); print("saved ->", OUT)
