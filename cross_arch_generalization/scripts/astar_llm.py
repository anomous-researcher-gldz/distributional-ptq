"""Closed-form alpha* (Eq. alpha-star) for an LLM, on BOTH paths.

  alpha*_A -- input activations of each Linear, over WikiText-2 calibration
              sequences (the tensors DBAF folds at runtime under W4A4)
  alpha*_W -- the Linear weights themselves (the tensors DBAF folds offline)

Both use the canonical definition, T = 3 * std(x) on the SIGNED tensor, matching
ahcptq/quantization/fake_quant.py::compute_T. This is the artifact backing the
alpha* values quoted in the method section.

NOT to be confused with the empirical global-MSE optimum in
results/w3_real_results.json ("global_opt"), which is a swept quantity on
weights and coincides numerically with alpha*_A on LLaMA-3-8B by accident.

Usage:  python3 astar_llm.py meta-llama/Meta-Llama-3-8B [n_seq]
"""
import json, os, sys
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Meta-Llama-3-8B"
NSEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 32
SEQLEN, SUB = 2048, 1_000_000
# Repo-root results/, not cross_arch_generalization/results/: the paper cites
# this file as results/astar_llm_<model>.json (Sec. 3, alpha-star paragraph).
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "astar_llm_%s.json" % MODEL.split("/")[-1])


def astar_from(x):
    """alpha* for one tensor sample. Returns None if the fold is degenerate."""
    x = x.reshape(-1).float()
    if x.numel() > SUB:
        x = x[torch.randint(0, x.numel(), (SUB,), device=x.device)]
    T = 3.0 * x.std().clamp_min(1e-8)
    a = x.abs()
    M = torch.quantile(a, 0.999)
    if M <= T:
        return None
    p_out = a.gt(T).float().mean()
    if p_out <= 0:
        return None
    r = (T * p_out) / ((M - T) * (1 - p_out) + 1e-12)
    return float(r ** (1 / 3)), float(p_out), float(M / (T / 3.0))


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                             device_map="cuda:0").eval()

# ---- alpha*_W : weights ----
wv = []
for n, m in model.named_modules():
    if isinstance(m, nn.Linear) and "layers." in n and m.weight.dim() == 2:
        r = astar_from(m.weight.data)
        if r:
            wv.append(r[0])
print(f"alpha*_W  n={len(wv)}  mean={sum(wv)/len(wv):.4f}", flush=True)

# ---- alpha*_A : input activations over WikiText-2 ----
acc = {}
def hook(name):
    def pre(mod, inp):
        r = astar_from(inp[0].detach())
        if r:
            acc.setdefault(name, []).append(r)
    return pre

hooks = [m.register_forward_pre_hook(hook(n)) for n, m in model.named_modules()
         if isinstance(m, nn.Linear) and "layers." in n]
ids = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                   split="train")["text"]), return_tensors="pt").input_ids
with torch.no_grad():
    for i in range(NSEQ):
        b = ids[:, i * SEQLEN:(i + 1) * SEQLEN]
        if b.shape[1] < SEQLEN:
            break
        model(b.to("cuda:0"))
for h in hooks:
    h.remove()


def agg(vals):
    vals = sorted(vals)
    n = len(vals)
    return dict(n=n, mean=sum(vals) / n, median=vals[n // 2],
                p10=vals[n // 10], p90=vals[(9 * n) // 10])


av = [sum(v[0] for v in s) / len(s) for s in acc.values()]
po = [sum(v[1] for v in s) / len(s) for s in acc.values()]
out = {
    "model": MODEL, "n_seq": NSEQ, "seqlen": SEQLEN,
    "definition": "T = 3*std(signed x); M = quantile(|x|,0.999); alpha* = cbrt(T*p_out/((M-T)*p_in))",
    "alpha_star_A_activations": agg(av),
    "alpha_star_W_weights": agg(wv),
    "p_out_at_3sigma_activations": agg(po),
}
print(json.dumps(out, indent=2))
json.dump(out, open(OUT, "w"), indent=1)
print("saved ->", OUT)
