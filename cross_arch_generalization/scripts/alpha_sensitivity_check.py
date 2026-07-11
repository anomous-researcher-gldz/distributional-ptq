"""Reproduce the numbers in ALPHA_SENSITIVITY_DERIVATION.md.

Computes, on real LLaMA-3-8B gate-pass activations:
  - the paper-style alpha* = cbrt(T p_out / ((M-T) p_in)), T=3sigma, M=99.9pct|x|
  - the energy-ratio proxy lambda = E[x^2 | |x|>T] / E[x^2 | |x|<T]
  - the predicted alpha = lambda^(1/3) * alpha*

Result (8 WikiText calib seqs): alpha* median ~0.2, lambda median ~28,
predicted alpha ~0.65 -- i.e. the energy-ratio proxy does NOT recover the operating
alpha=0.25, so lambda is treated as an empirical sensitivity weight, not a
first-principles predictor. Only the closed FORM alpha*_sens = lambda^(1/3) alpha*
and the lambda>=1 => lower-bound result are claimed in the response.
"""
import torch, torch.nn as nn, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
torch.set_grad_enabled(False)
M = "NousResearch/Meta-Llama-3-8B"
tok = AutoTokenizer.from_pretrained(M)
mdl = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float16).cuda().eval()
lin = [(n, m) for n, m in mdl.named_modules() if isinstance(m, nn.Linear) and "lm_head" not in n]

buf = {}
def mk(n):
    def h(mod, args):
        x = args[0]
        if torch.is_tensor(x):
            v = x.detach().float().reshape(-1, x.shape[-1])
            v = v[torch.randperm(v.shape[0])[:2048]].cpu()
            buf.setdefault(n, []).append(v)
    return h
hs = [m.register_forward_pre_hook(mk(n)) for n, m in lin]
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
for t in [t for t in ds["text"] if len(t) > 400][:8]:
    ids = tok(t, return_tensors="pt", truncation=True, max_length=1024).input_ids.cuda()
    mdl(ids)
for h in hs: h.remove()

def gate_ok(x):
    f = x.flatten(); mu, s = f.mean(), f.std().clamp_min(1e-8); z = (f - mu) / s
    return (abs((z**3).mean().item()) <= 0.7 and 3 <= (z**4).mean().item() <= 30
            and 1e-4 <= (z.abs() > 3).float().mean().item() <= 2e-2)

astar, lam, apred = [], [], []
for n, vs in buf.items():
    x = torch.cat(vs, 0)
    if not gate_ok(x): continue
    xa = x.abs(); sig = x.std(); T = 3 * sig
    Mv = torch.quantile(xa.flatten()[:200000], 0.999)
    out = xa > T; p_out = out.float().mean(); p_in = 1 - p_out
    if p_out <= 0 or Mv <= T: continue
    a = float((T * p_out / ((Mv - T) * p_in)).clamp_min(0) ** (1 / 3))
    L = float((x[out] ** 2).mean() / (x[~out] ** 2).mean().clamp_min(1e-8))
    astar.append(a); lam.append(L); apred.append(L ** (1 / 3) * a)
astar, lam, apred = map(np.array, (astar, lam, apred))
print(f"gate-pass activation tensors: {len(astar)}")
print(f"alpha* (calibration formula):        median {np.median(astar):.3f}")
print(f"lambda = E[x^2|out]/E[x^2|in]:        median {np.median(lam):.1f}")
print(f"predicted alpha = lambda^(1/3)*alpha*: median {np.median(apred):.3f}  (operating alpha = 0.25)")
print("=> energy-ratio lambda does NOT predict 0.25; only the form + lower-bound are claimed.")
