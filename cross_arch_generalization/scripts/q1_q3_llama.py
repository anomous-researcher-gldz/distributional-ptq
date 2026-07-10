"""Q1 + Q3 on LLaMA-3-8B (ungated NousResearch mirror), one model load.

Q1 -- calibration-set robustness. Recompute the per-Linear ACTIVATION gate under
      WikiText-2 vs C4 calibration and report how often the fire/skip decision
      agrees. (Weight gate is input-independent; we Q1 is about the
      activation/calibration side.)

Q3 -- compactness vs prompt diversity. Using the paper's exact compactness ratio
      (repo scripts/cluster_tractability.py: mean within-cluster distance under
      K=4 K-means, normalized by a permute-dims baseline), measure c at the LLM
      q_proj PCSA site as prompt diversity increases from a single WikiText
      article to a WikiText+C4 mix. Prediction: c stays above the 0.4 fire
      threshold (PCSA self-disables at this site) and rises with diversity, so
      the dispatch decision is preserved and grows more conservative.
"""
import sys, json, gc, itertools
import numpy as np, torch, torch.nn as nn
torch.set_grad_enabled(False)
MODEL = "NousResearch/Meta-Llama-3-8B"
CAP = 120_000          # per-layer activation subsample cap for gate stats

def profile(x, eps=1e-8):
    flat = x.detach().float().reshape(-1)
    if flat.numel() == 0: return None
    m, s = flat.mean(), flat.std().clamp_min(eps)
    z = (flat - m) / s
    return dict(skew=(z**3).mean().item(), kurt=(z**4).mean().item(),
                frac3=(z.abs() > 3).float().mean().item())
def gate_ok(s):
    return bool(abs(s["skew"]) <= 0.7 and 3.0 <= s["kurt"] <= 30.0
               and 1e-4 <= s["frac3"] <= 2e-2)

# ---- compactness (verbatim logic from repo) ----
def _kmeans(X, k=4, iters=150, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    a = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None] - C[None]) ** 2).sum(-1); a = d.argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = X[a == j].mean(0)
    return np.linalg.norm(X - C[a], axis=1)
def _permute(X, seed=0):
    rng = np.random.default_rng(seed + 7); Xp = X.copy()
    for j in range(X.shape[1]): Xp[:, j] = rng.permutation(Xp[:, j])
    return Xp
def compactness(X, k=4, nb=15):
    X = X.astype(np.float64)
    dr = _kmeans(X, k).mean()
    ratios = [dr / max(_kmeans(_permute(X, s), k, seed=s).mean(), 1e-8) for s in range(nb)]
    return float(np.mean(ratios)), float(np.std(ratios))

print("loading model...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
print(f"{len(linears)} Linear layers", flush=True)

# ---------- corpora ----------
def wikitext(n, minlen=200):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return [t for t in ds["text"] if len(t) > minlen][:n]
def c4(n):
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    out = []
    for r in ds:
        if len(r["text"]) > 200: out.append(r["text"])
        if len(out) >= n: break
    return out

# =========================== Q1 ===========================
def run_calib(texts, seqlen=1024):
    store = {}
    def mk(nm):
        def h(mod, args):
            x = args[0]
            if not torch.is_tensor(x): return
            v = x.detach().float().reshape(-1)
            if v.numel() > 20000:  # subsample per call
                v = v[torch.randperm(v.numel(), device=v.device)[:20000]]
            v = v.cpu()
            if nm not in store: store[nm] = v
            elif store[nm].numel() < CAP: store[nm] = torch.cat([store[nm], v])
        return h
    hs = [m.register_forward_pre_hook(mk(n)) for n, m in linears]
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=seqlen).input_ids.cuda()
        model(ids)
    for h in hs: h.remove()
    dec = {}
    for n, _ in linears:
        if n in store and store[n].numel() > 1000:
            dec[n] = gate_ok(profile(store[n]))
    return dec

print("\n[Q1] WikiText-2 calibration...", flush=True)
wt_txt = wikitext(16)
dec_wt = run_calib(wt_txt); gc.collect(); torch.cuda.empty_cache()
print("[Q1] C4 calibration...", flush=True)
c4_txt = c4(16)
dec_c4 = run_calib(c4_txt)
common = [n for n in dec_wt if n in dec_c4]
agree = [dec_wt[n] == dec_c4[n] for n in common]
q1 = dict(n_layers=len(common),
          agreement_pct=round(100*np.mean(agree), 1),
          gate_pass_wikitext_pct=round(100*np.mean([dec_wt[n] for n in common]), 1),
          gate_pass_c4_pct=round(100*np.mean([dec_c4[n] for n in common]), 1))
print(f"[Q1] activation-gate decision agreement WikiText vs C4: "
      f"{q1['agreement_pct']}% over {q1['n_layers']} layers "
      f"(gate-pass {q1['gate_pass_wikitext_pct']}% WT vs {q1['gate_pass_c4_pct']}% C4)")

# =========================== Q3 ===========================
def descriptors(texts, site="model.layers.0.self_attn.q_proj"):
    target = dict(model.named_modules())[site]
    ds = []
    def h(mod, args):
        x = args[0]
        if torch.is_tensor(x):
            d = torch.nn.functional.normalize(x.float().mean(1), dim=-1)
            ds.append(d.cpu().numpy())
    hd = target.register_forward_pre_hook(h)
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        model(ids)
    hd.remove()
    return np.concatenate(ds, 0)

print("\n[Q3] compactness vs prompt diversity at LLM q_proj site...", flush=True)
# diversity axis: (1) single WikiText article split into chunks -> homogeneous
#                 (2) many WikiText articles -> medium
#                 (3) WikiText + C4 mix -> high
big = [t for t in load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")["text"] if len(t)>1200]
one = big[0]
single_art = [one[i:i+300] for i in range(0, min(len(one),300*40), 300)][:40]
many_art = wikitext(40)
mixed = (wikitext(20) + c4(20))
q3 = {}
for label, texts in [("low_diversity_single_article", single_art),
                      ("medium_many_wikitext_articles", many_art),
                      ("high_wikitext+c4_mix", mixed)]:
    X = descriptors(texts)
    c, cs = compactness(X)
    fires = c <= 0.4
    q3[label] = dict(n=int(X.shape[0]), compactness=round(c,3), std=round(cs,3),
                     pcsa_decision="FIRE" if fires else "SKIP")
    print(f"[Q3] {label:34s} c={c:.3f}±{cs:.3f}  -> PCSA {'FIRE' if fires else 'SKIP'}")

out = dict(Q1=q1, Q3=q3)
p = "/tmp/claude-1000/-home-ubuntu/629c24a1-ecff-4b3c-8a83-a9277305528e/scratchpad/q1_q3_results.json"
json.dump(out, open(p,"w"), indent=2)
print(f"\nSaved -> {p}")
