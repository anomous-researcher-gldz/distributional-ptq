"""5dKj follow-up: (A) random-seed / calibration-resample variance, and
(B) instruction + multilingual distribution shift.

Reuses the paper's EXACT activation-gate stats (skew/kurt/frac3 bounds) and
compactness ratio (K=4 K-means / permute-dims baseline) from q1_q3_llama.py.

(A) Seed variance -- redraw the LLM calibration set under N seeds (different
    WikiText-2 subset + different activation subsample). Report per-seed
    activation gate-pass %, cross-seed fire/skip decision agreement (mean pairwise),
    and compactness at the q_proj PCSA site as mean+/-std. This is the honest
    "random seeds" number the diagnostic actually produces (a decision), rather
    than a single-seed point estimate.

(B) Distribution shift -- recompute the gate decision under instruction prompts
    (Alpaca, Dolly) and a multilingual mix (opus-100 de/fr/ru/zh), and report
    per-layer decision agreement vs the WikiText-2 baseline (same metric as the
    C4 swap in Q1). Also compactness at the PCSA site per regime.
"""
import json, gc
import os as _os_repo
_REPO_ROOT = _os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
torch.set_grad_enabled(False)
MODEL = "NousResearch/Meta-Llama-3-8B"
CAP = 120_000
SITE = "model.layers.0.self_attn.q_proj"
OUT = _REPO_ROOT + "/cross_arch_generalization/results/seed_and_shift_results.json"

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
def wikitext_pool(n=200, minlen=200):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return [t for t in ds["text"] if len(t) > minlen][:n]
def alpaca(n=16):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out = [(r["instruction"] + " " + (r.get("input") or "")).strip() for r in ds]
    return [t for t in out if len(t) > 120][:n]
def dolly(n=16):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    out = [(r["instruction"] + " " + (r.get("context") or "")).strip() for r in ds]
    return [t for t in out if len(t) > 120][:n]
def multiling(n=16):
    langs = ["de", "fr", "ru", "zh"]; out = []
    per = max(1, n // len(langs))
    for lg in langs:
        cfg = f"{lg}-en"  # opus-100 configs are alphabetical; try both directions
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", cfg, split="test")
        except Exception:
            cfg = f"en-{lg}"; ds = load_dataset("Helsinki-NLP/opus-100", cfg, split="test")
        got = 0
        for r in ds:
            s = r["translation"][lg]
            if len(s) > 200:
                out.append(s); got += 1
            if got >= per: break
    return out[:n]

POOL = wikitext_pool()

def run_calib(texts, subsample_seed=0, seqlen=1024):
    torch.manual_seed(subsample_seed)
    store = {}
    def mk(nm):
        def h(mod, args):
            x = args[0]
            if not torch.is_tensor(x): return
            v = x.detach().float().reshape(-1)
            if v.numel() > 20000:
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
    gc.collect(); torch.cuda.empty_cache()
    return dec

def descriptors(texts, site=SITE):
    target = dict(model.named_modules())[site]; ds = []
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

# ===================== (A) SEED / CALIBRATION-RESAMPLE VARIANCE =====================
print("\n== (A) seed variance: 5 calibration re-draws ==", flush=True)
NSEED = 5; NCAL = 16
seed_decs, gate_pass, comps = [], [], []
for s in range(NSEED):
    rng = np.random.default_rng(1000 + s)
    idx = rng.choice(len(POOL), NCAL, replace=False)
    texts = [POOL[i] for i in idx]
    dec = run_calib(texts, subsample_seed=s)
    seed_decs.append(dec)
    gp = 100 * np.mean(list(dec.values()))
    gate_pass.append(gp)
    c, _ = compactness(descriptors(texts))
    comps.append(c)
    print(f"  seed {s}: gate-pass={gp:.1f}%  PCSA c={c:.3f}", flush=True)

common = set(seed_decs[0])
for d in seed_decs[1:]: common &= set(d)
common = sorted(common)
# mean pairwise decision agreement across seeds
pair_agree = []
for i in range(NSEED):
    for j in range(i + 1, NSEED):
        pair_agree.append(np.mean([seed_decs[i][n] == seed_decs[j][n] for n in common]))
# per-layer stability: fraction of layers with identical decision across ALL seeds
allsame = np.mean([len({seed_decs[k][n] for k in range(NSEED)}) == 1 for n in common])
A = dict(
    n_seeds=NSEED, n_calib_per_seed=NCAL, n_common_layers=len(common),
    gate_pass_mean=round(float(np.mean(gate_pass)), 1),
    gate_pass_std=round(float(np.std(gate_pass)), 2),
    gate_pass_min=round(float(np.min(gate_pass)), 1),
    gate_pass_max=round(float(np.max(gate_pass)), 1),
    mean_pairwise_decision_agreement_pct=round(100 * float(np.mean(pair_agree)), 1),
    layers_identical_across_all_seeds_pct=round(100 * float(allsame), 1),
    pcsa_c_mean=round(float(np.mean(comps)), 3),
    pcsa_c_std=round(float(np.std(comps)), 3),
    pcsa_decision="SKIP" if np.mean(comps) > 0.4 else "FIRE",
)
print(f"  -> gate-pass {A['gate_pass_mean']}%+/-{A['gate_pass_std']} "
      f"(min {A['gate_pass_min']}, max {A['gate_pass_max']}); "
      f"pairwise decision agreement {A['mean_pairwise_decision_agreement_pct']}%; "
      f"identical-across-all-seeds {A['layers_identical_across_all_seeds_pct']}%; "
      f"PCSA c={A['pcsa_c_mean']}+/-{A['pcsa_c_std']} ({A['pcsa_decision']})")

# ===================== (B) INSTRUCTION + MULTILINGUAL SHIFT =====================
print("\n== (B) distribution shift vs WikiText-2 baseline ==", flush=True)
base_texts = [POOL[i] for i in range(NCAL)]
base = run_calib(base_texts, subsample_seed=0)
B = {}
for label, texts in [("instruction_alpaca", alpaca(NCAL)),
                     ("instruction_dolly", dolly(NCAL)),
                     ("multilingual_de_fr_ru_zh", multiling(NCAL))]:
    dec = run_calib(texts, subsample_seed=0)
    common_b = [n for n in base if n in dec]
    agree = 100 * np.mean([base[n] == dec[n] for n in common_b])
    gp = 100 * np.mean([dec[n] for n in common_b])
    c, cs = compactness(descriptors(texts))
    B[label] = dict(n_layers=len(common_b), agreement_vs_wikitext_pct=round(float(agree), 1),
                    gate_pass_pct=round(float(gp), 1),
                    pcsa_c=round(c, 3), pcsa_c_std=round(cs, 3),
                    pcsa_decision="SKIP" if c > 0.4 else "FIRE")
    print(f"  {label:28s} agree={agree:.1f}%  gate-pass={gp:.1f}%  "
          f"PCSA c={c:.3f} ({B[label]['pcsa_decision']})", flush=True)

res = dict(A_seed_variance=A, B_distribution_shift=B,
           baseline_wikitext_gate_pass_pct=round(100 * np.mean(list(base.values())), 1))
json.dump(res, open(OUT, "w"), indent=2)
print("\nsaved ->", OUT)
print(json.dumps(res, indent=2))
