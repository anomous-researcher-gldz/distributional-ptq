"""SQ6q-W4 -- PCSA descriptor ablation.

Reviewer SQ6q: "the l2-normalized mean-pooled hidden state is a simple choice,
but it is not compared against alternatives (Sentence-BERT, tokenwise attention
pooling, CLS). If a different descriptor yields tighter clusters, the gate's
decisions could change."

We test that directly at the LLaMA-3-8B self-attn q_proj PCSA site (the site the
paper reports at c~0.79, SKIP). For each of five descriptors we recompute the
paper's exact compactness ratio c and its fire/skip decision (fire iff c<=0.4):

  1. mean-pool   (paper default): l2-normalize(mean_t h_t)
  2. last-token  (decoder "CLS"): l2-normalize(h_last)
  3. max-pool                   : l2-normalize(max_t h_t)
  4. attn-pool   (norm-weighted): l2-normalize(sum_t softmax(||h_t||) h_t)
  5. Sentence-BERT (all-MiniLM-L6-v2 embedding of the raw prompt text)

Two prompt regimes:
  A. homogeneous  -- the paper's calibration regime (WikiText-2 chunks)
  B. multi-domain -- 3 genuinely distinct corpora (English prose / web / code),
                     which SHOULD cluster; we report cluster purity vs the known
                     domain label to check the descriptor measures real structure.

Claims under test:
  (i)  the SKIP decision at this site is invariant to descriptor choice, and
  (ii) when real cluster structure exists, every descriptor's c drops and
       recovers the domain labels (purity high) -- i.e. the descriptor is
       measuring structure, not an artefact of mean-pooling.
"""
import sys, os, glob, gc, json
import os as _os_repo
_REPO_ROOT = _os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_grad_enabled(False)

MODEL = "NousResearch/Meta-Llama-3-8B"
SITES = ["model.layers.0.self_attn.q_proj", "model.layers.15.self_attn.q_proj"]
K = 4
N_PER = 60          # prompts per regime (multi-domain: N_PER//3 per domain)
SEQ = 512
OUT = os.path.join(os.path.dirname(__file__), "..", "results",
                   "descriptor_ablation_results.json")

# ---------- compactness (verbatim logic from q1_q3_llama.py / repo) ----------
def _kmeans(X, k=K, iters=150, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    a = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None] - C[None]) ** 2).sum(-1); a = d.argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = X[a == j].mean(0)
    return np.linalg.norm(X - C[a], axis=1), a
def _permute(X, seed=0):
    rng = np.random.default_rng(seed + 7); Xp = X.copy()
    for j in range(X.shape[1]): Xp[:, j] = rng.permutation(Xp[:, j])
    return Xp
def compactness(X, k=K, nb=15):
    X = X.astype(np.float64)
    dr, assign = _kmeans(X, k)
    drm = dr.mean()
    ratios = [drm / max(_kmeans(_permute(X, s), k, seed=s)[0].mean(), 1e-8) for s in range(nb)]
    return float(np.mean(ratios)), float(np.std(ratios)), assign
def purity(assign, labels):
    labels = np.asarray(labels); tot = 0
    for c in np.unique(assign):
        m = assign == c
        if m.any(): tot += np.bincount(labels[m]).max()
    return float(tot / len(labels))

def l2(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

# ---------- corpora ----------
def wikitext(n):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return [t.strip() for t in ds["text"] if len(t) > 300][:n]
def c4(n):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    out = []
    for r in ds:
        if len(r["text"]) > 300: out.append(r["text"].strip())
        if len(out) >= n: break
    return out
def code(n):
    # genuinely distinct domain: python source from the local repo
    files = glob.glob(_REPO_ROOT + "/**/*.py", recursive=True)
    out = []
    for f in sorted(files):
        try:
            s = open(f).read()
        except Exception:
            continue
        if len(s) > 300: out.append(s[:2000])
        if len(out) >= n: break
    return out

print("loading LLaMA-3-8B...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()

# ---------- capture per-token hidden states feeding each q_proj site ----------
def site_token_states(texts, site):
    target = dict(model.named_modules())[site]
    store = []
    def h(mod, args):
        x = args[0]
        if torch.is_tensor(x):
            store.append(x.detach()[0].float().cpu())  # (T, D)
    hk = target.register_forward_pre_hook(h)
    per_prompt = []
    for t in texts:
        store.clear()
        ids = tok(t, return_tensors="pt", truncation=True, max_length=SEQ).input_ids.cuda()
        model(ids)
        per_prompt.append(store[-1])  # (T,D)
    hk.remove()
    return per_prompt  # list of (T,D)

def descriptors_from_states(states):
    """Return dict name-> (N,D) descriptor matrix from per-prompt token states."""
    mean_d, last_d, max_d, attn_d = [], [], [], []
    for h in states:                       # h: (T,D) float
        hn = h.numpy()
        mean_d.append(hn.mean(0))
        last_d.append(hn[-1])
        max_d.append(hn.max(0))
        w = np.linalg.norm(hn, axis=1); w = np.exp(w - w.max()); w /= w.sum()
        attn_d.append((w[:, None] * hn).sum(0))
    return {
        "mean_pool_paper": l2(np.stack(mean_d)),
        "last_token_CLS":  l2(np.stack(last_d)),
        "max_pool":        l2(np.stack(max_d)),
        "attn_pool":       l2(np.stack(attn_d)),
    }

# ---------- Sentence-BERT descriptor (all-MiniLM-L6-v2 via plain transformers) ----------
def sbert_descriptors(texts):
    from transformers import AutoModel, AutoTokenizer as AT
    name = "sentence-transformers/all-MiniLM-L6-v2"
    st = AT.from_pretrained(name); sm = AutoModel.from_pretrained(name).cuda().eval()
    embs = []
    for t in texts:
        enc = st(t, return_tensors="pt", truncation=True, max_length=256).to("cuda")
        out = sm(**enc).last_hidden_state[0]           # (T,H)
        mask = enc["attention_mask"][0].unsqueeze(-1).float()
        emb = (out * mask).sum(0) / mask.sum().clamp_min(1)
        embs.append(emb.float().cpu().numpy())
    del sm; gc.collect(); torch.cuda.empty_cache()
    return l2(np.stack(embs))

def eval_regime(texts, labels, tag):
    rows = {}
    sb = sbert_descriptors(texts)                       # descriptor independent of site
    for site in SITES:
        states = site_token_states(texts, site)
        descs = descriptors_from_states(states)
        descs["sentence_bert_MiniLM"] = sb
        for name, X in descs.items():
            c, std, assign = compactness(X)
            rows.setdefault(name, {})[site] = {
                "c": round(c, 3), "std": round(std, 3),
                "decision": "FIRE" if c <= 0.4 else "SKIP",
                "purity": round(purity(assign, labels), 3) if labels is not None else None,
            }
        gc.collect(); torch.cuda.empty_cache()
    return rows

results = {"K": K, "n_per_regime": N_PER, "sites": SITES, "fire_threshold": 0.4}

print("\n== Regime A: homogeneous (WikiText-2, paper calibration regime) ==", flush=True)
wt = wikitext(N_PER)
results["A_homogeneous_wikitext"] = eval_regime(wt, None, "A")

print("\n== Regime B: multi-domain (English prose / web / code) ==", flush=True)
m = N_PER // 3
dom = wikitext(m) + c4(m) + code(m)
lab = [0] * m + [1] * m + [2] * len(code(m)[:m])
lab = lab[:len(dom)]
results["B_multidomain"] = eval_regime(dom, lab, "B")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)
print("\nsaved ->", OUT)
print(json.dumps(results, indent=2))
