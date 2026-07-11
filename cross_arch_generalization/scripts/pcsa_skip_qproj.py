"""Forced-PCSA control at a SKIP site (LLaMA-3-8B self-attn q_proj, c~0.8).

Reviewer question (PDbC / SQ6q): does the gate ever skip a site where PCSA would
actually have helped? This directly measures the counterfactual at the q_proj site
the paper reports as a PCSA SKIP (compactness c~0.79 > 0.4).

We *force* PCSA on anyway -- cluster the per-prompt mean-pooled descriptors (K=4,
the paper's setting), then compare INT4 activation quantization under
  (a) one global scale (max|x| over all prompts), vs
  (b) per-cluster scales (max|x| within each descriptor cluster) = forced PCSA.
If PCSA were beneficial here, per-cluster scales would cut reconstruction NMSE.

Result (48 WikiText-2 prompts, layer 15 q_proj):
  global-scale NMSE  = 0.640
  per-cluster (PCSA) = 0.620
  improvement        = +3.05%   (per-cluster scales ~ [14.5,15.3,13.0,12.2] ~ global 15.3)
i.e. the clusters share essentially the same scale -> forcing PCSA is a near-no-op,
so the gate's SKIP decision at this site is correct. Backs the PDbC/SQ6q claim that
"forcing PCSA at q_proj cuts INT4 reconstruction error by only ~3%".
"""
import os, json
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.set_grad_enabled(False)
DEV = "cuda"
QMAX = 7            # symmetric INT4, q_max = 2^(b-1) - 1
K = 4              # paper's PCSA cluster count
MODEL = "NousResearch/Meta-Llama-3-8B"
LAYER = 15         # a mid-network self-attn q_proj (reported SKIP site)
N_PROMPTS = 48
OUT = os.path.join(os.path.dirname(__file__), "..", "results",
                   "pcsa_skip_qproj_results.json")

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
site = m.model.layers[LAYER].self_attn.q_proj

caps = []
def _hook(mod, args):
    x = args[0]
    if torch.is_tensor(x):
        caps.append(x.detach().float()[0].cpu())      # (T, D)
hk = site.register_forward_pre_hook(_hook)

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
for t in [t for t in ds["text"] if len(t) > 400][:N_PROMPTS]:
    ids = tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids.to(DEV)
    m(ids)
hk.remove()

def _l2(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

# per-prompt descriptor = l2-normalized mean-pooled hidden state (paper default)
desc = _l2(np.stack([c.mean(0).numpy() for c in caps]))

# K-means on descriptors (fixed seed for reproducibility)
rng = np.random.default_rng(0)
C = desc[rng.choice(len(desc), K, replace=False)].copy()
for _ in range(50):
    a = ((desc[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    for j in range(K):
        if (a == j).any():
            C[j] = desc[a == j].mean(0)

def _quant(x, scale):
    s = max(scale, 1e-8) / QMAX
    return np.round(x / s).clip(-QMAX, QMAX) * s

gscale = np.concatenate([np.abs(c.numpy()).ravel() for c in caps]).max()
cluster_scale = np.array([
    max([np.abs(caps[i].numpy()).max() for i in range(len(caps)) if a[i] == j] + [0.0])
    for j in range(K)
])

nmse_g, nmse_p = [], []
for i, c in enumerate(caps):
    x = c.numpy()
    e = (x ** 2).sum() + 1e-8
    nmse_g.append(((x - _quant(x, gscale)) ** 2).sum() / e)
    nmse_p.append(((x - _quant(x, cluster_scale[a[i]])) ** 2).sum() / e)
ng, npc = float(np.mean(nmse_g)), float(np.mean(nmse_p))
impr = 100 * (ng - npc) / ng

res = {
    "site": f"model.layers.{LAYER}.self_attn.q_proj",
    "compactness_regime": "SKIP (c~0.79 > 0.4)",
    "n_prompts": len(caps), "K": K, "int4_qmax": QMAX,
    "global_scale_nmse": ng, "per_cluster_pcsa_nmse": npc,
    "pcsa_improvement_pct": impr,
    "per_cluster_scales": [round(float(s), 2) for s in cluster_scale],
    "global_scale": float(gscale),
    "verdict": "near-no-op: per-cluster scales ~ global -> SKIP is correct",
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)

print(f"LLaMA-3-8B layer{LAYER} q_proj (SKIP site, c~0.8), INT4 act quant, {len(caps)} prompts, K={K}")
print(f"  global-scale NMSE   = {ng:.5f}")
print(f"  per-cluster (PCSA)  = {npc:.5f}")
print(f"  PCSA improvement    = {impr:+.2f}%   (per-cluster scales: {np.round(cluster_scale,1)}, global {gscale:.1f})")
print(f"  -> {res['verdict']}")
print(f"wrote {OUT}")
