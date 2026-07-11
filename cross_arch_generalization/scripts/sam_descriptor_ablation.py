"""Positive-site (FIRE) PCSA descriptor robustness — SQ6q-W4 future-work item.

Counterpart to the LLaMA q_proj SKIP-site descriptor ablation: here we test whether
the paper's PCSA FIRE decision at the SAM-B mask-decoder cross-attention q-projection
(reported c=0.189, FIRE) survives changing the descriptor's POOLING, using the repo's
exact permute-dims compactness metric (K=4) and the exact site (both decoder layers,
cross_attn_token_to_image.q_proj).

Descriptors (activation poolings over the decoder tokens, then L2-normalize):
  1. mean-pool  (paper default)
  2. last-token (CLS-style)
  3. max-pool
  4. attn-pool  (norm-weighted softmax over tokens)

Two protocols, matching sam_real_vs_synth.py:
  A. v2 synthetic (random img + random points)  -- the paper's reported regime
  B. real COCO images + points at GT object centers

FIRE iff c <= 0.4. Robust iff the decision is invariant to the pooling choice.
"""
import os, json
import numpy as np, torch, torch.nn.functional as F
torch.set_grad_enabled(False); DEV = "cuda"
import segment_anything as sa
from segment_anything import SamPredictor
from pycocotools.coco import COCO
from PIL import Image

WS = os.environ.get("SAM_WS", "/home/ubuntu/rebuttal_workspace")  # dir with sam_vit_b.pth + coco/
sam = sa.sam_model_registry["vit_b"](checkpoint=f"{WS}/sam_vit_b.pth").cuda().eval()
md = sam.mask_decoder
K = 4

# ---------- permute-dims compactness (paper's exact metric, from cluster_tractability_v2) ----------
def _km(A_, k, seed):
    rng = np.random.default_rng(seed); C = A_[rng.choice(len(A_), k, replace=False)].copy()
    for _ in range(80):
        a = ((A_[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = A_[a == j].mean(0)
    return np.linalg.norm(A_ - C[a], axis=1).mean()

def perm_ratio(X, k=4, nb=20):
    X = X.astype(np.float64); dr = _km(X, k, 0); rs = []
    for s in range(nb):
        Xp = X.copy(); rng = np.random.default_rng(s + 7)
        for j in range(X.shape[1]): Xp[:, j] = rng.permutation(Xp[:, j])
        rs.append(dr / max(_km(Xp, k, s), 1e-8))
    return float(np.mean(rs)), float(np.std(rs))

# ---------- capture RAW q per decoder-layer call (so we can re-pool 4 ways) ----------
raw = []  # list of [tokens, D] arrays
def hook(mod, args):
    q = args[0].float()             # [B=1, tokens, D]
    raw.append(q[0].cpu().numpy())
handles = [b.cross_attn_token_to_image.q_proj.register_forward_pre_hook(hook)
           for b in md.transformer.layers]

def run_decoder(img_emb, coords, labels):
    sparse, dense = sam.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
    md(image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
       sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)

def pool(qs, kind):
    """qs: list of [tokens_i, D] -> [N, D] descriptor matrix (L2-normalized)."""
    out = []
    for q in qs:
        t = torch.from_numpy(q)                      # [tokens, D]
        if kind == "mean":  v = t.mean(0)
        elif kind == "last": v = t[-1]
        elif kind == "max":  v = t.amax(0)
        elif kind == "attn":
            w = torch.softmax(t.norm(dim=1), dim=0)  # [tokens]
            v = (w[:, None] * t).sum(0)
        out.append(F.normalize(v, dim=-1).numpy())
    return np.stack(out, 0)

DESCS = ["mean", "last", "max", "attn"]

def evaluate(qs, tag):
    print(f"\n=== {tag}: {len(qs)} decoder-token descriptors ===", flush=True)
    row = {}
    for d in DESCS:
        X = pool(qs, d)
        c, cs = perm_ratio(X, K)
        dec = "FIRE" if c <= 0.4 else "SKIP"
        row[d] = dict(c=round(c, 3), std=round(cs, 3), decision=dec)
        print(f"  {d:5s}: c={c:.3f}±{cs:.3f} -> {dec}", flush=True)
    return row

# ---- Protocol A: v2 synthetic ----
raw.clear(); torch.manual_seed(0)
for i in range(50):
    s = 30.0 + 6.0 * i
    img = (torch.randn(1, 3, 1024, 1024) * s + 128.0).clamp(0, 255).cuda() / 255.0
    emb = sam.image_encoder(img)
    n = 2 + (i % 3)
    coords = torch.rand(1, n, 2, device=DEV) * 1024.0; labels = torch.ones(1, n, device=DEV)
    run_decoder(emb, coords, labels)
A = evaluate(list(raw), "Protocol A (v2 synthetic, paper regime)")

# ---- Protocol B: real COCO, points at GT object centers ----
coco = COCO(f"{WS}/coco/annotations/instances_val2017.json")
imgdir = f"{WS}/coco/val2017"
pred = SamPredictor(sam)
ids = coco.getImgIds(); np.random.default_rng(0).shuffle(ids)
raw.clear(); done = 0
for iid in ids:
    info = coco.loadImgs(iid)[0]; p = os.path.join(imgdir, info["file_name"])
    if not os.path.exists(p): continue
    im = np.array(Image.open(p).convert("RGB"))
    anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=False)) if a["area"] > 1024][:3]
    if not anns: continue
    pred.set_image(im); emb = pred.features
    for a in anns:
        x, y, w, h = a["bbox"]; cx, cy = x + w / 2, y + h / 2
        pt = pred.transform.apply_coords(np.array([[cx, cy]]), im.shape[:2])
        coords = torch.as_tensor(pt, dtype=torch.float, device=DEV)[None]
        labels = torch.ones(1, 1, device=DEV)
        run_decoder(emb, coords, labels); done += 1
    if done >= 100: break
B = evaluate(list(raw), "Protocol B (real COCO, object-center points)")
for h in handles: h.remove()

# ---- summary ----
def robust(row): return len({v["decision"] for v in row.values()}) == 1 and \
                        next(iter(row.values()))["decision"] == "FIRE"
res = dict(
    model="SAM-B", site="mask_decoder.cross_attn_token_to_image.q_proj (both layers)",
    metric="permute-dims compactness (paper's exact), K=4, fire iff c<=0.4",
    paper_reported=0.189, descriptors=DESCS,
    protocol_A_synthetic=A, protocol_B_real_coco=B,
    A_all_fire=robust(A), B_all_fire=robust(B),
)
os.makedirs(f"{WS}/results", exist_ok=True)
json.dump(res, open(f"{WS}/results/sam_descriptor_ablation_results.json", "w"), indent=2)
print("\n" + json.dumps({k: res[k] for k in ["A_all_fire", "B_all_fire"]}, indent=2))
print("saved -> results/sam_descriptor_ablation_results.json")
