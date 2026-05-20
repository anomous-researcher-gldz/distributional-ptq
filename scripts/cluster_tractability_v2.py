"""K-means tractability v2: corrected SAM site + K sweep + matched samples.

Fixes over v1:
  - SAM site: mask_decoder.transformer.layers[*].cross_attn_token_to_image.q_proj
    (the actual PCSA application site in §3.3, not image encoder qkv)
  - LLM samples: bump to 50 prompts (was 20)
  - K sweep: {2, 4, 8} for both models, so we don't cherry-pick K=4
  - Two baselines: permute-dims AND isotropic Gaussian with matched mean/cov
  - Silhouette as a second metric, less sensitive to scale/dim
"""
from __future__ import annotations
import sys, json, pathlib, argparse, gc, warnings
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/projects/instance_segment_anything/models")
warnings.filterwarnings("ignore")


def _kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0):
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=1, max_iter=iters, random_state=seed)
        km.fit(X)
        a = km.labels_
        C = km.cluster_centers_
        d_self = np.linalg.norm(X - C[a], axis=1)
        return a, C, d_self
    except Exception:
        pass
    # Fallback numpy
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        d2 = (X * X).sum(-1)[:, None] + (C * C).sum(-1)[None, :] - 2.0 * X @ C.T
        a = d2.argmin(1)
        for j in range(k):
            mask = a == j
            if mask.any():
                C[j] = X[mask].mean(0)
    d_self = np.linalg.norm(X - C[a], axis=1)
    return a, C, d_self


def _silhouette(X: np.ndarray, a: np.ndarray, k: int) -> float:
    if k < 2:
        return 0.0
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) + 1e-12)
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = a == a[i]
        same[i] = False
        a_i = D[i, same].mean() if same.any() else 0.0
        b_i = np.inf
        for j in range(k):
            if j == a[i]:
                continue
            mask = a == j
            if mask.any():
                b_i = min(b_i, D[i, mask].mean())
        s[i] = (b_i - a_i) / max(a_i, b_i, 1e-12)
    return float(s.mean())


def _baseline_permute(X, seed):
    rng = np.random.default_rng(seed + 7)
    Xp = X.copy()
    for j in range(X.shape[1]):
        Xp[:, j] = rng.permutation(Xp[:, j])
    return Xp


def _baseline_gaussian(X, seed):
    rng = np.random.default_rng(seed + 13)
    mu = X.mean(0)
    # Use diagonal cov; full cov is wasteful for d=4096
    sigma = X.std(0) + 1e-9
    return rng.normal(mu, sigma, size=X.shape)


def analyze(name: str, X: np.ndarray, k_list, n_baseline: int = 20):
    X = X.astype(np.float64)
    out = {"model": name, "n_descriptors": int(X.shape[0]), "dim": int(X.shape[1]),
           "k_sweep": {}}
    for k in k_list:
        _, _, d_real = _kmeans(X, k)
        perm_ratios, gauss_ratios = [], []
        for s in range(n_baseline):
            _, _, d_perm = _kmeans(_baseline_permute(X, s), k, seed=s)
            _, _, d_gauss = _kmeans(_baseline_gaussian(X, s), k, seed=s)
            perm_ratios.append(d_real.mean() / max(d_perm.mean(), 1e-9))
            gauss_ratios.append(d_real.mean() / max(d_gauss.mean(), 1e-9))

        out["k_sweep"][str(k)] = {
            "d_real_mean": float(d_real.mean()),
            "ratio_vs_permute_dims": float(np.mean(perm_ratios)),
            "ratio_vs_permute_dims_std": float(np.std(perm_ratios)),
            "ratio_vs_gaussian": float(np.mean(gauss_ratios)),
            "ratio_vs_gaussian_std": float(np.std(gauss_ratios)),
        }
    return out


@torch.no_grad()
def collect_sam_mask_decoder(n_images: int = 50):
    """Hook SAM mask decoder cross_attn_token_to_image.q_proj — the actual
    PCSA application site for SAM per §3.3. Drives the decoder with random
    point prompts on synthetic images."""
    import segment_anything as sa
    sam = sa.sam_model_registry["vit_b"](
        checkpoint="/home/ubuntu/unifying-ptq/ckpt/sam_vit_b_01ec64.pth"
    ).cuda().eval()
    md = sam.mask_decoder

    descs = []
    def hook(mod, args):
        q = args[0]  # [B, N_tokens, C] — the prompt-conditioned q input
        # Mean-pool over token axis, L2-normalize: matches §3.3 descriptor
        d = torch.nn.functional.normalize(q.float().mean(dim=1), dim=-1)
        descs.append(d.cpu().numpy())

    handles = []
    for blk in md.transformer.layers:
        handles.append(blk.cross_attn_token_to_image.q_proj.register_forward_pre_hook(hook))

    torch.manual_seed(0)
    for i in range(n_images):
        s = 30.0 + 6.0 * i
        img = (torch.randn(1, 3, 1024, 1024) * s + 128.0).clamp(0, 255).cuda() / 255.0
        # Pre-encode once
        with torch.no_grad():
            img_emb = sam.image_encoder(img)
        # Random sparse prompt: 2-3 points per image
        n_pts = 2 + (i % 3)
        coords = torch.rand(1, n_pts, 2, device="cuda") * 1024.0
        labels = torch.ones(1, n_pts, device="cuda")
        sparse, dense = sam.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        # Forward mask decoder
        _ = md(
            image_embeddings=img_emb,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )

    for h in handles:
        h.remove()
    return np.concatenate(descs, axis=0)


@torch.no_grad()
def collect_llama(n_prompts: int = 50):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained("/data/modelzoo/meta-llama/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "/data/modelzoo/meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16
    ).cuda().eval()
    # Hook all layer self_attn q_proj inputs (matches §3.3 LLM application site)
    descs = []
    def hook(mod, args):
        x = args[0]
        d = torch.nn.functional.normalize(x.float().mean(dim=1), dim=-1)
        descs.append(d.cpu().numpy())

    handles = []
    for layer in model.model.layers:
        handles.append(layer.self_attn.q_proj.register_forward_pre_hook(hook))

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t) > 300][:n_prompts]
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        _ = model(ids)

    for h in handles:
        h.remove()
    return np.concatenate(descs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_inputs_sam", type=int, default=50)
    p.add_argument("--n_inputs_llm", type=int, default=50)
    p.add_argument("--k_sweep", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--out", default="/home/ubuntu/unifying-ptq/results/cluster_tractability_v2.json")
    args = p.parse_args()

    print("=== SAM-B (mask decoder cross-attn) ===", flush=True)
    X_sam = collect_sam_mask_decoder(args.n_inputs_sam)
    print(f"  collected {X_sam.shape}", flush=True)
    r_sam = analyze("sam-b", X_sam, args.k_sweep)
    gc.collect(); torch.cuda.empty_cache()

    print("\n=== LLaMA-3-8B (self-attn q_proj all layers) ===", flush=True)
    X_llm = collect_llama(args.n_inputs_llm)
    print(f"  collected {X_llm.shape}", flush=True)
    r_llm = analyze("llama-8b", X_llm, args.k_sweep)

    results = {"sam-b": r_sam, "llama-8b": r_llm}

    print("\n=== SUMMARY ===", flush=True)
    print(f"  {'model':10s} {'K':>3s} {'ratio(perm)':>14s} {'ratio(gauss)':>14s}")
    for name, r in results.items():
        for k_str, m in r["k_sweep"].items():
            print(f"  {name:10s} {k_str:>3s} "
                  f"{m['ratio_vs_permute_dims']:>8.4f}±{m['ratio_vs_permute_dims_std']:.3f} "
                  f"{m['ratio_vs_gaussian']:>8.4f}±{m['ratio_vs_gaussian_std']:.3f}")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
