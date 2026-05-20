"""K-means tractability comparison: SAM-B vs LLaMA-3-8B per-input descriptors.

For each model, capture per-input mean-pooled L2-normalized descriptors from
the Linear-input we'd route PCSA on, run K-means with K=4, then compare the
mean within-cluster distance d_self to a permute-dims baseline that destroys
structure but keeps marginals.

If real_d_self / baseline_d_self << 1, the descriptors cluster meaningfully;
if the ratio is near 1, K-means is splitting Gaussian noise.
"""
from __future__ import annotations
import sys, json, pathlib, argparse, gc
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/projects/instance_segment_anything/models")


def _kmeans(X: np.ndarray, k: int, iters: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        for j in range(k):
            mask = a == j
            if mask.any():
                C[j] = X[mask].mean(0)
    d_self = np.linalg.norm(X - C[a], axis=1)
    return a, C, d_self


def _baseline_permute(X: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed + 7)
    Xp = X.copy()
    for j in range(X.shape[1]):
        Xp[:, j] = rng.permutation(Xp[:, j])
    return Xp


def _normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


@torch.no_grad()
def collect_sam_descriptors(n: int = 20):
    """Mean-pooled L2-normalized descriptors from SAM-B mask-decoder cross-attn
    q-projection input (the PCSA application site for SAM)."""
    import segment_anything as sa
    sam = sa.sam_model_registry["vit_b"](
        checkpoint="/home/ubuntu/unifying-ptq/ckpt/sam_vit_b_01ec64.pth"
    ).to(torch.bfloat16).cuda().eval()

    # Hook the image encoder's first attention q_proj (proxy for prompt-conditioning):
    enc = sam.image_encoder
    target = enc.blocks[0].attn.qkv  # input shape: [B, N, C]
    descs = []
    def hook(mod, args):
        x = args[0]
        if isinstance(x, torch.Tensor):
            d = x.float().mean(dim=tuple(range(1, x.ndim - 1)))  # mean over spatial dims
            d = torch.nn.functional.normalize(d, dim=-1)
            descs.append(d.cpu().numpy())
    h = target.register_forward_pre_hook(hook)

    # Use varied images: random Gaussian content with different scales
    torch.manual_seed(0)
    for i in range(n):
        s = 40.0 + 8.0 * i  # varying content scale
        x = (torch.randn(1, 3, 1024, 1024) * s + 128.0).clamp(0, 255)
        x = x.to(torch.bfloat16).cuda() / 255.0
        _ = enc(x)
    h.remove()
    out = np.concatenate(descs, axis=0)
    return out


@torch.no_grad()
def collect_llama_descriptors(n: int = 20):
    """Mean-pooled L2-normalized descriptors from LLaMA-3-8B layer-0 self-attn
    q-projection input — the PCSA application site for LLMs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained("/data/modelzoo/meta-llama/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "/data/modelzoo/meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16
    ).cuda().eval()
    target = model.model.layers[0].self_attn.q_proj
    descs = []
    def hook(mod, args):
        x = args[0]
        if isinstance(x, torch.Tensor):
            d = x.float().mean(dim=1)  # mean over sequence
            d = torch.nn.functional.normalize(d, dim=-1)
            descs.append(d.cpu().numpy())
    h = target.register_forward_pre_hook(hook)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t) > 200][:n]
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        _ = model(ids)
    h.remove()
    out = np.concatenate(descs, axis=0)
    return out


def analyze(name: str, X: np.ndarray, k: int = 4, n_baseline: int = 20):
    X = X.astype(np.float64)
    # Real K-means
    _, _, d_real = _kmeans(X, k)
    # Permute-dims baseline (avg over n_baseline seeds)
    ratios, d_perm_means = [], []
    for s in range(n_baseline):
        Xp = _baseline_permute(X, seed=s)
        _, _, d_perm = _kmeans(Xp, k, seed=s)
        d_perm_means.append(d_perm.mean())
        ratios.append(d_real.mean() / max(d_perm.mean(), 1e-8))
    return {
        "model": name,
        "n_descriptors": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "k": k,
        "d_real_mean": float(d_real.mean()),
        "d_real_median": float(np.median(d_real)),
        "d_perm_mean_over_seeds": float(np.mean(d_perm_means)),
        "compactness_ratio": float(np.mean(ratios)),
        "compactness_ratio_std": float(np.std(ratios)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_inputs", type=int, default=20)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--out", default="/home/ubuntu/unifying-ptq/results/cluster_tractability.json")
    args = p.parse_args()

    results = {}
    for name, collector in [("sam-b", collect_sam_descriptors),
                            ("llama-8b", collect_llama_descriptors)]:
        print(f"\n=== {name} ===", flush=True)
        X = collector(args.n_inputs)
        print(f"  collected {X.shape[0]} descriptors of dim {X.shape[1]}", flush=True)
        r = analyze(name, X, k=args.k)
        print(f"  d_real={r['d_real_mean']:.4f}  d_baseline={r['d_perm_mean_over_seeds']:.4f}  "
              f"ratio={r['compactness_ratio']:.4f} (std={r['compactness_ratio_std']:.4f})",
              flush=True)
        results[name] = r
        gc.collect(); torch.cuda.empty_cache()

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
