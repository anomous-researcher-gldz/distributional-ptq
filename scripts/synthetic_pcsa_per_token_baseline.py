"""G1: Synthetic clusterability sweep with PER-TOKEN baseline (not per-tensor).

The original sweep compared per-anchor scale against a global per-tensor scale.
A reviewer would object that the per-tensor baseline is weak: standard W4A4
LLM kernels use per-token activation scales (dynamic, per row of the activation
matrix). The fair question is: does PCSA help OVER per-token scaling?

This script replays the same K_true=4, D=256 mixture-of-modes sweep, but
adds a per-token-scale baseline and reports BOTH gain measurements:
    PCSA vs per-tensor   (loose baseline, original)
    PCSA vs per-token    (tight baseline, this script)

Per-token = one scale per (input, token) pair = max-abs over the D-dim row.
Per-anchor = K-means scale per (input, anchor) pair, broadcast across tokens.

If PCSA still beats per-token, the cluster-routing mechanism delivers value
beyond what dynamic per-row scaling captures.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
from sklearn.cluster import KMeans

K_TRUE = 4
D = 256
N = 1000
B = 64
SEPARATIONS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
N_BASELINE = 10
SEED = 0
QMAX = 7  # INT4 symmetric


def make_dataset(S, rng):
    centers = rng.normal(0, 1, size=(K_TRUE, D))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers * S
    a = rng.integers(0, K_TRUE, size=N)
    desc = centers[a] + 0.1 * rng.normal(0, 1, size=(N, D))
    desc_n = desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-9)
    mag = np.linalg.norm(centers, axis=1)[a]
    tok = rng.normal(0, 1, size=(N, B, D))
    tok = tok / np.linalg.norm(tok, axis=-1, keepdims=True)
    acts = tok * mag[:, None, None] * (1.0 + 0.05 * rng.normal(size=(N, 1, 1)))
    return desc_n, acts, a


def compactness_ratio(desc, k):
    km = KMeans(n_clusters=k, n_init=1, random_state=0, max_iter=50).fit(desc)
    d_real = np.linalg.norm(desc - km.cluster_centers_[km.labels_], axis=1).mean()
    ratios = []
    for s in range(N_BASELINE):
        rng2 = np.random.default_rng(s + 7)
        Xp = desc.copy()
        for j in range(desc.shape[1]):
            Xp[:, j] = rng2.permutation(Xp[:, j])
        kmp = KMeans(n_clusters=k, n_init=1, random_state=s, max_iter=50).fit(Xp)
        d_perm = np.linalg.norm(Xp - kmp.cluster_centers_[kmp.labels_], axis=1).mean()
        ratios.append(d_real / max(d_perm, 1e-9))
    return float(np.mean(ratios))


def mse_quant(acts, scale):
    """scale broadcasts to acts.shape; returns MSE after INT4 round/clip/dequant."""
    q = np.round(acts / np.maximum(scale, 1e-9))
    q = np.clip(q, -QMAX, QMAX)
    return float(((acts - q * scale) ** 2).mean())


def baselines_and_pcsa(desc, acts, k):
    N_, B_, D_ = acts.shape

    # (a) per-tensor: one scale for entire activation tensor
    s_tensor = np.full((N_, 1, 1), np.abs(acts).max() / QMAX)
    mse_tensor = mse_quant(acts, s_tensor)

    # (b) per-token: one scale per (input, token) row = max-abs over D-dim
    s_token = np.abs(acts).max(axis=-1, keepdims=True) / QMAX  # [N, B, 1]
    mse_token = mse_quant(acts, s_token)

    # (c) per-anchor (PCSA): K-means anchor, scale = cluster max-abs / QMAX
    km = KMeans(n_clusters=k, n_init=1, random_state=0, max_iter=50).fit(desc)
    labels = km.labels_
    anchor_scales = np.zeros((k,))
    for j in range(k):
        m = labels == j
        if m.any():
            anchor_scales[j] = np.abs(acts[m]).max() / QMAX
    s_anchor = anchor_scales[labels].reshape(N_, 1, 1)
    mse_anchor = mse_quant(acts, s_anchor)

    return {
        "mse_per_tensor": mse_tensor,
        "mse_per_token": mse_token,
        "mse_per_anchor": mse_anchor,
        "gain_pcsa_vs_tensor_pct": 100.0 * (mse_tensor - mse_anchor) / max(mse_tensor, 1e-12),
        "gain_pcsa_vs_token_pct": 100.0 * (mse_token - mse_anchor) / max(mse_token, 1e-12),
    }


def main():
    rows = []
    print(f"{'S':>6s}  {'compact':>8s}  {'MSE_tens':>10s}  {'MSE_tok':>10s}  {'MSE_anch':>10s}  {'vs_tens%':>9s}  {'vs_tok%':>8s}")
    for S in SEPARATIONS:
        rng_s = np.random.default_rng(SEED + int(S * 100))
        desc, acts, _ = make_dataset(S, rng_s)
        c = compactness_ratio(desc, K_TRUE)
        r = baselines_and_pcsa(desc, acts, K_TRUE)
        print(f"  {S:6.2f}  {c:8.4f}  {r['mse_per_tensor']:10.4f}  "
              f"{r['mse_per_token']:10.4f}  {r['mse_per_anchor']:10.4f}  "
              f"{r['gain_pcsa_vs_tensor_pct']:9.3f}  {r['gain_pcsa_vs_token_pct']:8.3f}")
        rows.append({"separation_S": S, "compactness_ratio": c, **r})

    out = {"K_true": K_TRUE, "D": D, "N": N, "B": B, "rows": rows}
    pathlib.Path("/home/ubuntu/unifying-ptq/results/synthetic_pcsa_per_token_baseline.json").write_text(
        json.dumps(out, indent=2)
    )
    print("\n -> /home/ubuntu/unifying-ptq/results/synthetic_pcsa_per_token_baseline.json")


if __name__ == "__main__":
    main()
