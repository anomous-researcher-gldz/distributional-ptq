"""Synthetic sweep: does PCSA's quantization benefit track descriptor clusterability?

Construct synthetic per-input activation distributions with controlled cluster
structure, sweep the separation parameter, and at each setting compare
(a) descriptor K-means compactness (real / permute-dims baseline)
(b) quantization MSE with per-anchor scale routing vs global max-abs scale.

If PCSA's benefit is structurally tied to clusterability, the two quantities
should be monotonically related across the sweep.

Setup:
  - K_true = 4 modes, each describes a per-input "activation distribution"
    parameterized by a center vector mu_k in R^D
  - Mode separation S controls how far modes are pushed apart on the sphere
  - For each input i: pick mode k, sample
        x_i = mu_{k} + 0.1 * eps,  eps ~ N(0, I)
    Then "activation" is a [B, D]-shape tensor whose per-input max-abs depends
    on the mode (modes with higher-magnitude mu have larger per-input scale)
  - Compactness ratio: K-means on mean-pooled L2-normalized descriptors / permute-dims baseline at K=K_true
  - PCSA gain: MSE_global - MSE_per_anchor over 4-bit quantization, normalized by MSE_global

Output: JSON with sweep result + table-ready summary.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
from sklearn.cluster import KMeans

K_TRUE = 4
D = 256
N = 1000              # number of inputs
B = 64                # tokens per input
SEPARATIONS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
N_BASELINE = 10
SEED = 0
QMAX = 7              # INT4 symmetric


def make_dataset(S: float, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (descriptors [N, D], activations [N, B, D], assignments [N])."""
    # K random unit-mean centers, pushed apart by S along orthogonal directions
    centers = rng.normal(0, 1, size=(K_TRUE, D))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers * S  # scale separation
    a = rng.integers(0, K_TRUE, size=N)
    # Per-input descriptor: cluster mean + small noise
    desc = centers[a] + 0.1 * rng.normal(0, 1, size=(N, D))
    desc_n = desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-9)
    # Per-input activations: cluster-magnitude * unit-norm tokens
    # The magnitude depends on the mode index — modes with larger ||center|| produce larger activations
    # This is the structural reason PCSA's per-cluster scale would help
    mag = np.linalg.norm(centers, axis=1)[a]  # [N]
    tok = rng.normal(0, 1, size=(N, B, D))
    tok = tok / np.linalg.norm(tok, axis=-1, keepdims=True)
    acts = tok * mag[:, None, None] * (1.0 + 0.05 * rng.normal(size=(N, 1, 1)))
    return desc_n, acts, a


def compactness_ratio(desc: np.ndarray, k: int, rng):
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
    return float(np.mean(ratios)), float(np.std(ratios))


def quant_int4(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Symmetric INT4 quantization with given per-input scale [N, 1, 1]."""
    q = np.round(x / np.maximum(scale, 1e-9))
    q = np.clip(q, -QMAX, QMAX)
    return q * scale


def pcsa_benefit(desc: np.ndarray, acts: np.ndarray, k: int):
    """Compare MSE of global scale vs per-anchor (K-means) scale routing."""
    N_, B_, D_ = acts.shape
    # Global scale: single max-abs / qmax across all inputs
    g_scale = (np.abs(acts).max() / QMAX) * np.ones((N_, 1, 1))
    mse_global = ((acts - quant_int4(acts, g_scale)) ** 2).mean()

    # Per-anchor scale: K-means cluster descriptors, anchor scale = max-abs per cluster
    km = KMeans(n_clusters=k, n_init=1, random_state=0, max_iter=50).fit(desc)
    labels = km.labels_
    anchor_scales = np.zeros((k, 1, 1))
    for j in range(k):
        mask = labels == j
        if mask.any():
            anchor_scales[j] = np.abs(acts[mask]).max() / QMAX
    per_anchor_scale = anchor_scales[labels].reshape(N_, 1, 1)
    mse_anchor = ((acts - quant_int4(acts, per_anchor_scale)) ** 2).mean()

    rel_gain = 100.0 * (mse_global - mse_anchor) / max(mse_global, 1e-12)
    return float(mse_global), float(mse_anchor), float(rel_gain)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    print(f"{'S':>6s}  {'compactness':>14s}  {'MSE_global':>12s}  {'MSE_anchor':>12s}  {'PCSA_gain_%':>12s}")
    for S in SEPARATIONS:
        rng_s = np.random.default_rng(SEED + int(S * 100))
        desc, acts, _ = make_dataset(S, rng_s)
        r_mean, r_std = compactness_ratio(desc, K_TRUE, rng_s)
        mse_g, mse_a, gain = pcsa_benefit(desc, acts, K_TRUE)
        print(f"  {S:6.2f}  {r_mean:8.4f}±{r_std:.4f}  {mse_g:12.6f}  {mse_a:12.6f}  {gain:12.3f}")
        rows.append({
            "separation_S": S, "compactness_ratio": r_mean,
            "compactness_std": r_std, "mse_global": mse_g,
            "mse_anchor": mse_a, "pcsa_gain_pct": gain,
        })

    out = {
        "K_true": K_TRUE, "D": D, "N": N, "B": B,
        "separations": SEPARATIONS, "rows": rows,
    }
    pathlib.Path("/home/ubuntu/unifying-ptq/results/synthetic_pcsa_clusterability_sweep.json").write_text(
        json.dumps(out, indent=2)
    )
    print("\n -> /home/ubuntu/unifying-ptq/results/synthetic_pcsa_clusterability_sweep.json")


if __name__ == "__main__":
    main()
