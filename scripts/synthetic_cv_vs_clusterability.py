"""Synthetic experiment: show CV and K-means compactness can diverge.

Construct two synthetic descriptor sets in R^256:
  A: 4-mode Gaussian mixture with well-separated means and small per-mode std
     -> low per-feature CV (centers cancel in marginal mean), HIGH clusterability
  B: isotropic Gaussian (no structure)
     -> matched per-feature CV by design, LOW clusterability

Compactness ratio (real/permute-dims baseline) should be << 1 for A and ~ 1 for B,
while their per-feature CVs are matched. Conclusion: CV does not predict
clusterability; the compactness ratio does.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
from sklearn.cluster import KMeans

rng = np.random.default_rng(0)
D = 256
N = 800
K_TRUE = 4

# --- Dataset A: K_TRUE-mode mixture. Means scattered on unit sphere; per-mode std small.
centers = rng.normal(0, 1, size=(K_TRUE, D))
centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
labels = rng.integers(0, K_TRUE, size=N)
A = centers[labels] + 0.1 * rng.normal(0, 1, size=(N, D))
A = A / np.linalg.norm(A, axis=1, keepdims=True)

# --- Dataset B: isotropic Gaussian on unit sphere, no cluster structure.
B = rng.normal(0, 1, size=(N, D))
B = B / np.linalg.norm(B, axis=1, keepdims=True)

# Match per-feature variance: scale B so that mean per-feature variance matches A.
sA = A.std(0).mean()
sB = B.std(0).mean()
B = B * (sA / sB)
B = B / np.linalg.norm(B, axis=1, keepdims=True)


def cv_per_feature(X):
    return (X.std(0) / (np.abs(X.mean(0)) + 1e-9)).mean()


def compactness_ratio(X, k, n_seeds=20):
    km = KMeans(n_clusters=k, n_init=1, random_state=0, max_iter=50)
    km.fit(X)
    d_real = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1).mean()
    ratios = []
    for s in range(n_seeds):
        rng2 = np.random.default_rng(s + 7)
        Xp = X.copy()
        for j in range(X.shape[1]):
            Xp[:, j] = rng2.permutation(Xp[:, j])
        kmp = KMeans(n_clusters=k, n_init=1, random_state=s, max_iter=50)
        kmp.fit(Xp)
        d_perm = np.linalg.norm(Xp - kmp.cluster_centers_[kmp.labels_], axis=1).mean()
        ratios.append(d_real / max(d_perm, 1e-9))
    return float(np.mean(ratios)), float(np.std(ratios))


cvA, cvB = cv_per_feature(A), cv_per_feature(B)
rA, sA_ = compactness_ratio(A, 4)
rB, sB_ = compactness_ratio(B, 4)

print(f"Dataset A (4-mode mixture):  mean per-feature CV = {cvA:.3f},  "
      f"K=4 compactness = {rA:.3f} ± {sA_:.4f}")
print(f"Dataset B (isotropic):       mean per-feature CV = {cvB:.3f},  "
      f"K=4 compactness = {rB:.3f} ± {sB_:.4f}")
print(f"\nCV ratio A/B = {cvA/cvB:.2f}x (matched)")
print(f"Compactness ratio A/B = {rA/rB:.3f}x (A {(1-rA)*100:.0f}% tighter; B {(1-rB)*100:.0f}% tighter)")

out = {
    "A_4mode_mixture": {"cv_mean": cvA, "compactness_k4": rA, "compactness_k4_std": sA_},
    "B_isotropic":     {"cv_mean": cvB, "compactness_k4": rB, "compactness_k4_std": sB_},
}
pathlib.Path("/home/ubuntu/unifying-ptq/results/synthetic_cv_vs_clust.json").write_text(
    json.dumps(out, indent=2)
)
print("\n -> /home/ubuntu/unifying-ptq/results/synthetic_cv_vs_clust.json")
