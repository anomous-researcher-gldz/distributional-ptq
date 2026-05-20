"""Negative controls for PCSA mechanism.

Three settings, each fixing K_calib=4:

(A) Clusters AND scale-correlated: descriptor mode determines activation
    magnitude (the realistic case). PCSA should help.
(B) Clusters but scale-DECORRELATED: descriptor modes are well-separated
    (compactness ratio low) but per-input activation magnitude is shuffled
    independent of descriptor mode. PCSA should NOT help, because the
    anchor's per-cluster scale is unrelated to the actual per-input scale.
(C) K_calib != K_true: descriptors have K_true=8 modes but PCSA calibrates
    with K_calib=4 anchors. PCSA capacity is bottlenecked.

If PCSA gain >> 0 only in (A), the mechanism is identified as
"cluster-correlated activation magnitude," not just clusterability.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
from sklearn.cluster import KMeans

D = 256
N = 1000
B = 64
QMAX = 7
S_MAIN = 8.0  # high-clusterability regime
SEED = 0


def make_corr(rng, K_true, S):
    """(A) Tight clusters with magnitude tied to cluster (PCSA-friendly)."""
    centers = rng.normal(0, 1, size=(K_true, D))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers * S
    a = rng.integers(0, K_true, size=N)
    desc = centers[a] + 0.1 * rng.normal(0, 1, size=(N, D))
    desc_n = desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-9)
    mag = np.linalg.norm(centers, axis=1)[a]
    tok = rng.normal(0, 1, size=(N, B, D))
    tok = tok / np.linalg.norm(tok, axis=-1, keepdims=True)
    acts = tok * mag[:, None, None] * (1.0 + 0.05 * rng.normal(size=(N, 1, 1)))
    return desc_n, acts


def make_decorr(rng, K_true, S):
    """(B) Tight clusters but per-input magnitude shuffled across cluster
    membership (PCSA-adversarial: clusters exist but scale is independent)."""
    centers = rng.normal(0, 1, size=(K_true, D))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers * S
    a = rng.integers(0, K_true, size=N)
    desc = centers[a] + 0.1 * rng.normal(0, 1, size=(N, D))
    desc_n = desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-9)
    # Shuffle magnitudes independent of cluster membership
    mag_pool = np.linalg.norm(centers, axis=1)[rng.integers(0, K_true, size=N)]
    tok = rng.normal(0, 1, size=(N, B, D))
    tok = tok / np.linalg.norm(tok, axis=-1, keepdims=True)
    acts = tok * mag_pool[:, None, None] * (1.0 + 0.05 * rng.normal(size=(N, 1, 1)))
    return desc_n, acts


def make_K_mismatch(rng, K_true, S):
    """(C) K_true=8 modes but K_calib=4 anchors (under-capacity)."""
    return make_corr(rng, K_true, S)


def compactness_ratio(desc, k):
    km = KMeans(n_clusters=k, n_init=1, random_state=0, max_iter=50).fit(desc)
    d_real = np.linalg.norm(desc - km.cluster_centers_[km.labels_], axis=1).mean()
    ratios = []
    for s in range(10):
        rng = np.random.default_rng(s + 7)
        Xp = desc.copy()
        for j in range(desc.shape[1]):
            Xp[:, j] = rng.permutation(Xp[:, j])
        kmp = KMeans(n_clusters=k, n_init=1, random_state=s, max_iter=50).fit(Xp)
        d_perm = np.linalg.norm(Xp - kmp.cluster_centers_[kmp.labels_], axis=1).mean()
        ratios.append(d_real / max(d_perm, 1e-9))
    return float(np.mean(ratios))


def pcsa_gain(desc, acts, k):
    g_scale = (np.abs(acts).max() / QMAX) * np.ones((N, 1, 1))
    mse_g = ((acts - np.clip(np.round(acts / g_scale), -QMAX, QMAX) * g_scale) ** 2).mean()
    km = KMeans(n_clusters=k, n_init=1, random_state=0, max_iter=50).fit(desc)
    labels = km.labels_
    anchors = np.zeros((k, 1, 1))
    for j in range(k):
        m = labels == j
        if m.any():
            anchors[j] = np.abs(acts[m]).max() / QMAX
    pa = anchors[labels].reshape(N, 1, 1)
    mse_a = ((acts - np.clip(np.round(acts / pa), -QMAX, QMAX) * pa) ** 2).mean()
    return float(mse_g), float(mse_a), 100.0 * (mse_g - mse_a) / max(mse_g, 1e-12)


def main():
    print(f"{'setting':25s} {'compact':>10s} {'MSE_g':>10s} {'MSE_a':>10s} {'PCSA%':>8s}")

    rows = []
    # (A) clusters + scale-correlated
    rng = np.random.default_rng(SEED)
    d, a = make_corr(rng, 4, S_MAIN)
    c = compactness_ratio(d, 4)
    mg, ma, gain = pcsa_gain(d, a, 4)
    print(f"  A: clusters + corr        {c:10.4f} {mg:10.4f} {ma:10.4f} {gain:8.2f}")
    rows.append({"setting": "A_clusters_correlated", "K_calib": 4, "K_true": 4,
                 "compactness": c, "mse_global": mg, "mse_anchor": ma, "pcsa_gain_pct": gain})

    # (B) clusters but scale-decorrelated
    rng = np.random.default_rng(SEED + 100)
    d, a = make_decorr(rng, 4, S_MAIN)
    c = compactness_ratio(d, 4)
    mg, ma, gain = pcsa_gain(d, a, 4)
    print(f"  B: clusters NO correlation {c:10.4f} {mg:10.4f} {ma:10.4f} {gain:8.2f}")
    rows.append({"setting": "B_clusters_decorrelated", "K_calib": 4, "K_true": 4,
                 "compactness": c, "mse_global": mg, "mse_anchor": ma, "pcsa_gain_pct": gain})

    # (C) K_true=8 but K_calib=4
    rng = np.random.default_rng(SEED + 200)
    d, a = make_K_mismatch(rng, 8, S_MAIN)
    c = compactness_ratio(d, 4)
    mg, ma, gain = pcsa_gain(d, a, 4)
    print(f"  C: K_true=8 K_calib=4      {c:10.4f} {mg:10.4f} {ma:10.4f} {gain:8.2f}")
    rows.append({"setting": "C_K_mismatch", "K_calib": 4, "K_true": 8,
                 "compactness": c, "mse_global": mg, "mse_anchor": ma, "pcsa_gain_pct": gain})

    pathlib.Path("/home/ubuntu/unifying-ptq/results/synthetic_pcsa_negative_controls.json").write_text(
        json.dumps({"S": S_MAIN, "settings": rows}, indent=2)
    )
    print("\n -> /home/ubuntu/unifying-ptq/results/synthetic_pcsa_negative_controls.json")


if __name__ == "__main__":
    main()
