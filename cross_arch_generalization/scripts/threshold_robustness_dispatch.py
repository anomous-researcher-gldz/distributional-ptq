"""supplementary study analyses on EXISTING per-layer feature JSONs (zero GPU).

Answers, with real numbers already in the repo:
  Q4 -- can the thresholded dispatch be *learned*? Fit a logistic regression on
        {frac3, kurt, skew} and check it recovers the hand-set gate, and whether
        a learned regressor beats the frozen a+b*p line at predicting DBAF gain.
  Q1 -- how stable is the gate under threshold perturbation? Jitter each bound
        +/-20% and measure gate-decision agreement per config.
"""
import json, glob, os, numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

BASE = "/home/ubuntu/distributional-ptq/results/S4-cross-model-layer-analysis"
files = sorted(glob.glob(os.path.join(BASE, "*.json")))
files = [f for f in files if os.path.basename(f) != "summary.json"]

rows = []
for f in files:
    model = os.path.splitext(os.path.basename(f))[0]
    d = json.load(open(f))
    for L in d["layers"]:
        rows.append({"model": model, **L})

print(f"Loaded {len(rows)} layers from {len(files)} configs: "
      f"{sorted(set(r['model'] for r in rows))}\n")

def col(key):
    return np.array([r.get(key, np.nan) for r in rows], float)

# ---------- Weight-side gate features ----------
frac3 = col("w_frac3"); kurt = col("w_kurt"); skew = np.abs(col("w_skew"))
gate  = np.array([1 if r.get("w_gate") else 0 for r in rows])
gain  = col("w_gain_force_pct")          # forced-DBAF MSE reduction %
groups = np.array([r["model"] for r in rows])

X = np.column_stack([frac3, np.log10(np.clip(kurt,1e-3,None)), skew])
good = np.isfinite(X).all(1) & np.isfinite(gain)
X, y_gate, y_gain, g = X[good], gate[good], gain[good], groups[good]
frac3g = frac3[good]

print("="*70)
print("Q4a  LEARNED DISPATCHER vs HAND-SET GATE")
print("="*70)
print(f"Hand-set gate fires on {y_gate.mean()*100:.1f}% of {len(y_gate)} layers.")
# Leave-one-MODEL-out: train dispatcher on all-but-one architecture, test on held-out
logo = LeaveOneGroupOut()
aucs = {}
for tr, te in logo.split(X, y_gate, g):
    m = groups[good][te][0]
    if len(np.unique(y_gate[tr])) < 2 or len(np.unique(y_gate[te])) < 2:
        aucs[m] = None; continue
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y_gate[tr])
    p = clf.predict_proba(X[te])[:,1]
    aucs[m] = roc_auc_score(y_gate[te], p)
print("Leave-one-architecture-out AUC (learned dispatcher recovers the gate on a")
print("held-out family it never saw):")
for m,a in aucs.items():
    print(f"   {m:14s}: {'n/a (single class)' if a is None else f'{a:.3f}'}")
valid=[a for a in aucs.values() if a is not None]
if valid: print(f"   mean AUC = {np.mean(valid):.3f}")

print()
print("="*70)
print("Q4b  LEARNED REGRESSOR vs FROZEN a+b*p LINE (predicting DBAF gain)")
print("="*70)
# frozen line: gain ~ a + b*frac3, single feature (paper's Eq)
r_line,_ = pearsonr(frac3g, y_gain)
# leave-one-model-out predictions, single-feature linear
pred_line = cross_val_predict(LinearRegression(), frac3g.reshape(-1,1), y_gain,
                              groups=g, cv=logo)
r_line_loo,_ = pearsonr(pred_line, y_gain)
# multi-feature learned regressor, leave-one-model-out
pred_multi = cross_val_predict(LinearRegression(), X, y_gain, groups=g, cv=logo)
r_multi,_ = pearsonr(pred_multi, y_gain)
print(f"Single-feature frac3 -> gain, in-sample Pearson r = {r_line:.3f} "
      f"(paper reports ~0.56 on LLaMA)")
print(f"Leave-one-architecture-out r, frozen 1-feature line = {r_line_loo:.3f}")
print(f"Leave-one-architecture-out r, learned 3-feature model = {r_multi:.3f}")

print()
print("="*70)
print("Q1  THRESHOLD STABILITY under +/-20% jitter of every gate bound")
print("="*70)
# reconstruct gate decision from raw stats so we can perturb bounds
def gate_decision(skew_a, kurt_a, frac3_a, s_th=0.7, k_lo=3.0, k_hi=30.0,
                  f_lo=1e-4, f_hi=2e-2):
    return (skew_a<=s_th)&(kurt_a>=k_lo)&(kurt_a<=k_hi)&(frac3_a>=f_lo)&(frac3_a<=f_hi)

skew_a = np.abs(col("w_skew")); kurt_a=col("w_kurt"); frac3_a=col("w_frac3")
m2 = np.isfinite(skew_a)&np.isfinite(kurt_a)&np.isfinite(frac3_a)
skew_a,kurt_a,frac3_a,grp2 = skew_a[m2],kurt_a[m2],frac3_a[m2],groups[m2]
base = gate_decision(skew_a,kurt_a,frac3_a)
rng = np.random.default_rng(0)
agree=[]
for _ in range(200):
    j = lambda v: v*(1+rng.uniform(-0.2,0.2))
    pert = gate_decision(skew_a,kurt_a,frac3_a, j(0.7),j(3.0),j(30.0),j(1e-4),j(2e-2))
    agree.append((pert==base).mean())
agree=np.array(agree)
print(f"Per-layer gate decision agreement with default thresholds under 200 draws")
print(f"of simultaneous +/-20% jitter on ALL five bounds:")
print(f"   mean {agree.mean()*100:.1f}%   min {agree.min()*100:.1f}%   "
      f"p5 {np.percentile(agree,5)*100:.1f}%")
# per-config gate-pass rate stability
print("\nWeight gate-pass rate per config (default thresholds):")
for m in sorted(set(grp2)):
    sel=grp2==m
    print(f"   {m:14s}: {base[sel].mean()*100:5.1f}%  (n={sel.sum()})")
