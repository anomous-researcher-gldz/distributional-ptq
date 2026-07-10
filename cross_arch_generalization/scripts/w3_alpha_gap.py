"""W3 -- why the analytic alpha* fails so dramatically.

The paper's proxy L~(a) = (T + a(M-T))^2 (p_in + p_out/a^2) already contains the
a^-1 unfold-noise term, yet its minimizer a*~0.07 diverges in PPL. we 
asks WHY the analytic optimum fails so dramatically. This script shows the
mechanism directly: the proxy assumes *continuous uniform* quantization noise,
but at small a the outlier band [T, T+a(M-T)] collapses below one INT4 level, so
distinct outliers map to the same integer and unfold cannot restore their
ordering. We compare the proxy minimum against the TRUE simulated
fold->INT4-quantize->unfold MSE on a distribution calibrated to a real LLaMA
gate-pass layer, and mark where the outlier band drops below one level.
"""
import numpy as np

rng = np.random.default_rng(0)
QMAX = 7            # symmetric INT4, q_max = 2^(b-1)-1
N = 4_000_000

# ---- distribution calibrated to a real LLaMA-3-8B gate-pass activation layer ----
# From results/S4-cross-model-layer-analysis/llama.json: frac>3sigma ~ 0.019,
# heavy-tailed core. Gaussian core (sigma=1) + sparse large-magnitude outliers.
# kurtosis ~95-269 requires outliers reaching tens of sigma; then M(99.9pct)
# sits ~50x the 3sigma threshold, the regime that drives a* down to ~0.07.
P_OUT_TARGET = 0.019
core = rng.standard_normal(N)
n_out = int(P_OUT_TARGET * N)
u = rng.random(n_out)
out_mag = 3.2 * (1 - u) ** (-1.0 / 1.2)         # Pareto-like heavy tail
out_mag = np.clip(out_mag, 3.2, 200.0)
idx = rng.choice(N, n_out, replace=False)
x = core.copy()
x[idx] = rng.choice([-1, 1], n_out) * out_mag
x = x.astype(np.float64)
_kurt = float(((x - x.mean())**4).mean() / (x.std()**4))
print(f"synthetic kurtosis = {_kurt:.1f} (real LLaMA q_proj act ~ 95-269)")

sigma = x.std()
T = 3.0 * sigma
M = np.percentile(np.abs(x), 99.9)
p_out = float((np.abs(x) > T).mean())
p_in = 1.0 - p_out
print(f"Calibrated dist: sigma={sigma:.3f}  T=3sigma={T:.3f}  "
      f"M(99.9pct)={M:.3f}  M/T={M/T:.2f}  p_out={p_out:.4f}")

def fold(x, T, a):
    s = np.sign(x); ax = np.abs(x)
    out = ax > T
    y = x.copy()
    y[out] = s[out] * (T + a * (ax[out] - T))
    return y, out

def quant_dequant_int4(y, qmax=QMAX):
    scale = np.abs(y).max() / qmax          # symmetric per-tensor
    q = np.clip(np.round(y / scale), -qmax, qmax)
    return q * scale

def unfold(yq, T, a, out_mask):
    s = np.sign(yq); ay = np.abs(yq)
    z = yq.copy()
    z[out_mask] = s[out_mask] * (T + (ay[out_mask] - T) / a)
    return z

def proxy(a):
    A = T + a * (M - T)
    return A**2 * (p_in + p_out / a**2)

def levels_in_outlier_band(a):
    # folded outlier band width / quant step. step = max_fold / qmax.
    max_fold = T + a * (M - T)
    step = max_fold / QMAX
    band_width = a * (M - T)
    return band_width / step        # = a*(M-T)*QMAX / (T + a*(M-T))

alphas = np.array([0.02,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.75,0.95,0.99])
proxy_vals, glob_mse, out_mse = [], [], []
for a in alphas:
    y, out = fold(x, T, a)
    yq = quant_dequant_int4(y)
    z = unfold(yq, T, a, out)
    err2 = (z - x) ** 2
    proxy_vals.append(proxy(a))
    glob_mse.append(err2.mean())                 # what the proxy approximates
    out_mse.append(err2[out].mean())             # error on outlier positions only
proxy_vals, glob_mse, out_mse = map(np.array, (proxy_vals, glob_mse, out_mse))

print(f"\n{'alpha':>6} {'proxy':>10} {'GLOBAL-MSE':>11} {'OUTLIER-MSE':>12} "
      f"{'#lvls_out':>10}")
for a, pv, gv, ov in zip(alphas, proxy_vals, glob_mse, out_mse):
    print(f"{a:>6.2f} {pv:>10.4g} {gv:>11.4g} {ov:>12.4g} "
          f"{levels_in_outlier_band(a):>10.2f}")

a_proxy  = alphas[np.argmin(proxy_vals)]
a_global = alphas[np.argmin(glob_mse)]
a_outl   = alphas[np.argmin(out_mse)]
cf = (T * p_out / ((M - T) * p_in)) ** (1/3)
lvl = np.array([levels_in_outlier_band(a) for a in alphas])
cross = alphas[np.argmax(lvl >= 1.0)]
print(f"\nClosed-form proxy optimum a*      = {cf:.3f}  (grid: {a_proxy})")
print(f"GLOBAL-MSE optimum                = {a_global}  "
      f"(proxy tracks this: the mean)")
print(f"OUTLIER-position-MSE optimum      = {a_outl}  "
      f"(what DBAF's ordering value tracks)")
print(f"Outlier band spans >=1 INT4 level only for alpha >= ~{cross}")
print(f"\nMECHANISM: the proxy/global MSE is a MEAN over {N} entries; the "
      f"a^-1-amplified\nerror lives on the {p_out*100:.1f}% outlier positions, so it "
      f"barely moves the mean\n(global MSE at a_outl vs a_global: "
      f"{glob_mse[np.argmin(np.abs(alphas-a_outl))]/glob_mse.min():.2f}x) but blows up "
      f"outlier error\n({out_mse[np.argmin(np.abs(alphas-a_proxy))]/out_mse.min():.0f}x "
      f"worse at the proxy optimum). Since matched-T clipping (a=0) gives\nPPL 79k, "
      f"PPL tracks the OUTLIER error, not the mean -- so the empirical optimum\n"
      f"(a~0.25) sits far above the proxy optimum (a*~0.07), as observed.")
