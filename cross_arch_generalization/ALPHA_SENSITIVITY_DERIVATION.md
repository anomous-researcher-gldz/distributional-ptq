# Sensitivity-weighted correction to the DBAF fold strength α\*

Author-response note (Submission 11057), addressing the reviewer request for "a
sensitivity-weighted derivation that accounts for unfold noise amplification."

## Setup

DBAF folds outliers (|x| > T) to `sgn·[T + α(|x|−T)]`, quantizes in folded space
with step `s = M_fold/q_max`, `M_fold = T + α(M−T)`, then unfolds outlier positions
by `1/α`. The paper's per-tensor MSE proxy (Appendix, Eq. of α\*) is

```
L̃(α) = A² · ( p_in + p_out / α² ),   A = T + αΔ,   Δ = M − T
```

where the bulk term carries `p_in` and the outlier term carries `p_out/α²` (the
`1/α²` is the unfold noise amplification). Minimizing gives the paper's

```
α* = ( T·p_out / ((M−T)·p_in) )^{1/3}.
```

## Sensitivity weighting

Reconstruction MSE is not the end task. Weighting the outlier-position term by its
end-task sensitivity λ (the outlier-vs-bulk Hessian ratio h_out/h_in that GPTQ/OBQ
use) gives

```
L(α) = A² · ( p_in + λ · p_out / α² ).
```

Differentiate and set to zero:

```
dL/dα = 2AΔ·p_in + 2λ p_out · A(Δα − A)/α³ = 0
      ⇒ Δ p_in + λ p_out (Δα − A)/α³ = 0            (÷ 2A)
      ⇒ Δ p_in α³ + λ p_out Δα − λ p_out T − λ p_out αΔ = 0   (× α³, A = T + αΔ)
```

The `λ p_out Δα` terms cancel, leaving `Δ p_in α³ = λ p_out T`, i.e.

```
α³ = λ · T·p_out / ((M−T)·p_in) = λ · (α*)³
```

**⇒  α*_sens = λ^{1/3} · α\***

## What this gives (and what it does not)

- **λ = 1 recovers the paper exactly** (α*_sens = α\*) — the derivation is consistent
  with Eq. (α-star).
- **λ ≥ 1 ⇒ α*_sens ≥ α\***: because PPL is far more sensitive to outlier-position
  error than to bulk error, the true optimum sits **above** α\*. This *derives* the
  paper's "α\* is a lower bound" statement rather than asserting it.

**Honest scope — λ is not derived to a value.** λ is an end-task quantity (a Hessian
ratio), not a fold-geometry quantity, so it cannot be obtained from T, M, p_out
alone. We tested the natural first-principles proxy, the energy ratio
`λ ≈ E[x²|outlier]/E[x²|bulk]`, on real LLaMA-3-8B activations
(`alpha_sensitivity_check.py`): it measures **λ ≈ 28**, which through
α*_sens = λ^{1/3}·α\* predicts α ≈ **0.65**, not the operating 0.25. (We also could
not reproduce the paper's α\*≈0.07 from calibration 99.9-percentile stats; we get
α\* ≈ 0.2, consistent with the paper's α\* using M = max magnitude.) So:

- The **form** α*_sens = λ^{1/3}·α\* and the **λ≥1 ⇒ lower-bound** result are exact.
- A **first-principles numerical prediction** of the operating α is **not** claimed;
  λ is treated as a measured/empirical sensitivity weight. The author response
  states only the form and the lower-bound result, not a predicted α.
