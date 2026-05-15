# PCSA-tf surgery gate result (2026-05-15)

## Cells

| Cell | WT2 PPL |
|------|---------|
| AWQ alone | 13.49 |
| AWQ + PCSA-tf (post qmax surgery) | **356,937** |
| Ratio | 26,458× |

## Verdict: FAIL

## Diagnosis

The Task 4 qmax surgery was **necessary but not sufficient**.

The qmax fix (`2^bits - 1` → `2^(bits-1) - 1`) is mathematically correct and
verified by the unit tests — when the per-prompt anchor scale matches the
actual activation magnitude at the target Linear, the function preserves
magnitude within one quantization step.

The remaining issue is **architectural**, not algebraic:

1. `_collect_llm_pcsa_state` fits per-prompt scales on **layer-0 hidden-state
   activations only** (first decoder layer input).
2. `_apply_pcsa_tf_to_llm` installs the hook on all 192 hidden-size-matched
   Linears across all 32 decoder layers.
3. Activation magnitudes drift across the layer stack (post-LayerNorm
   normalization is per-layer; inputs to deep-layer Linears have different
   distributions than layer-0 inputs).
4. The calibration log shows scales of `[0.149, 0.149, 0.149, 0.226]` —
   tiny relative to typical deep-layer inputs. Once a runtime input exceeds
   the anchor scale, fake-quant clips it to ±0.15 — destroying information.

## Decision (per plan diagnostic gate logic)

**Pause PCSA-tf cells in Phases B and C.** Specifically:

- Phase A (training-free row): do NOT re-run RTN/GPTQ/AWQ/SmoothQuant ×
  {+PCSA-tf, +both} cells. They'll all show the same architectural failure.
- Phase B (QuaRot row): run `alone` + `+DBAF` only. Skip `+PCSA-tf`, `+both`.
- Phase C (TesseraQ row): run `alone` + `+DBAF` only. Skip `+PCSA-tf`, `+both`.

**The DBAF half of the matrix is unaffected.**

## Paper-level implication

The 4×4 LLM matrix collapses to a **DBAF-focused 4×2** matrix:

|  | Alone | +DBAF |
|---|---|---|
| Non-rotation, training-free (RTN/GPTQ/AWQ/SmoothQuant) | done | done |
| Rotation, training-free (QuaRot) | run | run |
| Non-rotation, trained (TesseraQ) | run | run |
| Rotation, trained (FlatQuant) | done (saturated) | done (saturated) |

PCSA validation rests on:
- SAM AHCPTQ ablation (ICML data — 13.4 → 18.2 mAP with synergy)
- S5 trained FlatQuant + KV-PCSA-v2 (saturated, ~6.97 PPL)

A proper training-free PCSA that handles per-layer magnitude drift would
require per-Linear scale fitting (a non-trivial design change, not a
constant tweak). Out of scope for this submission cycle; flagged as a
future-work item in the limitations.
