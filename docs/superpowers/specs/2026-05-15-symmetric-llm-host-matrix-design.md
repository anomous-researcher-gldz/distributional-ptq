# Symmetric LLM Host Matrix for DBAF/PCSA Validation

**Date:** 2026-05-15  **Deadline:** EMNLP 2026 submission, 2026-05-25 AoE.

## Motivation

The current PCSA validation on LLMs uses **FlatQuant** (rotation-based, trained)
as the host. FlatQuant's learned rotation already absorbs per-prompt activation
variance, so PCSA has no signal to exploit — the cell saturates. That is not
"PCSA fails on LLMs"; it is "PCSA's mechanism was pre-empted by the host's
rotation." To honestly test whether the same mechanism that wins on SAM
transfers to LLMs, we need a **symmetric matrix** of LLM hosts spanning two
axes: `rotation` vs. `non-rotation`, and `training-free` vs. `trained`.

## Goal

Fill a 2×2 LLM host matrix, each cell ×4 conditions
(`alone`, `+DBAF`, `+PCSA-tf`, `+both`), and report W4A4 PPL on WikiText-2 and
C4. Then refresh the EMNLP §4 tables to expose the structure:

|                | Training-free                      | Trained    |
|----------------|------------------------------------|------------|
| Non-rotation   | RTN, GPTQ, AWQ, SmoothQuant        | TesseraQ   |
| Rotation       | QuaRot                             | FlatQuant  |

SAM is left as-is — there is no standard rotation-based SAM PTQ baseline.

## Current State (2026-05-15)

| Cell                                 | Status                                                                    |
|--------------------------------------|---------------------------------------------------------------------------|
| RTN / GPTQ / AWQ × alone             | done (G8)                                                                 |
| RTN / GPTQ / AWQ × +DBAF             | done (G8: −14 / −16 / −30% PPL)                                           |
| RTN / GPTQ / AWQ × +PCSA-tf / +both  | done but catastrophic (PPL 300K–1M) — **PCSA-tf scale-application bug**   |
| SmoothQuant × all 4                  | not done — `_collect_act_scales` hook crashes (hidden_states 2D vs 3D)    |
| QuaRot × all 4                       | not done — patch exists (`patches/quarot_dbaf_pcsa_patch.py`)             |
| TesseraQ × all 4                     | not done — env not created; patch exists (`patches/tesseraq_dbaf_pcsa_patch.py`) |
| FlatQuant × all 4                    | done; saturated (~6.97 PPL across all 4 columns)                          |

## Three Parallel Setup Tracks (~1 day, run concurrently)

### Track 1: SmoothQuant baseline fix

**File:** `FlatQuant/flatquant/baselines/smoothquant.py:65`

**Symptom:** `ValueError: not enough values to unpack (expected 3, got 2)` at
`bsz, q_len, _ = hidden_states.size()` inside Llama's attention forward.
Triggered when `_collect_act_scales` runs the calibration forward pass.

**Hypothesis:** the activation-scale hook (or the calibration tokenizer
wrapper) returns a 2D tensor where Llama expects 3D, likely via a
`squeeze()` or batch-collapsing reshape.

**Fix:** inspect the hook's tensor handling; preserve the `(bsz, q_len, hidden)`
shape. Add a unit test on a 4-token, 2-batch synthetic input.

**Exit criterion:** SmoothQuant alone runs end-to-end on Llama-3-8B and
produces a finite PPL (expected range: 8–20 based on published numbers).

### Track 2: PCSA-tf surgery

**File:** `FlatQuant/flatquant/baselines/pcsa_tf.py`

**Symptom:** all 6 PCSA-tf-augmented G8 cells produce PPL 297K–1.09M.

**Root cause:** scales are applied as `x * scale` immediately before
fake-quantization. Since fitted scales are small (~0.15), multiplying compresses
the input range; the quantizer's grid is sized for the original range, so it
loses resolution. The output remains scaled-down, propagating destructively.

**Fix:** apply scales **around** fake-quant, not before it. New per-Linear
forward:

```python
x_scaled = x * scale          # pre-condition the quantizer grid
y        = fake_quant(x_scaled)
y_rescaled = y / scale        # restore output magnitude
```

This makes the routing scale-invariant in magnitude — only the quantizer
grid alignment changes per anchor. Implement this in the per-Linear
forward hook used by `_apply_pcsa_tf_to_llm`.

**Unit tests (new file `tests/test_pcsa_tf_surgery.py`):**

1. `pcsa_tf(fp16_input)` ≈ `fp16_input` within float tolerance when the
   quantizer is identity (sanity: routing should not change magnitude).
2. On a single Llama Linear with RTN W4A4, `pcsa_tf` output PPL ≤
   `rtn_alone` PPL × 1.05 on a 256-token sample (sanity floor: routing
   must not regress).
3. With more anchors (k=2, 4, 8), output remains within ±5% of k=1 on the
   sanity sample (no destruction of dynamic range).

**Exit criterion:** all 3 unit tests pass + one full-model AWQ+PCSA-tf cell
produces PPL ≤ AWQ-alone × 1.05.

### Track 3: TesseraQ env + smoke

**Steps:**

1. `conda create -n tesseraq python=3.10 -y`
2. `pip install -r TesseraQ/requirements.txt`
3. Reproduce paper smoke: `python TesseraQ/main.py` with the Llama-3.1-8B
   W4A4 config, single layer, verify gradient block reconstruction converges
   within published bounds.
4. Full-model reproduction: confirm PPL within ±5% of the paper's reported
   25.73 (Llama-3.1-8B W4A4).

Note: TesseraQ paper reports Llama-3.1-8B. Our target is Llama-3-8B; expect
similar but not identical numbers.

**Exit criterion:** TesseraQ-alone on Llama-3-8B W4A4 produces PPL within
±5% of TesseraQ-alone on Llama-3.1-8B (their published value).

## Diagnostic Gate (after Track 2)

Run **one cell**: AWQ + PCSA-tf with the new surgery. Compare to AWQ-alone.

- **PASS** (PPL ≤ AWQ-alone × 1.05): proceed to all Phases A–D.
- **FAIL** (still catastrophic): pause TesseraQ +PCSA-tf and +both cells.
  Still run all `alone` and `+DBAF` cells across QuaRot and TesseraQ — those
  validate the DBAF×host-type claim independent of PCSA.

The gate decouples the DBAF story from PCSA-tf risk.

## Sequential Run Phases

### Phase A: Training-free LLM (4 hosts × 4 columns = 16 cells)

| Host          | alone | +DBAF | +PCSA-tf | +both |
|---------------|-------|-------|----------|-------|
| RTN           | done  | done  | rerun    | rerun |
| GPTQ          | done  | done  | rerun    | rerun |
| AWQ           | done  | done  | rerun    | rerun |
| SmoothQuant   | run   | run   | run      | run   |

**10 new cells** (6 already produce usable numbers — `alone` and `+DBAF` for
RTN/GPTQ/AWQ). ~3–4 GPU hours total.

Re-uses `scripts/run_training_free_full_table.py` after both tracks land.

### Phase B: QuaRot row (4 cells)

Wire `patches/quarot_dbaf_pcsa_patch.py` into a single-host runner. Run all 4
columns. ~2 GPU hours. Expected: QuaRot+DBAF cells should saturate (rotation
absorbs DBAF-style outlier reshaping); QuaRot+PCSA-tf cells should also
saturate. This is the diagnostic for PCSA's rotation-saturation claim.

### Phase C: TesseraQ row (4 cells, or 2 cells under gate FAIL)

Run TesseraQ alone, +DBAF (always); +PCSA-tf, +both (only if gate PASS).
~3–5 GPU hours per cell. ~12–20 hours total under PASS.

Expected: TesseraQ+DBAF lifts headline PPL (non-rotation host, dense-outlier
mitigation still has signal). TesseraQ+PCSA-tf lifts headline PPL under PASS
(this is the **strongest LLM PCSA validation cell**).

### Phase D: Aggregate + paper updates

1. Rewire `scripts/aggregate_results_to_latex.py` to emit a 6-row LaTeX table
   (4 training-free + 1 QuaRot + 1 TesseraQ + 1 FlatQuant). Reuse the
   existing G8 aggregator and add QuaRot + TesseraQ readers.
2. Update `paper/emnlp2026/sections/04-experiments.tex`:
   - Replace the current `tab:trainfree` with the 4-row training-free matrix.
   - Insert new subsection `\subsection{Rotation vs. Non-Rotation Hosts}`
     with the rotation row (QuaRot + FlatQuant) for both primitives.
   - Insert the TesseraQ row in the trained-non-rotation slot.
   - Reframe the FlatQuant saturation cell honestly: cite §X.Y's rotation
     argument as the explanation rather than presenting it as a PCSA failure.

## Out of Scope

- SAM symmetric expansion (no standard rotation SAM PTQ baseline exists).
- Qwen-2.5-7B (already noted as pending in task #69; can be appendix-only).
- OmniQuant — LET-GQA incompatibility on Llama-3 not worth debugging for one
  extra cell when TesseraQ covers the trained non-rotation slot.

## Total Budget

| Track / Phase | Wall clock        | Type            |
|---------------|-------------------|-----------------|
| Tracks 1–3    | 1 day             | dev (parallel)  |
| Phase A       | 3–4 GPU hours     | run             |
| Phase B       | 2 GPU hours       | run             |
| Phase C       | 12–20 GPU hours   | run             |
| Phase D       | 1–2 days          | dev + paper     |

**Worst case: 4 days.** Comfortable in the 10-day window.

## Risks and Mitigations

| Risk                                              | Mitigation                                                                 |
|---------------------------------------------------|----------------------------------------------------------------------------|
| PCSA-tf surgery still produces broken cells       | Diagnostic gate fails; DBAF half of TesseraQ/QuaRot still runs; story is "PCSA shines on SAM and on trained LLM hosts; training-free LLM remains an open problem" |
| TesseraQ env setup blocks on dependency conflicts | Track 3 runs in parallel; if it stalls past day 2, drop TesseraQ +PCSA-tf cells but keep alone+DBAF (~6h instead of ~20h)  |
| SmoothQuant fix uncovers a deeper transformers-version issue | Drop SmoothQuant from the table; report 3 training-free hosts (RTN/GPTQ/AWQ) |

## Success Criteria

1. At least one positive PCSA-tf cell on a non-rotation LLM host (proves
   mechanism transfers).
2. DBAF wins across all 4 quadrants of the matrix (proves DBAF is
   host-agnostic).
3. The FlatQuant saturation cell is **explained** in the paper rather than
   hidden — it becomes evidence for the rotation-pre-empts-PCSA hypothesis.
4. Total page count for §4 remains ≤8 (paper limit).
