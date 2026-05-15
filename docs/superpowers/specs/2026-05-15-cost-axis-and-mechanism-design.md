# Cost-Axis Evidence + Mechanism Analysis Design

**Date:** 2026-05-15  **Deadline:** EMNLP 2026 submission, 2026-05-25 AoE.

## Motivation

Two reviewer-facing concerns remain after the LLM host-matrix plan
(`2026-05-15-symmetric-llm-host-matrix-design.md`) executes:

1. **Cost axis** — reviewers will ask "are your primitives actually cheap, or
   just simple?" We already have theoretical FLOPs and synthetic
   micro-benchmarks; we need end-to-end latency on **real** models plus a
   production INT4 GEMM comparison.

2. **Mechanism** — reviewers will ask "*why* does DBAF help?" The §4 currently
   shows DBAF wins across hosts but doesn't isolate the per-layer mechanism on
   the headline models. The B1 synthetic outlier study + B3 SwinIR per-layer
   ablation establish the pattern; B4 (per-layer ablation on SAM-B and
   Llama-3-8B) closes the mechanism story on the *real* headline models.

These are independent from the host-matrix plan but feed the same paper.

## Goal

Produce three new artifacts:

1. **End-to-end latency table** (Item 1): per-token latency on 9 real
   models — LLM × 3 (Llama-3-8B / Qwen-2.5-7B / Mistral-7B), SAM × 3
   (SAM-B/L/H), SR × 3 (SwinIR-light ×2/×3/×4) — measured under
   {none, +Hadamard, +learned-R, +DBAF, +PCSA-tf, +DBAF+PCSA-tf} via
   `forward_pre_hook` injection.

2. **INT4 GEMM latency benchmark** (Item 3): a single A100 sweep using
   torchao's `int4_dynamic_activation_int4_weight` kernel as the deployment
   baseline, with our primitives as pre-INT4-GEMM operations. This is the
   strong credibility tier — same kernel that powers vLLM/SGLang.

3. **Per-layer mechanism evidence** (B4): for SAM-B and Llama-3-8B,
   produce per-layer (outlier-fraction, DBAF-gain) scatter plots. Establishes
   that DBAF's PPL/mAP lift is correlated with per-layer outlier prevalence,
   not a global tuning artifact. Reuses `scripts/per_layer_ablation_swinir.py`
   pattern; we already have `scripts/per_layer_ablation_llama.py` +
   `scripts/per_layer_ablation_sam.py` scaffolds.

## Current State (2026-05-15)

| Artifact | Status |
|---|---|
| FLOP table (Item 2) | ✅ done — `scripts/_out/flop_table.json` + `figures/flop_table.tex` |
| Synthetic micro-benchmark (Item 2) | ✅ done — `scripts/_out/micro_benchmark*.json` |
| Long-context sweep (Item 4) | ✅ done — `scripts/_out/long_context/*.json` |
| **End-to-end latency (Item 1)** | scaffold at `scripts/end_to_end_latency.py`, NOT run |
| **INT4 GEMM (Item 3)** | scaffold at `scripts/int4_gemm_latency.py`, NOT run |
| Synthetic outlier study (B1) | ✅ done — `results/S4-dbaf-weak/synthetic/` |
| Per-layer SwinIR (B3) | ✅ done — `results/S4-dbaf-weak/per_layer/` |
| **Per-layer SAM-B (B4)** | script exists `scripts/per_layer_ablation_sam.py`, NOT run |
| **Per-layer Llama-3-8B (B4)** | script exists `scripts/per_layer_ablation_llama.py`, NOT run |

## Three Independent Workstreams

### Workstream 1: End-to-end latency (Item 1)

**Input:** 9 real-model checkpoints, all on disk.

**Method:** For each (model, primitive) pair, load the model, hook each Linear
with a `forward_pre_hook` that applies the primitive to the input, run K
forward passes with random inputs at the model's reference seq_len, record
wall-clock per token.

**Output:** `scripts/_out/e2e_latency/e2e_<model>.json` with all primitives
benchmarked.

**Headline table:** 9 rows × 6 primitives, presented as relative-to-fp16
multipliers in a single page of the paper.

### Workstream 2: INT4 GEMM benchmark (Item 3)

**Input:** Llama-3-8B (or smaller proxy for memory) at a fixed seq_len.

**Method:** Apply torchao's `int4_dynamic_activation_int4_weight` quant pass to
a Linear-stack; for each primitive, add a `forward_pre_hook` that fp16-multiplies
or rotates the input before the INT4 GEMM. Record kernel-level time.

**Output:** `scripts/_out/int4_gemm_latency.json` showing primitive overhead on
top of the deployment INT4 kernel.

**Headline:** "DBAF and PCSA-tf add ≤X% overhead on top of the production INT4
GEMM, vs. rotation primitives which require an additional dense fp16 kernel
launch."

### Workstream 3: Per-layer mechanism (B4)

**Inputs:** SAM-B + Llama-3-8B checkpoints + calibration data.

**Method:** For each Linear in the model, run two configurations:
- Baseline: W4A4 RTN
- +DBAF: W4A4 RTN with DBAF gated on for ONLY this Linear

Measure (a) the layer's outlier fraction in fp16, (b) the per-layer
output-MSE reduction with DBAF on. Plot a scatter (x: outlier fraction, y:
gain). Existing B3 produced this for SwinIR; B4 extends to SAM-B and Llama-3-8B.

**Output:** `results/S4-dbaf-weak/per_layer/{sam_b, llama3_8b}.json` plus a
combined scatter PDF `paper/emnlp2026/figures/per_layer_mechanism_full.pdf`
(extends the existing `per_layer_outlier_correlation.pdf` to all three
modalities).

## Paper Integration

Three §4 subsections to add or extend:

1. **§4.X Inference Cost Comparison** — new subsection with the e2e latency
   table + a short paragraph contrasting rotation cost (extra dense fp16
   matmul) with DBAF/PCSA-tf cost (in-place fp16 op).
2. **§4.Y Production INT4 Deployment** — new subsection with the INT4 GEMM
   benchmark + "deployment-ready" framing.
3. **§4.Z Mechanism** (extend existing §4.9) — add SAM-B + Llama-3-8B panels
   to the existing per-layer correlation figure.

## Total Budget

| Workstream | Wall clock | Type |
|---|---|---|
| WS1 (e2e latency × 9 models × 6 primitives) | ~6-8 GPU hours | run |
| WS2 (INT4 GEMM benchmark) | ~1 GPU hour | run |
| WS3 (per-layer SAM-B + Llama-3-8B) | ~4 GPU hours | run |
| Paper integration (§4 subsections + figure) | 1 day | dev |

**Total: ~12 GPU hours + 1 day of paper work.** Fits in 10-day window
alongside the host-matrix plan; they share no calibration GPU time.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| INT4 GEMM kernel doesn't support all our primitive hooks | Fallback: report kernel-only vs kernel+primitive deltas separately; if a hook fails, mark cell `--` and explain. |
| Llama-3-8B per-layer ablation exceeds 4h | Reduce the number of layers sampled (e.g., one per group of 4); paper figure only needs ~32 points to make the correlation visible. |
| SAM-H latency at full image resolution OOMs | Drop SAM-H from WS1 if needed; SAM-B + SAM-L already cover the size scaling. |

## Success Criteria

1. End-to-end latency table populates all 9 × 6 cells.
2. INT4 GEMM benchmark shows DBAF + PCSA-tf overhead < rotation overhead.
3. Per-layer correlation extends to SAM-B and Llama-3-8B with the same
   monotonic outlier-vs-gain pattern visible in SwinIR.
4. §4 stays ≤8 pages.

## Out of Scope

- Mistral / Llama-3-70B (cost prohibitive; Llama-3-8B + Qwen-2.5-7B cover the
  LLM size range adequately).
- SAM2 (no SAM2 quantization baseline exists in the literature).
- Multi-GPU latency (single A100 is the reference platform).
