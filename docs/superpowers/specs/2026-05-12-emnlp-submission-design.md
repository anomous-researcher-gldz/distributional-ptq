# EMNLP 2026 Submission: ICML Rejection → EMNLP Resubmission

**Deadline:** May 25 2026 AoE (13 days from 2026-05-12)
**Venue:** EMNLP 2026 main, ACL template
**Compute:** Local A100 80GB + 1 rented A100 80GB (SSH-orchestrated)

## Goal

Take the rejected ICML 2026 paper "Towards a Unified Distribution-Centric Post-Training Quantization" and resubmit to EMNLP 2026, addressing all three reviewers' core critiques with:

1. New experiments that strengthen the **LLM** story (the weakest part of ICML)
2. **Real INT4 deployment** across all three architectures: FlatQuant's own kernels (LLM) + torchao (SAM, SR; LLM in supplementary). Target W4A4 everywhere; fall back to W4A16 if W4A4 fails.
3. A **modernized codebase** so all three sub-projects run on a current stack
4. A **full paper revision** that addresses every presentation issue, ports to ACL template, and incorporates rebuttal-only material
5. **Preserve the distribution-centric framing** — unified across architectures stays the core claim; LLM is *prominent* but never the lead at the expense of unification

## Background

### ICML 2026 outcome — Reject

Three reviewers (VtXm, aCWD, r3g3) converged on:

1. **DBAF novelty** vs companding/clipping (aCWD, VtXm)
2. **α\* theory–practice gap** — closed-form α\* almost never used; grid wins (VtXm)
3. **PCSA "unified" framing overclaim** — PCSA is architecture-specific (aCWD)
4. **LLM gains marginal** — 0.01 PPL, no error bars, α=0.99 ≈ DBAF off (aCWD)
5. **Presentation** — "ICML 2025" header, undefined M, AHPTQ typo, unclear q-projections, "element" ambiguity (aCWD)
6. **Coverage gaps** — no SAM-H, no H-DETR (added in rebuttal), no real INT4 latency (added in rebuttal)

### Rebuttal material that needs to land in the EMNLP paper

- Real INT4 hardware latency (DBAF ~0.1–1.7% overhead under real INT4 GEMM)
- H-DETR result for SAM-B (20.6 mAP W4A4)
- α=1.0 in grid (PPL: PCSA-only 6.97, DBAF-only 6.97, both 6.96)
- LLaMA-3-8B calibration sensitivity (WikiText-2 vs C4 stability)
- α\*/α_grid alignment (α\* < α_grid < 1.0 in 100% of classified tensors)
- Cross-detector PCSA transfer (YOLOX→H-DETR, <5% relative degradation)

### Concrete issues found from reading the current paper

**Figure-level:**
- `dist_figs_icml.pdf` caption has **duplicate "(D)"** labels (should be C and D for SAM-L and SAM-H)
- `Figure_1.png` is mis-named — it's actually the ablation bar chart, not paper Figure 1
- Two redundant ablation figures: `ablation_icml.png`, `figure.pdf`, plus `alphas_ablation.png` — pick one canonical version
- `pcsa_figs.pdf` panel D bar chart shows 13.4 → 13.1 → 13.8 → 13.6 across anchor counts — the gap is tiny and visually weakens the PCSA contribution. Either reframe (PCSA = small but consistent), or add error bars to make significance visible.
- `canvas-image-1-...png` (Figure 1) is cluttered; the unified-framework story would be clearer with a single-panel flowchart: tensor → taxonomy → DBAF/PCSA branch → quantized output (this is add-on #6)
- `qualitative.pdf` is buried in appendix — it's actually a strong figure showing AHCPTQ's noisy masks vs. ours; move to main paper or at least cross-ref prominently
- All figures need axis labels and self-contained captions; current α-ablation figure spans 10+ orders of magnitude in log loss which makes the trends hard to read

**Text-level:**
- Three duplicate Related Works sections commented out in `.tex` (lines 183–211, 197–211) — clean up before submission
- Multiple commented-out Methodology drafts (lines 282–301) — clean up
- Line 433: "AHPTQ" (typo for AHCPTQ) in body text
- Line 567: `M` first appears in the MSE proxy (Eq. 2) — never defined inline; reviewer aCWD specifically flagged this
- Line 569: "element" never disambiguated between weight/activation; reviewer aCWD flagged
- Line 600: PCSA only says "post-decoder-q-layer" — should explicitly specify SAM mask decoder cross-attention vs LLM decoder self-attention
- Algorithm 1 line "Update s_{k*} from assigned activation statistics" — never specifies *how* (min/max + EMA per rebuttal)
- INT4 format unspecified anywhere — say "symmetric uniform INT4 (qmax=7)" at first use
- Table 4 (PCSA anchors vs PPL for LLaMA-2 7B): values 5.81/5.80/5.81/5.81 are within noise — needs error bars or honest framing
- No standard deviations / error bars anywhere in result tables
- Model Complexity section (line 867) still reports simulated FP32 2× slowdown — must be replaced with real INT4 numbers from rebuttal (~0.1–1.7% overhead)
- Implementation Details section is too short — needs calibration size, hyperparams, hardware specs, seeds, training time
- SwinIR table shows ×2/×3 but ×4 was discussed in α-ablation figure — keep consistent across paper

## Scope

### In scope

**Environments:**
- Old AHCPTQ env: Py 3.9 + torch 1.13 + mmcv-full 1.7 + vendored mmdet 2.x
- Modern `unifyptq` env: Py 3.10 + torch 2.6 + torchao (FlatQuant + CompSRT already work)

**Experiments (all using W4A4 target with W4A16 fallback for torchao):**
- **A**: DBAF on GPTQ/RTN/AWQ baselines, LLaMA-3-8B (must) + Qwen-2.5-7B (must) + Mistral-7B or Phi-3 (stretch, only if extra GPU + add-on #10 selected)
- **C**: KV-cache PCSA design 1 (per-prompt routing), LLaMA-3-8B on RULER 4k/8k/16k/32k
- **D**: Real INT4 deployment — LLM via FlatQuant's `deploy/` kernels (main paper) + torchao W4A4/W4A16 (supplementary); SAM via torchao; SR via torchao
- **CompSRT-A**: DBAF on weak SR PTQ baseline (RTN on SwinIR or EDSR), W4A4
- **CompSRT-B**: torchao real INT4 on SwinIR with PSNR/SSIM
- **Ablations**: α=1.0 grid row; 3 seeds on headline LLaMA-3-8B numbers
- **AHCPTQ paper-faithful**: rerun SAM-B and SAM-L W4A4 in old env to confirm; SAM-H if compute allows
- **Add-on #7**: MMLU 5-shot + GSM8K on LLaMA-3-8B (and Qwen if time)
- **Add-on #8**: Per-layer learned α via gradient (lightweight, risky bet)
- **Add-on #10**: Mistral-7B or Phi-3 as third LLM family (stretch)

**Paper revision (parallel from day 1):**
- ACL template port
- All 16 reviewer issues addressed (full table below)
- All rebuttal numbers folded into main paper
- All figure/text issues from "Concrete issues" section above addressed
- Add-ons #1 DBAF>clipping theoretical proof, #2 Limitations section, #3 "composable primitive" framing, #4 statistical significance tests, #5 EMNLP-tuned abstract (still distribution-centric, LLM more prominent), #6 new conceptual figure, #9 HuggingFace code+model release

### Out of scope

- SAM2
- LLaMA-3-70B (no budget for 70B compute)
- HDETR/DINO/FocalNet new training (reuse rebuttal H-DETR numbers)
- mmcv 2.x / mmdet 3.x port of AHCPTQ (old env handles it)
- New CompSRT base experiments beyond -A and -B
- W2/W3 sub-4-bit aggressive settings
- Live demo / Gradio app

## Compute plan: 2× A100 80GB

**Local A100** — primary, runs FlatQuant LLM experiments, KV-PCSA, paper rendering, orchestration.

**Rented A100** (Lambda Labs / RunPod / Vast.ai ~$300–400 for 13 days) — runs AHCPTQ old env SAM experiments, CompSRT experiments, MMLU/GSM8K evals. Set up via SSH:

1. User rents GPU, shares host + port + SSH key
2. Claude orchestrates via Bash:
   - `ssh-add <key>`
   - SSH in to install miniconda (one-time, ~10 min)
   - Clone unifying-ptq repo on remote
   - Set up `unifyptq` and `ahcptq-old` envs on remote (re-run install scripts)
   - Configure rsync between machines: `/home/ubuntu/unifying-ptq/results/` ↔ `remote:/work/results/`
   - Launch experiments via `ssh remote 'tmux new-session -d -s expN "cd /work && python ..."'`
   - Pull results periodically with rsync

**Workload split:**

| Machine | Primary jobs |
|---|---|
| Local A100 | LLaMA-3-8B calibration, KV-PCSA implementation + RULER, real INT4 measurement, paper rendering |
| Remote A100 | AHCPTQ old env smoke + SAM-B/L runs, Qwen-2.5-7B baseline, CompSRT-A and -B, MMLU/GSM8K, optional Mistral |

## Checkpointing strategy

Every experiment saves:

1. **Calibrated quantization state** → `results/<exp>/<model>/<method>/<seed>/state.pt`
   - For FlatQuant: rotation matrices, learned diagonals, LWC/LAC params
   - For DBAF: per-layer α values, T thresholds
   - For PCSA: anchor banks + per-anchor scales
   - For torchao: packed INT4 weights
2. **Evaluation results** → `results/<exp>/<model>/<method>/<seed>/eval.json` with structured fields `{model, bits, method, task, metric, value, seed, timestamp}`
3. **Real INT4 deployment models** → `results/deploy/<model>/<format>/`
4. **Pre-quantized HuggingFace models** (camera-ready deliverable) → push to `unifying-ptq/<modelname>-W4A4` on HF Hub

Sync between machines nightly: `rsync -avz --partial local:results/ remote:results/` and vice versa. Total expected disk: ~40 GB across both machines.

## Sub-projects

```
S1. Environment setup
    S1a. Old AHCPTQ env (Py 3.9 + torch 1.13 + mmcv-full 1.7 + vendored mmdet 2.x) — LOCAL + REMOTE
    S1b. Install torchao in unifyptq env — LOCAL + REMOTE
    S1c. Set up rsync sync between local and remote
    S1d. Set up checkpoint directory layout

S2. AHCPTQ paper-faithful reproduction (old env)
    Smoke: test_quant.py on COCO val2017 with SAM-B + YOLOX
    Re-run SAM-B, SAM-L W4A4 to confirm published numbers
    SAM-H optional (memory permitting on 80GB)

S3. torchao integration
    S3a. torchao → FlatQuant W4A4 (W4A16 fallback) — for LLM supplementary
    S3b. torchao → AHCPTQ-calibrated SAM image encoder W4A4 (W4A16 fallback)
    S3c. torchao → CompSRT W4A4 (W4A16 fallback)

S4. Experiment A: DBAF on weak baselines
    Hook DBAF onto GPTQ/RTN/AWQ paths in FlatQuant repo
    LLaMA-3-8B (must), Qwen-2.5-7B (must), Mistral-7B (stretch)

S5. Experiment C: KV-cache PCSA design 1
    Per-prompt descriptor computed once at prompt time
    Routes to a single K/V scale for whole generation
    Eval: RULER 4k → 8k → 16k → 32k

S6. Experiment D: real INT4 measurement
    LLaMA-3-8B via FlatQuant deploy/ kernels (main, W4A4)
    LLaMA-3-8B + Qwen via torchao W4A4 (W4A16 fallback) (supplementary)
    SAM via torchao
    SR via torchao
    Metrics: tokens/sec, peak GPU memory, accuracy preservation

S7. Ablations
    α=1.0 in grid for LLaMA-3-8B
    3 seeds on headline LLaMA-3-8B W4A4 numbers
    Statistical significance tests (Wilcoxon paired test, paired bootstrap)
    Add-on #8 risky bet: per-layer learned α via gradient

S8. CompSRT push
    S8a. DBAF on weak SR baseline (RTN on SwinIR/EDSR)
    S8b. torchao real INT4 on SwinIR + PSNR/SSIM

S9. MMLU + GSM8K downstream evals (Add-on #7)
    Run on LLaMA-3-8B + Qwen-2.5-7B
    Compare FP, FlatQuant baseline, FlatQuant + DBAF/PCSA

S10. Paper revision (parallel from day 1)
    S10a. Clone ACL template + scaffold paper
    S10b. Presentation fixes (header, AHPTQ typo, undefined M, "element" ambiguity, q-projection specifics, Alg.1 detail, INT4 format)
    S10c. Clean up duplicate commented-out Related Works + Methodology drafts
    S10d. Add-on #1: half-page DBAF > matched-T clipping theoretical proof in appendix
    S10e. Add-on #2: Limitations section (DBAF helps less on rotation-based baselines, PCSA arch-specific)
    S10f. Add-on #3: reframe DBAF as composable primitive in intro
    S10g. Add-on #4: significance tests on headline numbers
    S10h. Add-on #5: EMNLP-tuned abstract — distribution-centric remains the lead, LLM more prominent in evidence section
    S10i. Add-on #6: replace Figure 1 with single-panel flowchart (taxonomy → DBAF/PCSA → output)
    S10j. Add α*/α_grid alignment scatter plot + table (from rebuttal)
    S10k. Add real INT4 latency table (LLM via FlatQuant kernels + torchao)
    S10l. Add H-DETR row to SAM table
    S10m. Reframe LLM section: weak-baseline gains as the headline; FlatQuant+DBAF as "composes cleanly"
    S10n. Add experiment C (KV-cache PCSA) section + RULER table
    S10o. Add experiment D (real INT4) section + multi-architecture deployment table
    S10p. Add CompSRT-A and CompSRT-B sections
    S10q. Reframe "unified" claim per aCWD (distribution-centric framework, PCSA = general mechanism at architecture-specific sites)
    S10r. Fix dist_figs caption duplicate (D); rename Figure_1.png; consolidate ablation figures
    S10s. Move qualitative.pdf to main paper or boost cross-references
    S10t. Update Model Complexity section with real INT4 numbers (not simulated FP32)
    S10u. Add Implementation Details (calibration size, hardware, hyperparams, seeds)
    S10v. Add-on #9: code + pre-quantized models on HuggingFace
    S10w. Final polish + reference cleanup + ACL format check + page-limit pass
```

## Day-by-day timeline (Approach 2 — parallel)

| Day | Local A100 | Remote A100 | Paper |
|---|---|---|---|
| 1 | S1a old env; S1b torchao install; verify FlatQuant + CompSRT | S1a/b mirror; S1c rsync setup; S1d checkpoint layout | S10a ACL template scaffold |
| 2 | S3a torchao→FlatQuant hooks; smoke test torchao on FlatQuant LLaMA | S2 AHCPTQ smoke on COCO | S10b presentation fixes; S10c commented-out cleanup |
| 3 | S4 DBAF-on-weak-baselines LLaMA-3-8B (GPTQ, then RTN) | S2 SAM-B AHCPTQ reproduction | S10d DBAF>clipping proof; S10f composable primitive framing |
| 4 | S4 LLaMA-3-8B AWQ + start Qwen-2.5-7B | S2 SAM-L AHCPTQ reproduction; S4 Qwen baseline | S10e Limitations section; S10j α*/α_grid section |
| 5 | S5 KV-PCSA design 1 implementation | S4 Qwen weak baselines | S10k real INT4 latency table |
| 6 | S5 KV-PCSA + RULER setup, RULER 4k | S8a CompSRT-A start (DBAF on weak SR baseline) | S10l H-DETR row; S10q "unified" reframe |
| 7 | S5 RULER 8k; S6 LLM real INT4 (FlatQuant kernels) start | S9 MMLU+GSM8K on Qwen | S10m LLM section reframe |
| 8 | S5 RULER 16k; S6 torchao LLM W4A4 | S3b torchao SAM; S9 MMLU+GSM8K LLaMA | S10n KV-PCSA section + tables; S10i new conceptual figure |
| 9 | S5 RULER 32k; S6 wrap LLM real INT4 | S3c torchao CompSRT; S8b torchao SwinIR | S10o real INT4 section; S10p CompSRT sections |
| 10 | S7 α=1.0 + 3 seeds LLaMA-3-8B; #8 learned α (risky) | (optional) Mistral-7B; HF model uploads | S10g significance tests; S10v HF release prep |
| 11 | Compute slack; sanity check; figure final renders | rsync results; final remote sweeps | S10r-s figure consolidation; S10h abstract finalize |
| 12 | Final figure generation; rebuild tables | (optional final runs) | S10t-u Model Complexity + Implementation Details; full revision pass |
| 13 | Buffer for failed experiments | rsync, archive | S10w final polish, references, ACL format check, **submit** |

## Experiment matrix

| Exp | Method | Models | Bits | Eval | Where | Provenance |
|---|---|---|---|---|---|---|
| Baseline | FlatQuant only | LLaMA-3-8B, Qwen-2.5-7B | W4A4 | WikiText-2 PPL + 6 commonsense | Local | Reuse where possible |
| **A** | GPTQ/RTN/AWQ + DBAF | LLaMA-3-8B, Qwen-2.5-7B, Mistral-7B (stretch) | W4A16 | WikiText-2 PPL | Local | New |
| **C** | FlatQuant + KV-PCSA | LLaMA-3-8B | W4A4 KV4 | RULER 4k/8k/16k/32k | Local | New |
| **D-LLM-FQ** | FlatQuant + DBAF + own kernels | LLaMA-3-8B | W4A4 | tokens/sec, peak mem, PPL | Local | New + rebuttal verify |
| **D-LLM-AO** | FlatQuant + DBAF + torchao | LLaMA-3-8B (+ Qwen) | W4A4 → W4A16 | tokens/sec, peak mem, PPL | Local | New, supplementary |
| **D-SAM** | AHCPTQ + DBAF/PCSA + torchao | SAM-B | W4A4 → W4A16 | COCO segm mAP, tokens/sec | Remote | New |
| **D-SR** | CompSRT + DBAF + torchao | SwinIR | W4A4 → W4A16 | PSNR/SSIM, tokens/sec | Remote | New |
| α=1.0 | FlatQuant grid w/ α=1.0 | LLaMA-3-8B | W4A4 | WikiText-2 PPL | Local | Rebuttal value 6.97 |
| Seeds | FlatQuant + DBAF/PCSA × 3 seeds | LLaMA-3-8B | W4A4 | error bars | Local | New |
| **#7** | FlatQuant + DBAF/PCSA | LLaMA-3-8B, Qwen-2.5-7B | W4A4 | MMLU 5-shot + GSM8K | Remote | New |
| **#8** | Per-layer learned α (gradient) | LLaMA-3-8B | W4A4 | WikiText-2 PPL | Local | New, risky bet |
| AHCPTQ-faithful | AHCPTQ | SAM-B, SAM-L (SAM-H optional) | W4A4 | COCO segm mAP | Remote | Reproduce |
| **CompSRT-A** | DBAF on weak SR baseline | SwinIR or EDSR | W4A4 | PSNR/SSIM | Remote | New |
| **CompSRT-B** | torchao on SwinIR | SwinIR | W4A4 → W4A16 | tokens/sec, PSNR/SSIM | Remote | New |

## Reviewer issue → revision target

| Issue | From | Fix |
|---|---|---|
| ICML 2025 header | aCWD | ACL template auto-fixes |
| AHPTQ→AHCPTQ typo (line 433) | aCWD | global find/replace |
| Variable M undefined in Eq. 2 | aCWD | inline definition "M = robust 0.999 percentile of \|x\|" |
| "element" ambiguity in §3.2 | aCWD | "elementwise across flattened tensor; weights and activations both" |
| Which q-projections | aCWD | "SAM mask decoder cross-attn; LLM decoder self-attn" |
| Algorithm 1 scale update | aCWD | "min/max statistics of activations assigned to each cluster, EMA update with momentum 0.9" |
| INT4 format unspecified | aCWD | "symmetric uniform INT4 (qmax=7)" at first use |
| α=1.0 not in grid | aCWD | add row to ablation table |
| DBAF vs clipping (matched-T) | aCWD | add explicit matched-threshold-clipping baseline row + theoretical proof in appendix (add-on #1) |
| α*/α_grid alignment | VtXm | scatter plot per layer + summary table |
| Prompt distribution shift | VtXm | cross-detector transfer table (rebuttal) |
| H-DETR results | VtXm | add to SAM table |
| Real INT4 latency | r3g3, VtXm | dedicated table with FlatQuant kernels + torchao numbers |
| Marginal LLM gains | aCWD | weak-baseline experiment A + KV-PCSA experiment C + MMLU/GSM8K (#7) |
| "Unified" framing overclaim | aCWD | reframe distribution-centric, PCSA = "general mechanism, architecture-specific application sites" — **but keep cross-architecture as central claim** |
| DBAF novelty vs companding | aCWD, VtXm | sharpen related work + theoretical proof DBAF > matched-T clipping |
| No error bars / noise vs signal | aCWD | 3-seed runs + Wilcoxon paired test + paired bootstrap |
| dist_figs duplicate (D) caption | (found) | fix caption to C and D for SAM-L and SAM-H |
| Figure_1.png misleading filename | (found) | rename to ablation_dbaf_pcsa.png |
| Redundant ablation figures | (found) | pick canonical version, delete others |
| Qualitative buried in appendix | (found) | move to main paper or boost cross-reference |
| Model Complexity reports simulated FP32 2× | (found) | replace with real INT4 numbers from rebuttal |
| Implementation Details too short | aCWD (implicit) | expand: calibration size, hardware, hyperparams, seeds, training time |
| Three duplicate commented-out sections | (found) | delete |

## Add-ons summary (all in scope)

| # | Add-on | Days | Type | Outcome |
|---|---|---|---|---|
| 1 | DBAF > matched-T clipping theoretical proof | 1 | Writing | Half-page appendix proof |
| 2 | Limitations section | 0.5 | Writing | Explicit, honest about rotation-based baselines |
| 3 | "Composable primitive" framing | 0 | Writing | Intro sentence change |
| 4 | Statistical significance tests | 0.5 | Code+Writing | Wilcoxon + paired bootstrap p-values in headline tables |
| 5 | EMNLP-tuned abstract | 0.5 | Writing | Lead with distribution-centric; LLM evidence prominent but not at expense of unified claim |
| 6 | Conceptual figure | 1 | Figure | Single-panel flowchart: taxonomy → DBAF/PCSA → quantized output |
| 7 | MMLU + GSM8K | 1-2 | Compute | Harder downstream tasks |
| 8 | Per-layer learned α | 1.5 | Code | Risky bet; closes α*/α_grid gap if it works |
| 9 | HuggingFace code + model release | 0.5 | Engineering | Reproducibility deliverable |
| 10 | Mistral-7B or Phi-3 | 2-3 | Compute | Third LLM family (uses remote GPU) |

## Risks + cuts

| Risk | Likelihood | Cut plan |
|---|---|---|
| mmcv-full 1.7 doesn't build cleanly | medium | Docker container with pre-built wheels; or use saved checkpoints from ICML run |
| KV-PCSA design 1 bugs | high | Simplify to per-prompt scalar scale modulation (no clustering) |
| RULER 32k OOMs | medium | Cap at 16k |
| Qwen-2.5-7B baseline reproduction fails | low-medium | Drop Qwen, LLaMA-3-8B only |
| Mistral/Phi-3 (#10) fails | medium | Drop (it's a stretch); paper still works |
| torchao W4A4 doesn't compile on a codebase | medium-high | Fall back to W4A16 with caveat in paper |
| Weak-baseline DBAF (Exp A) doesn't show gain | medium-high | Pivot framing to "DBAF is composable primitive — minor over strong baselines, larger over weak ones, with theoretical guarantee" |
| FlatQuant deploy/ kernels need rebuilding for cu124 | low-medium | Rebuild from source; or use rebuttal numbers as-is |
| Per-layer learned α (#8) doesn't outperform grid | medium-high | Cut #8 silently; not in critical path |
| Remote GPU rental SSH issues | low | Switch providers; or run sequentially on local |
| Paper revision lags experiments | medium | Days 11–13 are buffered; cut Mistral, RULER 32k, Qwen if needed |

## Definition of done

- [ ] Paper compiles in ACL template within page limit (8 pages main + unlimited refs/appendix)
- [ ] All 25 reviewer + found-issue revision targets addressed
- [ ] α=1.0 row in ablation table
- [ ] 3-seed error bars + significance tests on LLaMA-3-8B W4A4 headline numbers
- [ ] Experiment A table for LLaMA-3-8B + Qwen-2.5-7B (Mistral if time)
- [ ] Experiment C RULER results for ≥3 context lengths
- [ ] Experiment D real INT4 measurements for LLM (FlatQuant kernels + torchao), SAM, SR
- [ ] AHCPTQ paper-faithful SAM-B and SAM-L reproduced; torchao-on-AHCPTQ row in real INT4 table
- [ ] CompSRT-A and CompSRT-B results in paper
- [ ] All rebuttal numbers in main paper
- [ ] Add-ons #1, #2, #3, #4, #5, #6, #7, #9 incorporated; #8, #10 if time
- [ ] HuggingFace pre-quantized models uploaded
- [ ] Internal proofread complete
- [ ] Submitted to EMNLP 2026 by May 25 AoE

## Implementation handoff

After this spec is approved, the next step is to invoke the `writing-plans` skill to produce a detailed implementation plan covering each sub-project (S1–S10). The implementation plan will turn the timeline into concrete tasks with file paths, function signatures, and acceptance checks.
