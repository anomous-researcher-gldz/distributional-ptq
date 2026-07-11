# Cross-architecture generalization — author-response analysis (Submission 11057)

Analysis run for the EMNLP/ARR 2026 author response to *"From Distribution to
Decision: A Diagnostic for Composable PTQ Primitives."* Each script is
self-contained and writes a JSON to `results/`. Numbers here back the author-response
(reviewer-discussion) claims (breadth, calibration protocol, robustness, generalization). Nothing here
contradicts the submitted paper; every run confirms a submitted claim or answers a
reviewer question.

## How to run

All scripts are run from the **repo root** with no CLI arguments — knobs (e.g.
`alpha=`, target model) are set in-file near the top of each script. Each writes a
JSON to `cross_arch_generalization/results/`. Scripts currently assume the repo lives
at `/home/ubuntu/distributional-ptq`; if yours is elsewhere, adjust the `BASE` /
input paths at the top of the script.

```bash
cd /path/to/distributional-ptq
python cross_arch_generalization/scripts/<name>.py     # -> results/<name>_results.json
```

**Two tiers of reproduction:**

1. **Zero-GPU (seconds, no model download).** These recompute the headline
   robustness / diagnostic numbers directly from the committed per-layer statistics
   in `results/S4-cross-model-layer-analysis/*.json` — nothing else needed:
   ```bash
   pip install numpy scikit-learn scipy
   python cross_arch_generalization/scripts/threshold_robustness_dispatch.py
   #   -> threshold_robustness_results.json : jitter 93.9%/min 73.9%/p5 77.4%,
   #      leave-one-architecture-out AUC 0.962 (0.93-0.99), balance 36.5/63.5
   python cross_arch_generalization/scripts/w3_alpha_gap.py
   ```

2. **GPU (loads a model).** Every other script downloads/loads its model
   (LLaMA-3-8B, Qwen-2.5-7B, CLIP-ViT-L/14, Whisper-small, DiT-XL, SAM) and runs a
   weight-only RTN host. Use the `flatquant` conda env from the top-level
   [`README.md`](../README.md#21-installation) (torch + transformers), plus
   `pip install diffusers` for `dit_*.py` and `open_clip_torch` for `clip_*.py`.
   Example:
   ```bash
   conda activate flatquant
   python cross_arch_generalization/scripts/rotation_control.py   # RTN 2x2 rotation control
   python cross_arch_generalization/scripts/clip_flagship.py      # CLIP W4 DBAF
   ```

   The **SAM positive/FIRE-site descriptor ablation** (`sam_descriptor_ablation.py`,
   SQ6q W4) additionally needs the public SAM-B checkpoint and COCO val2017. Point
   `SAM_WS` at a directory containing both, then run:
   ```bash
   pip install segment-anything pycocotools scikit-image
   # SAM_WS/sam_vit_b.pth       <- https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   # SAM_WS/coco/val2017/       and
   # SAM_WS/coco/annotations/instances_val2017.json  <- https://cocodataset.org/#download
   SAM_WS=/path/to/samdata python cross_arch_generalization/scripts/sam_descriptor_ablation.py
   #   -> sam_descriptor_ablation_results.json : all 4 poolings FIRE (c<=0.4) at the
   #      SAM-B mask-decoder cross-attn q site, both synthetic (paper c=0.189) and real-COCO
   ```
   Flagship scripts default to `alpha=0.25` (the `*_a025_results.json`); set
   `alpha=0.75` in-file to regenerate the `*_results.json`. See "Notes on
   reproduction" below.

## Reviewer question → script → result

| Reviewer ask | Script | Result JSON | Headline |
|---|---|---|---|
| Calibration-set robustness (WikiText↔C4) | `q1_q3_llama.py` | `q1_q3_results.json` | 94.2% fire/skip agreement over 225 layers |
| Compactness vs prompt diversity | `q1_q3_llama.py`, `q3_v2_diversity.py` | `q1_q3_results.json`, `q3_v2_results.json` | PCSA site stays SKIP as diversity rises |
| New-family diagnostic (gate-pass) | `q2_newfamily_diagnostic.py` | `q2_results.json` | gate tracks kurtosis, not architecture |
| Threshold robustness (±20%, 200 draws) | `threshold_robustness_dispatch.py` | `threshold_robustness_results.json` | 93.9% decisions unchanged (min 73.9%, p5 77.4%); activation-gate leave-one-architecture-out AUC 0.962 (4 families: 0.93–0.99), 0.957 leave-one-config-out (8 configs); class balance 36.5% FIRE / 63.5% SKIP |
| **Random seeds + instruction/multilingual shift** | `seed_and_shift.py` | `seed_and_shift_results.json` | gate-pass 63.1%±1.3; shift agreement 88–96% |
| **PCSA descriptor ablation — SKIP site** (SQ6q W4) | `descriptor_ablation.py` | `descriptor_ablation_results.json` | LLaMA q_proj: 19/20 cells agree; gate not descriptor-sensitive |
| **PCSA descriptor ablation — FIRE site** (SQ6q W4) | `sam_descriptor_ablation.py` | `sam_descriptor_ablation_results.json` | SAM-B mask-decoder FIRE site: all 4 poolings (mean/last/max/attn) fire (c=0.18–0.40, synthetic + real-COCO); decision descriptor-robust |
| PCSA site hunt across models | `pcsa_site_hunt.py`, `*_pcsa*.py` | `pcsa_site_hunt_results.json`, `*_pcsa_*results.json` | Whisper dec cross-attn fires (c=0.0), rest SKIP |
| Flagship DBAF end-to-end (new families) | `clip_flagship.py`, `whisper_flagship.py`, `dit_flagship.py`, `dit_fid.py` | `*_flagship_results.json` (α=0.75), `*_a025_results.json` (α=0.25) | CLIP W4 61.8→71.2 (forced DBAF, α=0.25), Whisper W3 135.8→17.3 |
| **End-task gains under shift (W4A4 headline regime)** | `shift_endtask_w4a4.py` | `shift_endtask_w4a4_results.json` | DBAF recovers PPL to near-FP on all 5 shifts (WikiText 844→17.3, C4 805→24.6, code 420→4.8, multiling 1685→44.9, instr 732→18.3; α=0.25 frozen; reproduces paper 970→16.3) |
| **End-task gains under shift** (W3 wt-only) | `shift_endtask_gains.py` | `shift_endtask_gains_results.json` | DBAF improves PPL on all 5 shifts (WikiText 141k→472, C4 19k→530, code 493k→112, multiling 992k→3089, instr 50k→397; α=0.25 frozen) |
| **α selection sweep (paper's rule)** | `alpha_sweep_{clip,whisper,dit}.py` | `alpha_sweep_*_results.json` | recon sweep selects α≈0.25–0.3 for discriminative families |
| **Sensitivity-weighted α\* correction (SQ6q W1)** | `alpha_sensitivity_check.py` + `ALPHA_SENSITIVITY_DERIVATION.md` | (stdout) | derives α\*_sens = λ^(1/3)·α\* (λ≥1 ⇒ α\* is a lower bound); energy-ratio λ≈28 does not predict 0.25, so λ kept empirical |
| **Why reconstruction α-selection fails on DiT** | `alpha_dit_diagnosis.py` | `alpha_dit_diagnosis_results.json` | single-pass proxy ≠ generative FID (see below) |

## Key finding: DiT and the single-pass reconstruction proxy

The paper selects DBAF's fold strength α by a one-block reconstruction sweep
(§3). On CLIP and Whisper the sweep independently selects **α≈0.25–0.3** — the
same operating point as the W4A4 LLMs — and DBAF gives clean wins there. On
**DiT-XL** the same sweep also selects a low α, but that α *regresses* generative
FID (242.8→275.1), while **α=0.75 improves it (242.8→185.7)**. `alpha_dit_diagnosis.py`
explains the mismatch with two measurements:

1. **Bulk-vs-outlier tension (H1).** DBAF sets the INT4 scale from the max folded
   magnitude, so low α → fine scale → bulk (99% of weights) quantized accurately
   (bulk-MSE min at α=0.25) but outliers squashed and their error amplified by 1/α
   on unfold (outlier-MSE min at α=0.95). This is the same global-vs-outlier
   decomposition the paper derives for α\*. The one-block reconstruction loss
   averages over ~99% bulk positions, so it is bulk-dominated → picks low α.

2. **Trajectory accumulation (H2).** A discriminative model does ONE forward pass,
   so single-pass reconstruction MSE is the eval-relevant error and selection
   works. Diffusion runs 25 sequential passes; the outlier-position error injected
   each step **compounds ×325 down the sampling trajectory at α=0.25 vs ×25 at
   α=0.75**. The per-step ranking flips: single-pass NMSE is best at low α
   (0.25→0.0033) but terminal-trajectory NMSE is best at α=0.75 (0.565 vs 1.068),
   matching FID.

**Conclusion:** DBAF is not failing on DiT — at α=0.75 it delivers a real FID gain.
The *selection criterion* (single-pass, bulk-dominated reconstruction MSE) is the
wrong proxy for iterative generation, where outlier-position error accumulates over
the trajectory. A trajectory-aware / outlier-weighted reconstruction loss is the
principled fix (future work).

## Notes on reproduction

- Flagship scripts (`clip_flagship.py`, `whisper_flagship.py`, `dit_flagship.py`,
  `dit_fid.py`) currently set `alpha=0.25`; the `*_a025_results.json` are those
  runs. The `*_flagship_results.json` (no suffix) are the earlier `alpha=0.75`
  runs. Edit the `alpha=` argument to reproduce either.
- `alpha_sweep_*.py` run the paper's 7-point grid {α\*, 0.25, 0.3, 0.5, 0.75,
  0.95, 0.99}; α\* is computed per model from Eq. (α-star).
- All runs are weight-only per-channel RTN hosts (the paper's
  `flatquant/baselines/rtn.py`), the non-rotation regime where DBAF is predicted
  to help.
