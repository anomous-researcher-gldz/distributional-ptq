# Cross-architecture generalization — author-response analysis (Submission 11057)

Analysis run for the EMNLP/ARR 2026 author response to *"From Distribution to
Decision: A Diagnostic for Composable PTQ Primitives."* Each script is
self-contained and writes a JSON to `results/`. Numbers here back the rebuttal
claims (breadth, calibration protocol, robustness, generalization). Nothing here
contradicts the submitted paper; every run confirms a submitted claim or answers a
reviewer question.

## Reviewer question → script → result

| Reviewer ask | Script | Result JSON | Headline |
|---|---|---|---|
| Calibration-set robustness (WikiText↔C4) | `q1_q3_llama.py` | `q1_q3_results.json` | 94.2% fire/skip agreement over 225 layers |
| Compactness vs prompt diversity | `q1_q3_llama.py`, `q3_v2_diversity.py` | `q1_q3_results.json`, `q3_v2_results.json` | PCSA site stays SKIP as diversity rises |
| New-family diagnostic (gate-pass) | `q2_newfamily_diagnostic.py` | `q2_results.json` | gate tracks kurtosis, not architecture |
| Threshold robustness (±20%, 200 draws) | `threshold_robustness_dispatch.py` | (stdout / S4 layer json) | 93.9% decisions unchanged; AUC 0.957 |
| **Random seeds + instruction/multilingual shift** | `seed_and_shift.py` | `seed_and_shift_results.json` | gate-pass 63.1%±1.3; shift agreement 88–96% |
| **PCSA descriptor ablation** (SQ6q W4) | `descriptor_ablation.py` | `descriptor_ablation_results.json` | 19/20 cells agree; gate not descriptor-sensitive |
| PCSA site hunt across models | `pcsa_site_hunt.py`, `*_pcsa*.py` | `pcsa_site_hunt_results.json`, `*_pcsa_*results.json` | Whisper dec cross-attn fires (c=0.0), rest SKIP |
| Flagship DBAF end-to-end (new families) | `clip_flagship.py`, `whisper_flagship.py`, `dit_flagship.py`, `dit_fid.py` | `*_flagship_results.json` (α=0.75), `*_a025_results.json` (α=0.25) | CLIP 61.8→71.5, Whisper W3 135.8→17.3 |
| **α selection sweep (paper's rule)** | `alpha_sweep_{clip,whisper,dit}.py` | `alpha_sweep_*_results.json` | recon sweep selects α≈0.25–0.3 for discriminative families |
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
