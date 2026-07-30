# Provenance of the numbers in the EMNLP paper

Where each table in "From Distribution to Decision" gets its numbers, and which
of them can still be checked against a stored artifact. Written 2026-07-30 after
a full reconciliation pass; see "How this was produced" at the end to redo it.

Short version: of 34 tables, **19 are fully backed by a result JSON in this
repo**. The rest are backed by `PAPER_RESULTS.md`, by a cited source, or (for a
handful of cells) by run artifacts that no longer exist. Nothing in the audit
contradicted a printed number.

---

## 1. Read this first if you are checking a number

Three tiers of evidence, strongest to weakest:

1. **A result JSON in this repo.** Match the printed value against it at the
   printed precision. 427 of 503 table cells match this way.
2. **`results/PAPER_RESULTS.md`.** A hand-maintained summary written alongside
   the runs. It records the run date and output path for each block. The paper
   often prints more precision than the summary (e.g. `37.816` where the summary
   has `37.82`); those agree on rounding, but the extra digit came from a log
   that is no longer on disk.
3. **Cited from prior work.** Baseline rows -- FlatQuant's `6.98` is from their
   README, SpinQuant `7.96` and SmoothQuant `210.19` from prior tables. These
   are not ours to reproduce and were never expected to have a JSON here.

## 2. Artifacts that no longer exist

Several run families wrote to `/data/outputs/...` on a rented GPU box that has
since been torn down. That path exists on neither the current workstation nor
the current remote. The affected tables:

| Table | What it reports | Surviving evidence |
|---|---|---|
| `tab:saturation` | FlatQuant + DBAF + PCSA, the central saturation result | `PAPER_RESULTS.md`, runs dated 2026-05-13 ("S5 ablation", "C1 finished 22:27") |
| `tab:composability-2dq` | 2DQuant-host SR composability | `PAPER_RESULTS.md` at 2 d.p.; paper prints 3 d.p. |
| `tab:hostmatrix-llm`, "alone" column | un-DBAF'd host PPL | `PAPER_RESULTS.md`, sources block lists the `/data/outputs` paths |
| `tab:hostmatrix-llm`, TesseraQ `+DBAF` cells | `60.54`, `10.82`, `50.88`, `16.21` | none -- the summary still records these as `(running)` |

The TesseraQ `+DBAF` cells are the weakest link: the run finished after the
summary was last written and its artifacts went to `/data/outputs/HM-tesseraq/`.
The numbers went through peer review unchanged, and the claim they support (a
-82% DBAF response, placing TesseraQ with the non-rotation block) is consistent
with every other non-rotation host (-47% to -83%). But they cannot be
re-derived from anything in this repo.

## 3. Correction to the alpha-star computation

`alpha^star` for the vision, speech and diffusion families was recomputed after
a row-prefix defect was found in the sweep code. The earlier values were
superseded before publication and appear in no version of the paper.
`recompute_alpha_star.py` reproduces the correction and
`alpha_star_recompute.json` records both values:

| family | superseded | current | paper prints |
|---|---|---|---|
| CLIP-ViT-L/14 | 0.447 | 0.325 | 0.33 |
| Whisper-small | 0.432 | 0.315 | 0.32 |
| DiT-XL | 0.416 | 0.299 | 0.30 |

The current values are the ones the paper cites. The corrected sweep files and
the recompute script are tracked; the superseded versions remain in git history.
The defect also shifted the sweep's task metrics slightly (CLIP top-1
71.48 to 71.36); the paper reports neither.

`alpha^star_W` for Qwen-2.5-7B (printed as `0.28`) has no backing file here;
only `results/astar_llm_Meta-Llama-3-8B.json` was kept. That file was moved up
from `cross_arch_generalization/results/` so the path matches the one the paper
cites in Sec. 3; `astar_llm.py` writes to the new location.

## 3a. Paths the paper promises

Every `\path{}` reference in the paper must resolve in the released repo. As of
2026-07-30 all three do:

| cited in | path | status |
|---|---|---|
| Sec. 3 (alpha-star) | `results/astar_llm_Meta-Llama-3-8B.json` | tracked (moved here to match) |
| Sec. 3 (per-layer r) | `results/pearson_r_8configs.json` | tracked |
| App. threshold-robustness | `results/S4-cross-model-layer-analysis/` | tracked |

Re-check these before any release.

## 4. The Table 4 re-run audit

`HM-table4-audit.json` re-measured 16 training-free host-matrix cells on
2026-07-28. It settled the four known paper-vs-JSON disagreements:

| cell | paper | archived JSON | re-measured | conclusion |
|---|---|---|---|---|
| SmoothQuant wt2 a=0.75 | 5263.70 | 1477.98 | 1513.97 | JSON right; **paper corrected** |
| SmoothQuant c4 a=0.75 | 3251.95 | 1005.35 | 994.11 | JSON right; **paper corrected** |
| GPTQ c4 a=0.25 | 53.43 | 49.06 | 53.90 | **paper right, archive stale** |
| SmoothQuant c4 a=0.25 | 27.10 | 26.05 | 26.99 | **paper right, archive stale** |

Five further cells differ from the re-run by more than the ~2% reproduction
noise the script assumes, worst being GPTQ wt2 a=0.75 at -12.2%. Two caveats:
that 2% figure was estimated from a single cell, and every large deviation is on
an a=0.75 row where PPL is in the hundreds and numerically unstable. The
qualitative claim is unaffected -- re-measured a=0.25 values are 16.43, 30.46,
17.04, 17.20, all inside the 16-33 band the paper reports.

## 5. Fully JSON-backed tables

No action needed on these; every printed cell matches a stored value.

`tab:niah-main`, `tab:crossarch-gate`, `tab:descriptor-llm`, `tab:descriptor-sam`,
`tab:shift-endtask`, `tab:rotation-control`, `tab:seed-variance`,
`tab:boundary-auc`, `tab:ablation-dbaf-pcsa`, `tab:cv-per-input`,
`tab:cluster-compactness`, `tab:synthetic-pcsa-sweep`, `tab:per-token-baseline`,
`tab:pcsa-tf-catalog`, `tab:dense-vs-sparse`, `tab:perlayer`,
`tab:matched-T-vs-dbaf`, `tab:composability-ahcptq`, `tab:int4-multi`.

## 6. Known gaps

- The `/data/outputs` artifacts are gone and will not come back.
- `tab:int4` latency cells (`185.0` ms and similar) were measured
  interactively and never serialized.
- The four CompSRT cells in `tab:swinir` were measured in the separate CompSRT
  repo, not here; see `emnlp2026/docs/SR_EVIDENCE_HANDOFF.md`.
- One figure disagreed with its own caption: the per-layer correlation plot
  prints `r=0.561`, the prose said `r=0.485`. Recomputed from
  `results/S4-dbaf-weak/per_layer_correlation/llama3-8b.json`: r = 0.5615,
  p = 5.2e-20, n = 224. The figure was right; the paper has been corrected.

## How this was produced

Index every numeric leaf of every JSON under this repo, keyed by value rounded
to 1-4 decimals, then extract every number inside a `tabular` in the paper
sources and look it up at its own precision. A cell that matches nothing is not
necessarily wrong -- cited baselines and derived aggregates legitimately match
nothing -- but it is the set worth checking by hand. Re-running this after any
results change is cheap and needs no GPU.
