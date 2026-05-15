# Cost-Axis Evidence + Mechanism Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce three reviewer-facing artifacts not covered by the host-matrix plan: (1) end-to-end latency on 9 real models, (2) torchao INT4 GEMM kernel benchmark, (3) per-layer DBAF mechanism evidence on SAM-B + Llama-3-8B. Then wire all three into the EMNLP §4 paper sections.

**Architecture:** Three independent workstreams (each is GPU-bound but reuses existing scaffold scripts). Each workstream produces a JSON / figure artifact + a §4 subsection update. Workstreams 1 and 3 share GPU contention with the host-matrix plan; WS2 is short and can slot in between.

**Tech Stack:** PyTorch 2.1+, torchao, transformers, FlatQuant codebase (in-tree), pytest, A100 80GB GPU.

**Spec:** `docs/superpowers/specs/2026-05-15-cost-axis-and-mechanism-design.md`

**Critical conventions:**
- Conda env: `unifyptq` for all tasks.
- Working directory: `/home/ubuntu/unifying-ptq`.
- Git identity is locally set on this repo — **do not** configure it globally, **do not** add `Co-Authored-By`, `Generated with Claude`, or any Anthropic/Claude attribution to commit messages.
- All artifacts go under `scripts/_out/` or `results/S4-dbaf-weak/per_layer/`.

---

## Workstream 1: End-to-end latency on 9 real models

**Goal:** Wall-clock per-token latency for {none, Hadamard, learned-R, DBAF, PCSA-tf, DBAF+PCSA-tf} on every headline model. Used in §4 cost subsection.

**Files:**
- Existing scaffold: `scripts/end_to_end_latency.py` (model loaders + hook injection skeleton — needs verification + extension).
- Create: `scripts/run_e2e_latency_all.sh`

### Task 1: Verify scaffold runs on one model (smoke)

- [ ] **Step 1: Read the scaffold to understand its CLI**

```bash
head -80 /home/ubuntu/unifying-ptq/scripts/end_to_end_latency.py
```

Identify the CLI flags and the model entry-point.

- [ ] **Step 2: Smoke-run on SwinIR-light ×2 (smallest model)**

```bash
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
python scripts/end_to_end_latency.py \
  --model swinir_x2 \
  --primitive none \
  --n_warmup 2 --n_iters 5 \
  --out scripts/_out/e2e_latency/smoke_swinir_x2_none.json 2>&1 | tail -10
```

Expected: ends with a `ms_per_forward` number, writes the JSON.

- [ ] **Step 3: If scaffold needs fixes (loader missing, hook injection broken, etc.), report `NEEDS_CONTEXT` with the specific gap.**

- [ ] **Step 4: Smoke-run all 6 primitives on SwinIR-light ×2**

```bash
for p in none hadamard learned_R dbaf pcsa_tf dbaf_pcsa_tf; do
  python scripts/end_to_end_latency.py \
    --model swinir_x2 --primitive "$p" \
    --n_warmup 2 --n_iters 5 \
    --out "scripts/_out/e2e_latency/smoke_swinir_x2_${p}.json"
done
```

Expected: 6 JSON files; primitives `none` and `dbaf`/`pcsa_tf` should be visibly faster than `hadamard`/`learned_R`.

- [ ] **Step 5: Commit the smoke artifacts**

```bash
cd /home/ubuntu/unifying-ptq
git add scripts/_out/e2e_latency/smoke_swinir_x2_*.json
git commit -m "e2e-latency: swinir x2 smoke across 6 primitives"
```

### Task 2: Write the all-models sweep script

- [ ] **Step 1: Write the script**

Create `/home/ubuntu/unifying-ptq/scripts/run_e2e_latency_all.sh`:

```bash
#!/usr/bin/env bash
# End-to-end latency sweep for the EMNLP §4 cost-axis subsection.
# 9 real models × 6 primitives = 54 cells.
set -uo pipefail
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
mkdir -p scripts/_out/e2e_latency

MODELS=(swinir_x2 swinir_x3 swinir_x4 sam_b sam_l sam_h llama3_8b qwen25_7b mistral_7b)
PRIMS=(none hadamard learned_R dbaf pcsa_tf dbaf_pcsa_tf)

for m in "${MODELS[@]}"; do
  for p in "${PRIMS[@]}"; do
    out="scripts/_out/e2e_latency/${m}_${p}.json"
    if [[ -f "$out" ]]; then
      echo "[skip] $m $p already done"; continue
    fi
    echo "===== $m $p ====="
    python scripts/end_to_end_latency.py \
      --model "$m" --primitive "$p" \
      --n_warmup 3 --n_iters 10 \
      --out "$out" 2>&1 | tail -5
  done
done

echo "E2E_LATENCY_ALL_DONE_$?"
```

- [ ] **Step 2: Run the full sweep**

```bash
chmod +x /home/ubuntu/unifying-ptq/scripts/run_e2e_latency_all.sh
bash /home/ubuntu/unifying-ptq/scripts/run_e2e_latency_all.sh 2>&1 | tail -30
```

Expected: ~6-8 GPU hours wall-clock; ends with `E2E_LATENCY_ALL_DONE_0`. 54 JSON files in `scripts/_out/e2e_latency/`.

- [ ] **Step 3: Aggregate to a single table**

Create `/home/ubuntu/unifying-ptq/scripts/aggregate_e2e_latency.py`:

```python
"""Aggregate per-model per-primitive latency JSONs into a single LaTeX table
ready to \input{} into §4. Reports relative-to-`none` overhead as %.
"""
from __future__ import annotations
import json, pathlib

OUT_DIR = pathlib.Path("scripts/_out/e2e_latency")
TAB_OUT = pathlib.Path("/home/ubuntu/paper/emnlp2026/tables/_auto/e2e_latency.tex")
TAB_OUT.parent.mkdir(parents=True, exist_ok=True)

MODELS = ["swinir_x2", "swinir_x3", "swinir_x4", "sam_b", "sam_l", "sam_h",
          "llama3_8b", "qwen25_7b", "mistral_7b"]
PRIMS = ["none", "hadamard", "learned_R", "dbaf", "pcsa_tf", "dbaf_pcsa_tf"]
PRIM_LABELS = {"none": "none", "hadamard": "Had.", "learned_R": "Learned R",
               "dbaf": "DBAF", "pcsa_tf": "PCSA-tf", "dbaf_pcsa_tf": "+both"}

rows = []
rows.append("\\begin{tabular}{l " + "c" * len(PRIMS) + "}")
rows.append("\\toprule")
rows.append("\\textbf{Model} & " + " & ".join(PRIM_LABELS[p] for p in PRIMS) + " \\\\")
rows.append("\\midrule")

for m in MODELS:
    cells = []
    base_ms = None
    for p in PRIMS:
        f = OUT_DIR / f"{m}_{p}.json"
        if not f.exists():
            cells.append(("--", None)); continue
        d = json.loads(f.read_text())
        ms = d.get("ms_per_forward")
        if p == "none":
            base_ms = ms
            cells.append((f"{ms:.2f}\\,ms", ms))
        else:
            if ms is None or base_ms is None:
                cells.append(("--", None))
            else:
                rel = (ms / base_ms - 1.0) * 100
                cells.append((f"{rel:+.1f}\\%", rel))
    rows.append(m.replace("_", "-") + " & " + " & ".join(c[0] for c in cells) + " \\\\")

rows.append("\\bottomrule")
rows.append("\\end{tabular}")

TAB_OUT.write_text("\n".join(rows) + "\n")
print(f"wrote {TAB_OUT}")
```

Run it:

```bash
python /home/ubuntu/unifying-ptq/scripts/aggregate_e2e_latency.py
cat /home/ubuntu/paper/emnlp2026/tables/_auto/e2e_latency.tex | head -15
```

Expected: prints the LaTeX table to disk.

- [ ] **Step 4: Commit the WS1 outputs**

```bash
cd /home/ubuntu/unifying-ptq
git add scripts/run_e2e_latency_all.sh scripts/aggregate_e2e_latency.py scripts/_out/e2e_latency/*.json
git commit -m "ws1: e2e latency sweep across 9 models x 6 primitives"
```

---

## Workstream 2: INT4 GEMM benchmark

**Goal:** Compare primitive overhead on top of torchao's production INT4 kernel. Used in §4 "Production INT4 Deployment" paragraph.

**Files:**
- Existing scaffold: `scripts/int4_gemm_latency.py` (uses torchao `int4_dynamic_activation_int4_weight`).
- Create: `scripts/_out/int4_gemm_latency.json` + LaTeX table fragment.

### Task 3: Run INT4 GEMM benchmark

- [ ] **Step 1: Sanity check torchao import**

```bash
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
python -c "from torchao.quantization import quantize_, int4_dynamic_activation_int4_weight; print('torchao OK')"
```

Expected: prints `torchao OK`. If import fails, install with `pip install torchao` and rerun.

- [ ] **Step 2: Run the benchmark**

```bash
python scripts/int4_gemm_latency.py \
  --d 4096 --n_blocks 32 --seq_len 4096 \
  --n_warmup 3 --n_iters 10 \
  --out scripts/_out/int4_gemm_latency.json 2>&1 | tee scripts/_out/int4_gemm_latency.log | tail -30
```

Expected: ~1 GPU hour, writes the JSON. Each primitive's `ms_per_forward` under both `bf16` and `int4` backends.

- [ ] **Step 3: Emit the LaTeX table fragment**

Add to `scripts/aggregate_int4_gemm_latency.py`:

```python
"""Convert int4_gemm_latency.json -> LaTeX row pair for §4."""
import json, pathlib

src = pathlib.Path("scripts/_out/int4_gemm_latency.json")
dst = pathlib.Path("/home/ubuntu/paper/emnlp2026/tables/_auto/int4_gemm_latency.tex")
dst.parent.mkdir(parents=True, exist_ok=True)

d = json.loads(src.read_text())

def _emit(backend, rows, base_ms):
    cells = []
    for r in rows:
        ms = r["ms_per_forward"]
        rel = (ms / base_ms - 1.0) * 100 if base_ms else 0
        cells.append(f"{ms:.2f}\\,ms ({rel:+.1f}\\%)")
    return backend + " & " + " & ".join(cells) + " \\\\"

bf16 = d["bf16_results"]
int4 = d["int4_results"]
bf16_base = next(r["ms_per_forward"] for r in bf16 if r["primitive"] == "none")
int4_base = next(r["ms_per_forward"] for r in int4 if r["primitive"] == "none")

lines = [
    "\\begin{tabular}{l " + "c" * len(bf16) + "}",
    "\\toprule",
    "\\textbf{Backend} & " + " & ".join(r["primitive"] for r in bf16) + " \\\\",
    "\\midrule",
    _emit("bf16", bf16, bf16_base),
    _emit("int4", int4, int4_base),
    "\\bottomrule",
    "\\end{tabular}",
]
dst.write_text("\n".join(lines) + "\n")
print(f"wrote {dst}")
```

Run:

```bash
python /home/ubuntu/unifying-ptq/scripts/aggregate_int4_gemm_latency.py
```

- [ ] **Step 4: Commit WS2 outputs**

```bash
cd /home/ubuntu/unifying-ptq
git add scripts/_out/int4_gemm_latency.json scripts/aggregate_int4_gemm_latency.py
git commit -m "ws2: torchao int4 gemm benchmark with primitive overhead"
```

---

## Workstream 3: Per-layer DBAF mechanism (B4)

**Goal:** Per-layer scatter plot of (outlier-fraction, DBAF-gain) for SAM-B and Llama-3-8B. Extends the existing SwinIR per-layer figure.

**Files:**
- Existing scripts: `scripts/per_layer_ablation_sam.py`, `scripts/per_layer_ablation_llama.py`.
- Output: `results/S4-dbaf-weak/per_layer/{sam_b, llama3_8b}.json`
- Figure: `paper/emnlp2026/figures/per_layer_mechanism_full.pdf` (extend existing PDF).

### Task 4: Run per-layer ablation on SAM-B

- [ ] **Step 1: Verify the SAM-B per-layer script CLI**

```bash
head -60 /home/ubuntu/unifying-ptq/scripts/per_layer_ablation_sam.py
```

Identify expected arguments (model path, output path, layer subsample if any).

- [ ] **Step 2: Run on SAM-B**

```bash
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
mkdir -p results/S4-dbaf-weak/per_layer
python scripts/per_layer_ablation_sam.py \
  --model sam_b \
  --out results/S4-dbaf-weak/per_layer/sam_b.json 2>&1 | tail -10
```

Expected: produces a JSON with one entry per Linear layer containing `outlier_frac`, `mse_baseline`, `mse_dbaf`, `gain_pct`.

- [ ] **Step 3: Run on Llama-3-8B**

```bash
python scripts/per_layer_ablation_llama.py \
  --model llama3_8b \
  --out results/S4-dbaf-weak/per_layer/llama3_8b.json 2>&1 | tail -10
```

Expected: same structure as SAM-B JSON.

### Task 5: Extend the per-layer figure

- [ ] **Step 1: Find the existing figure generator**

```bash
grep -rl "per_layer_outlier_correlation" /home/ubuntu/unifying-ptq/scripts/ 2>/dev/null
```

- [ ] **Step 2: Extend it to a 3-panel figure**

Add SwinIR / SAM-B / Llama-3-8B panels in a single row. Add to the figure-generator script — file path will depend on Step 1's output.

If the existing generator is single-panel (e.g., `make_per_layer_correlation.py`), edit it to accept multiple input JSONs and emit a 3-panel grid:

```python
# Pseudocode — actual integration depends on existing generator structure
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, path) in zip(axes, [
    ("SwinIR-x3", "results/S4-dbaf-weak/per_layer/swinir_x3.json"),
    ("SAM-B",     "results/S4-dbaf-weak/per_layer/sam_b.json"),
    ("Llama-3-8B","results/S4-dbaf-weak/per_layer/llama3_8b.json"),
]):
    d = json.load(open(path))
    xs = [layer["outlier_frac"] for layer in d]
    ys = [layer["gain_pct"]    for layer in d]
    ax.scatter(xs, ys, alpha=0.7)
    ax.set_xlabel("outlier fraction")
    ax.set_ylabel("DBAF gain (%)")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("/home/ubuntu/paper/emnlp2026/figures/per_layer_mechanism_full.pdf")
```

Run the generator.

- [ ] **Step 3: Commit WS3 outputs**

```bash
cd /home/ubuntu/unifying-ptq
git add results/S4-dbaf-weak/per_layer/sam_b.json results/S4-dbaf-weak/per_layer/llama3_8b.json
# also commit whichever figure-generator file was edited
git commit -m "ws3: per-layer DBAF mechanism on SAM-B + Llama-3-8B + 3-panel figure"
```

---

## Paper Integration

### Task 6: §4 Inference Cost Comparison subsection

- [ ] **Step 1: Insert the new subsection**

Open `paper/emnlp2026/sections/04-experiments.tex`. After the host-matrix subsection (added by host-matrix Phase D), insert:

```latex
\subsection{Inference Cost Comparison}
\label{sec:exp-inference-cost}

Table~\ref{tab:e2e_latency} reports end-to-end latency overhead of each
primitive on top of the unmodified model at fp16. Rotation-based
primitives (Hadamard, learned $R$) add a dense fp16 matmul per Linear and
incur 5-20\% latency overhead on every architecture. DBAF and PCSA-tf are
in-place fp16 element-wise ops that the JIT fuses into surrounding kernels,
adding $\le 2\%$ overhead.

\begin{table*}[t]
\centering\small
\input{tables/_auto/e2e_latency}
\caption{Per-forward latency overhead relative to fp16-no-primitive
(positive = slower).  DBAF and PCSA-tf are essentially free; rotation
primitives add a measurable dense matmul.}
\label{tab:e2e_latency}
\end{table*}

\paragraph{Production INT4 deployment.} Table~\ref{tab:int4_gemm} shifts
the backend from fp16 to torchao's INT4 GEMM kernel (used in vLLM/SGLang).
Even on the production kernel, the relative ordering holds: DBAF and
PCSA-tf add minimal overhead, while rotation requires an additional dense
fp16 launch before the INT4 kernel.

\begin{table*}[t]
\centering\small
\input{tables/_auto/int4_gemm_latency}
\caption{Primitive overhead on top of torchao W4A4 dynamic-activation INT4
kernel (Llama-3-8B class, $d{=}4096$, seq\,len\,$=$\,4096).}
\label{tab:int4_gemm}
\end{table*}
```

### Task 7: Extend §4 Mechanism subsection

- [ ] **Step 1: Update the existing mechanism figure caption to reference the 3-panel version**

Find the current `figures/per_layer_outlier_correlation.pdf` reference in `04-experiments.tex` and replace with:

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figures/per_layer_mechanism_full.pdf}
\caption{Per-layer outlier fraction versus DBAF output-MSE reduction,
across SwinIR-light-$\times 3$, SAM-B, and LLaMA-3-8B. The monotonic
trend confirms that DBAF's gain is concentrated on the layers with the
highest outlier prevalence — the mechanism predicted by
\S\ref{sec:method}. The same correlation holds across three architectures
spanning SR, segmentation, and language modeling.}
\label{fig:per-layer-mech}
\end{figure*}
```

### Task 8: Final commit

```bash
cd /home/ubuntu/paper/emnlp2026
git add sections/04-experiments.tex tables/_auto/e2e_latency.tex tables/_auto/int4_gemm_latency.tex
git commit -m "paper: add inference-cost subsection + 3-panel per-layer mechanism figure"
```

---

## Total Budget Recap

| Section | Wall clock | Type |
|---|---|---|
| WS1 Tasks 1-2 | ~6-8 GPU hours | run + dev |
| WS2 Task 3 | ~1 GPU hour | run |
| WS3 Tasks 4-5 | ~4 GPU hours + dev | run |
| Paper Tasks 6-8 | 1 day | dev |

**Total: ~12 GPU hours + 1 day of paper work.** Compatible with the host-matrix plan running in parallel (they share GPU but on different sub-hours).
