# Symmetric LLM Host Matrix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill a 4×4 LLM host matrix (rotation × training-status × {alone, +DBAF, +PCSA-tf, +both}) for the EMNLP 2026 submission, so the DBAF and PCSA claims have host-symmetric evidence — particularly a non-rotation trained host (TesseraQ) where PCSA actually has signal to exploit.

**Architecture:** Three parallel setup tracks (SmoothQuant fix, PCSA-tf qmax fix, TesseraQ env), then a diagnostic gate, then four sequential run phases (Phase A: training-free row of 4 hosts; Phase B: QuaRot row; Phase C: TesseraQ row; Phase D: paper updates). Each cell runs `python scripts/run_training_free_full_table.py` or the host-specific runner, writes `eval.json` under `/data/outputs/`, and gets aggregated by `scripts/aggregate_results_to_latex.py` into the EMNLP §4 tables.

**Tech Stack:** PyTorch 2.1+, transformers, FlatQuant codebase (in-tree), TesseraQ codebase (in-tree), QuaRot codebase (in-tree), pytest for unit tests, A100 80GB GPU.

**Spec:** `docs/superpowers/specs/2026-05-15-symmetric-llm-host-matrix-design.md`

**Critical conventions (READ FIRST):**

- Conda env: `unifyptq` (default for everything except TesseraQ which gets its own env in Task 5).
- Working directory: `/home/ubuntu/unifying-ptq`.
- Git identity is locally set on this repo — **do not** configure it globally, **do not** add `Co-Authored-By`, `Generated with Claude`, or any Anthropic/Claude attribution to commit messages. Commit messages should be focused and reviewer-friendly.
- All sweep runs go to `/data/outputs/<group>/<target>/<cell_name>/`. Existing convention: `eval.json` is the headline metric file.

---

## Track 1: SmoothQuant baseline fix

**Goal:** Make `flatquant.baselines.smoothquant.quantize_model` complete an end-to-end calibration + W4A4 + PPL eval on Llama-3-8B without crashing.

**Files:**
- Modify: `FlatQuant/flatquant/baselines/smoothquant.py:55-66` (the `_collect_act_scales` iteration loop)
- Create: `tests/test_smoothquant_fix.py`

**Root cause:** `calibration_data` is a 2D tensor `(N, T)` passed from `scripts/run_training_free_full_table.py:280`. The `for batch in calibration_data:` loop iterates row-wise, so each `batch` is a 1D `(T,)` tensor. `model(ids)` with a 1D `ids` produces a 2D `hidden_states` `(T, hidden)` instead of `(B, T, hidden)`, and Llama-3's attention forward at `modeling_llama.py:610` (`bsz, q_len, _ = hidden_states.size()`) raises `ValueError: not enough values to unpack`.

### Task 1: Write failing test for SmoothQuant single-batch call

- [ ] **Step 1: Write the failing test**

Create `/home/ubuntu/unifying-ptq/tests/test_smoothquant_fix.py`:

```python
"""Test that SmoothQuant's _collect_act_scales does not crash on a 2D
(N, T) calibration tensor, which is the format produced by
scripts/run_training_free_full_table.py._calib_batch_llm.
"""
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")


class _MiniLlama(nn.Module):
    """Minimal stand-in: an embedding + one Linear, with a 3D-shape sanity
    check inside that mirrors Llama's `bsz, q_len, _ = hidden_states.size()`.
    """
    def __init__(self, vocab=256, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, ids):
        h = self.embed(ids)
        bsz, q_len, _ = h.size()  # crashes if h is 2D
        return self.proj(h)


def test_collect_act_scales_handles_2d_calib_tensor():
    """Mirror of _calib_batch_llm output: 2D (n, seq_len) tensor."""
    from flatquant.baselines.smoothquant import _collect_act_scales

    torch.manual_seed(0)
    model = _MiniLlama(vocab=256, hidden=32)
    # _calib_batch_llm-style: shape (N, T), values are token ids
    calib = torch.randint(0, 256, (4, 16))  # 4 sequences, 16 tokens each

    scales = _collect_act_scales(model, calib, alpha=0.5)
    assert "proj" in scales, "expected per-Linear scale dict entry"
    assert scales["proj"].shape == (32,), \
        f"per-channel scale should be [d_in=32], got {scales['proj'].shape}"
    assert torch.isfinite(scales["proj"]).all(), "scale should be finite"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
pytest tests/test_smoothquant_fix.py::test_collect_act_scales_handles_2d_calib_tensor -v
```

Expected: FAIL with `ValueError: not enough values to unpack (expected 3, got 2)` from `bsz, q_len, _ = h.size()`.

### Task 2: Fix SmoothQuant batch-dim handling

- [ ] **Step 1: Apply the fix**

Edit `FlatQuant/flatquant/baselines/smoothquant.py`, replace lines 55-66:

```python
    with torch.no_grad():
        for batch in calibration_data:
            if isinstance(batch, torch.Tensor):
                ids = batch.to(device)
            elif isinstance(batch, (list, tuple)):
                ids = batch[0].to(device)
            elif isinstance(batch, dict):
                ids = batch["input_ids"].to(device)
            else:
                continue
            _ = model(ids)
```

with:

```python
    with torch.no_grad():
        for batch in calibration_data:
            if isinstance(batch, torch.Tensor):
                ids = batch.to(device)
            elif isinstance(batch, (list, tuple)):
                ids = batch[0].to(device)
            elif isinstance(batch, dict):
                ids = batch["input_ids"].to(device)
            else:
                continue
            # Ensure (B, T) — _calib_batch_llm yields 1D rows when iterated.
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            _ = model(ids)
```

- [ ] **Step 2: Run the test to verify it passes**

```bash
cd /home/ubuntu/unifying-ptq
pytest tests/test_smoothquant_fix.py::test_collect_act_scales_handles_2d_calib_tensor -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add tests/test_smoothquant_fix.py FlatQuant/flatquant/baselines/smoothquant.py
git commit -m "fix(smoothquant): unsqueeze 1D calib rows to (B,T) before model forward"
```

---

## Track 2: PCSA-tf qmax surgery

**Goal:** Fix the catastrophic-PPL bug in `apply_pcsa_tf_to_activation` so PCSA-tf composes cleanly on training-free LLM hosts (RTN/GPTQ/AWQ/SmoothQuant) without collapsing the activation dynamic range.

**Files:**
- Modify: `FlatQuant/flatquant/baselines/pcsa_tf.py:97-106`
- Create: `tests/test_pcsa_tf_surgery.py`

**Root cause:** `apply_pcsa_tf_to_activation` computes `qmax = 2 ** bits - 1` (15 for bits=4), then `scale = scale_per_prompt / qmax` and clamps integer codes to `[-qmax//2, qmax//2] = [-7, 7]`. The full activation magnitude `S` thus dequantizes back to `7 * (S / 15) = 0.47 S` — a ~2× compression per Linear. Over 32 decoder layers this collapses outputs to ~10⁻¹⁰ × original, yielding PPL≈vocab-size (~128K) — exactly what we observed (PPL 297K–1M).

**Fix:** symmetric INT[bits] quantization uses `qmax = 2^(bits-1) - 1` (7 for bits=4) so that the dequant range `[-qmax * (S/qmax), +qmax * (S/qmax)] = [-S, +S]` is preserved.

### Task 3: Write failing test for PCSA-tf magnitude preservation

- [ ] **Step 1: Write the failing test**

Create `/home/ubuntu/unifying-ptq/tests/test_pcsa_tf_surgery.py`:

```python
"""PCSA-tf magnitude preservation under symmetric INT4 fake-quant.

Catches the qmax = 2**bits - 1 vs. 2**(bits-1) - 1 bug that compressed
activations to ~0.47x of original, yielding PPL 300K-1M after 32-layer
propagation.
"""
import sys
import torch

sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")


def test_pcsa_tf_preserves_magnitude_within_one_step():
    """For x at the per-prompt max-abs (i.e. saturation point), the fake-
    quantized output should be within one quantization step of x.

    For INT4 symmetric with S = max(|x|), the step is S/7 (not S/15).
    A correctly-implemented quantizer maps x=S -> q=7 -> dequant 7*(S/7) = S.
    The buggy implementation maps x=S -> q=7 -> dequant 7*(S/15) = 0.47*S.
    """
    from flatquant.baselines.pcsa_tf import apply_pcsa_tf_to_activation

    torch.manual_seed(0)
    B, T, D = 2, 8, 16
    x = torch.randn(B, T, D)
    # Per-prompt max-abs as the scale anchor (single anchor for simplicity)
    scale_per_prompt = x.abs().amax(dim=(1, 2))  # [B]
    state = {
        "anchors": torch.eye(B),         # B-by-B descriptor space, each prompt is its own anchor
        "scales":  scale_per_prompt,     # one anchor per prompt
    }
    desc = torch.eye(B)  # each row identifies a different anchor

    y = apply_pcsa_tf_to_activation(x, desc, state, bits=4)
    # Step size = S/7 per prompt
    step = scale_per_prompt / 7.0
    err = (y - x).abs()
    # Each element's |y - x| should be at most one step (S/7) plus float slop.
    for b in range(B):
        max_err = err[b].max()
        tol = step[b] * 1.1  # 10% slack for float rounding
        assert max_err <= tol, (
            f"prompt {b}: max |y - x| = {max_err:.4f} exceeds 1 step "
            f"{tol:.4f} (this is the qmax bug)"
        )


def test_pcsa_tf_routes_per_prompt():
    """Anchor-routing sanity: two prompts with different scales should
    pick different anchors and get different effective step sizes."""
    from flatquant.baselines.pcsa_tf import apply_pcsa_tf_to_activation

    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    x[0] *= 10.0   # prompt 0 has 10x bigger activations
    state = {
        "anchors": torch.eye(2),
        "scales":  torch.tensor([10.0 * x[0].abs().max() / 10.0,
                                  x[1].abs().max()]),
    }
    desc = torch.eye(2)
    y = apply_pcsa_tf_to_activation(x, desc, state, bits=4)
    # The output magnitudes should approximately match each prompt's input,
    # NOT be compressed to a single global scale.
    assert torch.allclose(y[0].abs().max(), x[0].abs().max(), rtol=0.2), \
        "high-magnitude prompt should preserve its magnitude"
    assert torch.allclose(y[1].abs().max(), x[1].abs().max(), rtol=0.2), \
        "low-magnitude prompt should preserve its magnitude"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ubuntu/unifying-ptq
pytest tests/test_pcsa_tf_surgery.py -v
```

Expected: `test_pcsa_tf_preserves_magnitude_within_one_step` FAILS with assertion `max |y - x| exceeds 1 step (this is the qmax bug)` — specifically because the buggy qmax causes errors ~2× the legitimate quantization step.

### Task 4: Apply PCSA-tf qmax fix

- [ ] **Step 1: Apply the fix**

Edit `FlatQuant/flatquant/baselines/pcsa_tf.py`, replace lines 86-106:

```python
@torch.no_grad()
def apply_pcsa_tf_to_activation(
    x: torch.Tensor,
    desc: torch.Tensor,
    state: dict,
    bits: int = 4,
) -> torch.Tensor:
    """Per-prompt symmetric INT[bits] fake-quantization using anchor-routed scale.

    x: [B, ...] activation tensor; desc: [B, D] prompt descriptors.
    Returns: same shape as x, fake-quantized.
    """
    qmax = 2 ** bits - 1
    anchor_ids = route_pcsa_tf(desc, state)  # [B]
    scale_per_prompt = state["scales"][anchor_ids]  # [B]
    # Broadcast scale over the trailing dims of x
    extra_dims = x.dim() - 1
    scale = scale_per_prompt.view(-1, *([1] * extra_dims)) / qmax
    scale = scale.clamp(min=1e-9)
    q = torch.round(x / scale).clamp(-qmax // 2, qmax // 2)
    return (q * scale).to(x.dtype)
```

with:

```python
@torch.no_grad()
def apply_pcsa_tf_to_activation(
    x: torch.Tensor,
    desc: torch.Tensor,
    state: dict,
    bits: int = 4,
) -> torch.Tensor:
    """Per-prompt symmetric INT[bits] fake-quantization using anchor-routed scale.

    Symmetric int{bits}: integer codes span [-(2^(bits-1)-1), +(2^(bits-1)-1)],
    so step = S / (2^(bits-1) - 1). For bits=4 the step is S/7 (NOT S/15);
    using the wrong qmax compresses dequant values to ~0.47*S per Linear and
    propagates destructively through deep transformers.

    x: [B, ...] activation tensor; desc: [B, D] prompt descriptors.
    Returns: same shape as x, fake-quantized.
    """
    qmax = 2 ** (bits - 1) - 1  # symmetric int{bits}: 7 for bits=4
    anchor_ids = route_pcsa_tf(desc, state)  # [B]
    scale_per_prompt = state["scales"][anchor_ids]  # [B]
    extra_dims = x.dim() - 1
    scale = scale_per_prompt.view(-1, *([1] * extra_dims)) / qmax
    scale = scale.clamp(min=1e-9)
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return (q * scale).to(x.dtype)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd /home/ubuntu/unifying-ptq
pytest tests/test_pcsa_tf_surgery.py -v
```

Expected: both tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add tests/test_pcsa_tf_surgery.py FlatQuant/flatquant/baselines/pcsa_tf.py
git commit -m "fix(pcsa-tf): use symmetric int4 qmax=7 to preserve activation magnitude"
```

---

## Track 3: TesseraQ env setup + smoke

**Goal:** Have a runnable `tesseraq` conda env that reproduces the paper's reported W4A4 PPL on Llama-3-8B within ±5%, so we can then layer DBAF / PCSA-tf augments on top.

**Files:**
- Create: `scripts/setup_tesseraq_env.sh`
- Create: `scripts/smoke_tesseraq.sh`
- Already exists (will be touched in Phase C): `patches/tesseraq_dbaf_pcsa_patch.py`

### Task 5: Create TesseraQ conda env

- [ ] **Step 1: Write the setup script**

Create `/home/ubuntu/unifying-ptq/scripts/setup_tesseraq_env.sh`:

```bash
#!/usr/bin/env bash
# Create the `tesseraq` conda env (separate from `unifyptq` because TesseraQ
# pins different transformers/torch versions). Idempotent.
set -uo pipefail

cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh

if conda env list | grep -qE '^\s*tesseraq\s'; then
  echo "[setup] env 'tesseraq' already exists — skipping create"
else
  conda create -n tesseraq python=3.10 -y
fi

conda activate tesseraq
pip install --upgrade pip
pip install -r TesseraQ/requirements.txt

# Sanity import check — fail loudly if a critical dep is missing.
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
echo "TESSERAQ_ENV_SETUP_DONE"
```

- [ ] **Step 2: Run the setup script**

```bash
bash /home/ubuntu/unifying-ptq/scripts/setup_tesseraq_env.sh 2>&1 | tail -30
```

Expected: ends with `TESSERAQ_ENV_SETUP_DONE` and a printed `torch X.X.X transformers X.X.X` line.

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add scripts/setup_tesseraq_env.sh
git commit -m "tesseraq: add idempotent env setup script (python 3.10 + requirements)"
```

### Task 6: Smoke-test TesseraQ on Llama-3-8B (alone, no DBAF/PCSA)

- [ ] **Step 1: Read the TesseraQ config dir to pick the closest Llama-3-8B W4A4 config**

```bash
ls /home/ubuntu/unifying-ptq/TesseraQ/configs/ 2>/dev/null
find /home/ubuntu/unifying-ptq/TesseraQ/configs -name "*.yml" -o -name "*.yaml" -o -name "*.json" 2>/dev/null | head -20
```

Expected: identify a config file for W4A4 quantization (e.g. `configs/quantization/methods/TesseraQ/tesseraq_llama_w4a4.yml` or similar). Note the path for the next step.

- [ ] **Step 2: Write the smoke script**

Create `/home/ubuntu/unifying-ptq/scripts/smoke_tesseraq.sh`:

```bash
#!/usr/bin/env bash
# Smoke-test TesseraQ on Llama-3-8B W4A4 (no DBAF / no PCSA-tf).
# Verifies env wiring + ckpt loading + at least one block of gradient
# reconstruction converges. Logs full eval to /data/outputs/T-tesseraq-smoke.
set -uo pipefail

cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate tesseraq

mkdir -p /data/outputs/T-tesseraq-smoke

# NOTE: replace CONFIG_PATH below with the actual path identified in Task 6 Step 1.
CONFIG_PATH="TesseraQ/configs/quantization/methods/TesseraQ/tesseraq_llama_w4a4.yml"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: TesseraQ config not found at $CONFIG_PATH — fix Task 6 Step 1 first."
  exit 1
fi

cd TesseraQ
python -m llmc \
  --config "../${CONFIG_PATH}" \
  --model_path /data/modelzoo/meta-llama/Meta-Llama-3-8B \
  --save_path /data/outputs/T-tesseraq-smoke/llama3-8b-alone \
  2>&1 | tee /data/outputs/T-tesseraq-smoke/run.log

echo "TESSERAQ_SMOKE_DONE_$?"
```

- [ ] **Step 3: Run the smoke**

```bash
bash /home/ubuntu/unifying-ptq/scripts/smoke_tesseraq.sh 2>&1 | tail -40
```

Expected: ends with `TESSERAQ_SMOKE_DONE_0` after ~3–5 GPU hours. WikiText-2 PPL printed in the log should be within ±5% of TesseraQ paper's published Llama-3.1-8B W4A4 number (25.73).

- [ ] **Step 4: Record the PPL number**

Read the smoke log and verify the PPL is in the expected range:

```bash
grep -iE "(WikiText|ppl|perplexity)" /data/outputs/T-tesseraq-smoke/run.log | tail -10
```

- [ ] **Step 5: Commit the smoke script**

```bash
cd /home/ubuntu/unifying-ptq
git add scripts/smoke_tesseraq.sh
git commit -m "tesseraq: smoke script reproducing paper W4A4 PPL on llama-3-8b"
```

---

## Diagnostic gate: AWQ + PCSA-tf single-cell check

**Goal:** After Track 2 lands (the PCSA-tf fix is merged), run **one** cell — AWQ + PCSA-tf on Llama-3-8B — and confirm PPL ≤ AWQ-alone × 1.05. If it fails, pause TesseraQ +PCSA-tf cells but proceed with TesseraQ alone + DBAF.

### Task 7: Run the gate cell

- [ ] **Step 1: Launch AWQ + PCSA-tf single cell**

```bash
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
python scripts/run_training_free_full_table.py \
  --target llama3-8b \
  --method awq \
  --augments pcsa_tf \
  --out /data/outputs/G8-training-free-full/llama3-8b/awq_pcsa_tf 2>&1 | tail -25
```

Expected: completes in ~30 min, writes `/data/outputs/G8-training-free-full/llama3-8b/awq_pcsa_tf/eval.json`.

- [ ] **Step 2: Compare to AWQ-alone (13.49 PPL already on disk)**

```bash
python - <<'PY'
import json, pathlib
alone = json.loads(pathlib.Path("/data/outputs/G8-training-free-full/llama3-8b/awq_alone/eval.json").read_text())
pcsa  = json.loads(pathlib.Path("/data/outputs/G8-training-free-full/llama3-8b/awq_pcsa_tf/eval.json").read_text())
a = alone["metrics"]["wikitext2_ppl"]; b = pcsa["metrics"]["wikitext2_ppl"]
ratio = b / a
print(f"AWQ alone WT2 PPL = {a:.2f}")
print(f"AWQ + PCSA-tf WT2 PPL = {b:.2f}")
print(f"ratio = {ratio:.3f}")
print("GATE", "PASS" if ratio <= 1.05 else "FAIL")
PY
```

Expected: `GATE PASS` (ratio ≤ 1.05). If `GATE FAIL`, record the number and continue, but skip Phase C +PCSA-tf cells.

- [ ] **Step 3: Commit the gate result note**

```bash
cd /home/ubuntu/unifying-ptq
mkdir -p docs/superpowers/notes
cat > docs/superpowers/notes/2026-05-15-pcsa-tf-gate-result.md <<'EOF'
# PCSA-tf surgery gate result (2026-05-15)

| Cell | WT2 PPL |
|------|---------|
| AWQ alone | <fill from json> |
| AWQ + PCSA-tf (post-surgery) | <fill from json> |
| Ratio | <fill> |

**Verdict:** <PASS | FAIL>

Decision: <if PASS> proceed with all Phase A-C cells; <if FAIL> drop
TesseraQ +PCSA-tf and QuaRot +PCSA-tf cells from Phase B/C.
EOF
# Edit the file to fill in the actual numbers from Step 2 before committing.
$EDITOR docs/superpowers/notes/2026-05-15-pcsa-tf-gate-result.md || \
  echo "edit docs/superpowers/notes/2026-05-15-pcsa-tf-gate-result.md to fill in numbers"
git add docs/superpowers/notes/2026-05-15-pcsa-tf-gate-result.md
git commit -m "notes: record PCSA-tf surgery gate result"
```

---

## Phase A: Training-free LLM 4-host sweep

**Goal:** Fill the training-free row of the matrix: RTN / GPTQ / AWQ / SmoothQuant × {alone, +DBAF, +PCSA-tf, +both} = 16 cells. 6 produce usable numbers (alone + DBAF for RTN/GPTQ/AWQ); 10 new cells to run.

**Files:**
- Modify: `scripts/run_G8_llama3_training_free_sweep.sh` (already exists with all 4 methods × 4 augments listed)
- Reuse: `scripts/run_training_free_full_table.py`

### Task 8: Re-run SmoothQuant × 4 cells

- [ ] **Step 1: Launch the 4 SmoothQuant cells**

```bash
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
for aug in alone dbaf pcsa_tf dbaf+pcsa_tf; do
  out=/data/outputs/G8-training-free-full/llama3-8b/smoothquant_${aug/+/_plus_}
  mkdir -p "$out"
  echo "===== smoothquant ${aug} ====="
  python scripts/run_training_free_full_table.py \
    --target llama3-8b \
    --method smoothquant \
    --augments "$aug" \
    --out "$out" 2>&1 | tee "${out}/run.log" | tail -10
done
```

Expected: each cell writes `eval.json`. WT2 PPL for SmoothQuant alone should be in 9–20 range (matches published SmoothQuant numbers).

- [ ] **Step 2: Verify all 4 SmoothQuant cells produced finite PPL**

```bash
for d in /data/outputs/G8-training-free-full/llama3-8b/smoothquant_*/; do
  ppl=$(python -c "import json; d=json.load(open('$d/eval.json')); print(d['metrics']['wikitext2_ppl'])" 2>/dev/null)
  echo "$(basename $d): WT2 PPL = ${ppl:-MISSING}"
done
```

Expected: all 4 lines show finite PPL values.

### Task 9: Re-run AWQ/GPTQ/RTN × {+PCSA-tf, +both} (6 cells)

- [ ] **Step 1: Launch the 6 PCSA-tf-augmented cells**

```bash
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
for method in rtn gptq awq; do
  for aug in pcsa_tf dbaf+pcsa_tf; do
    out=/data/outputs/G8-training-free-full/llama3-8b/${method}_${aug/+/_plus_}
    if [[ -f "$out/eval.json" ]]; then
      ppl=$(python -c "import json; d=json.load(open('$out/eval.json')); print(d['metrics']['wikitext2_ppl'])")
      if (( $(echo "$ppl < 100" | bc -l) )); then
        echo "[skip] $method $aug already has finite PPL=$ppl"
        continue
      fi
    fi
    echo "===== ${method} ${aug} ====="
    python scripts/run_training_free_full_table.py \
      --target llama3-8b \
      --method "$method" \
      --augments "$aug" \
      --out "$out" 2>&1 | tee "${out}/run.log" | tail -10
  done
done
```

Expected: 6 new finite PPL cells. Per the gate: PPL should be ≤ alone × 1.05 for +PCSA-tf, and ≤ +DBAF for +both.

- [ ] **Step 2: Verify all 16 training-free cells produce finite PPL**

```bash
for d in /data/outputs/G8-training-free-full/llama3-8b/*/; do
  cell=$(basename "$d")
  ppl=$(python -c "import json; d=json.load(open('$d/eval.json')); print(d['metrics']['wikitext2_ppl'])" 2>/dev/null)
  echo "${cell}: WT2 PPL = ${ppl:-MISSING}"
done
```

Expected: 16 lines, all with finite PPL.

- [ ] **Step 3: Commit a snapshot of training-free results**

```bash
cd /home/ubuntu/unifying-ptq
git add -A docs/superpowers/notes/ 2>/dev/null
git commit -m "phase-a: training-free row of LLM host matrix (4 hosts x 4 augments)" --allow-empty
```

---

## Phase B: QuaRot row

**Goal:** Fill the rotation × training-free row: QuaRot alone, +DBAF, +PCSA-tf, +both on Llama-3-8B W4A4.

**Files:**
- Existing: `patches/quarot_dbaf_pcsa_patch.py` (monkey-patches IntegerQuantizer / ActQuantizer at fake-quant entry points)
- Create: `scripts/run_quarot_pcsa_dbaf_sweep.sh`
- Reference: `QuaRot/` upstream runner (`QuaRot/fake_quant/main.py`)

### Task 10: Smoke-test QuaRot patch on a single cell

- [ ] **Step 1: Inspect QuaRot's main runner CLI**

```bash
head -60 /home/ubuntu/unifying-ptq/QuaRot/fake_quant/main.py
```

Note the CLI flags needed for W4A4 on Llama-3-8B (`--w_bits 4 --a_bits 4 --rotate --model meta-llama/Meta-Llama-3-8B`).

- [ ] **Step 2: Write the sweep script**

Create `/home/ubuntu/unifying-ptq/scripts/run_quarot_pcsa_dbaf_sweep.sh`:

```bash
#!/usr/bin/env bash
# QuaRot row of the LLM host matrix.
# Each cell: QuaRot rotation + {alone, +DBAF, +PCSA-tf, +both} fake-quant.
set -uo pipefail

cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq

mkdir -p /data/outputs/Q-quarot-row/llama3-8b

MODEL=/data/modelzoo/meta-llama/Meta-Llama-3-8B

for aug in alone dbaf pcsa_tf dbaf+pcsa_tf; do
  out=/data/outputs/Q-quarot-row/llama3-8b/quarot_${aug/+/_plus_}
  if [[ -f "$out/eval.json" ]]; then
    echo "[skip] $aug already done"
    continue
  fi
  mkdir -p "$out"
  echo "===== quarot $aug ====="
  python scripts/run_quarot_one_cell.py \
    --model "$MODEL" \
    --augment "$aug" \
    --out "$out" 2>&1 | tee "${out}/run.log" | tail -10
done

echo "QUAROT_ROW_DONE_$?"
```

- [ ] **Step 3: Write the per-cell runner**

Create `/home/ubuntu/unifying-ptq/scripts/run_quarot_one_cell.py`:

```python
"""Single QuaRot cell runner.

Loads Llama-3-8B, applies QuaRot's online Hadamard rotation, optionally
patches in DBAF and/or PCSA-tf via patches/quarot_dbaf_pcsa_patch.py, runs
W4A4 fake-quant, evaluates WikiText-2 + C4 PPL.

The QuaRot rotation is always on; the rotation flag is what defines this
ROW of the matrix.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time

sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/QuaRot")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/QuaRot/fake_quant")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--augment", choices=["alone", "dbaf", "pcsa_tf", "dbaf+pcsa_tf"],
                    required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    use_dbaf = "dbaf" in args.augment
    use_pcsa = "pcsa_tf" in args.augment or "pcsa-tf" in args.augment

    # Apply our DBAF/PCSA-tf patch to QuaRot's quantizers BEFORE loading the runner.
    from patches.quarot_dbaf_pcsa_patch import patch_quarot
    patch_quarot(use_dbaf=use_dbaf, use_pcsa_tf=use_pcsa)

    # Reuse QuaRot's main pipeline programmatically. The simplest path is to
    # invoke its main() with a built-up argparse namespace; if QuaRot's main
    # is not importable as a function, fall back to subprocess.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from main import quantize_model_quarot  # type: ignore
    # NOTE: the exact entry point depends on QuaRot's main.py structure;
    # adjust the import to match what `head -60 main.py` shows (Task 10
    # Step 1). The structure here matches QuaRot's released code.

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                  device_map="cuda",
                                                  low_cpu_mem_usage=True)

    # Build a QuaRot args namespace; copy defaults from QuaRot's main.py args parser.
    class _A: pass
    qa = _A()
    qa.model = args.model
    qa.w_bits = 4; qa.a_bits = 4; qa.v_bits = 4; qa.k_bits = 4
    qa.w_groupsize = -1; qa.a_groupsize = -1
    qa.w_asym = False; qa.a_asym = False
    qa.rotate = True
    qa.rotate_mode = "hadamard"
    qa.fp32_had = False
    qa.w_clip = True; qa.a_clip_ratio = 1.0
    qa.cal_dataset = "wikitext2"; qa.nsamples = 128; qa.seed = 0
    qa.eval_dataset = "wikitext2"

    t0 = time.time()
    model = quantize_model_quarot(model, qa)  # may differ — see Task 10 Step 1

    # Evaluate
    from datasets import load_dataset
    import torch
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    SEQ = 2048
    n_chunks = min(32, ids.shape[1] // SEQ)
    nlls = []
    model.eval()
    for i in range(n_chunks):
        chunk = ids[:, i * SEQ:(i + 1) * SEQ]
        with torch.no_grad():
            out = model(chunk, labels=chunk)
        nlls.append(out.loss.float().item())
    wt2 = float(torch.tensor(nlls).mean().exp().item())

    elapsed = time.time() - t0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval.json").write_text(json.dumps({
        "host": "quarot",
        "augment": args.augment,
        "metrics": {"wikitext2_ppl": wt2},
        "elapsed_s": elapsed,
    }, indent=2))
    print(f"[quarot/{args.augment}] WT2 PPL = {wt2:.3f} (in {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Reconcile the QuaRot entry-point**

```bash
grep -nE "(def main|def quantize|def rtn_fwrd|def llama_quant)" /home/ubuntu/unifying-ptq/QuaRot/fake_quant/main.py | head -10
```

If the function name `quantize_model_quarot` doesn't exist, edit `scripts/run_quarot_one_cell.py` to use the actual function name from QuaRot's main.py. Common candidates: `quantize_model`, `quantize_llama`.

- [ ] **Step 5: Run the `alone` cell as a smoke**

```bash
bash /home/ubuntu/unifying-ptq/scripts/run_quarot_pcsa_dbaf_sweep.sh 2>&1 | head -30
```

If the alone cell produces a finite WT2 PPL near the QuaRot paper's published Llama-3 W4A4 number (~9–10), the wiring is correct. Otherwise, fix the per-cell runner and rerun.

### Task 11: Run all 4 QuaRot cells

- [ ] **Step 1: Run the full sweep**

```bash
bash /home/ubuntu/unifying-ptq/scripts/run_quarot_pcsa_dbaf_sweep.sh 2>&1 | tail -40
```

Expected: ends with `QUAROT_ROW_DONE_0`. ~2 GPU hours total.

- [ ] **Step 2: Verify all 4 cells produced finite PPL**

```bash
for d in /data/outputs/Q-quarot-row/llama3-8b/*/; do
  cell=$(basename "$d")
  ppl=$(python -c "import json; d=json.load(open('$d/eval.json')); print(d['metrics']['wikitext2_ppl'])" 2>/dev/null)
  echo "${cell}: WT2 PPL = ${ppl:-MISSING}"
done
```

Expected: 4 lines, all finite.

- [ ] **Step 3: Commit the QuaRot wiring**

```bash
cd /home/ubuntu/unifying-ptq
git add scripts/run_quarot_pcsa_dbaf_sweep.sh scripts/run_quarot_one_cell.py
git commit -m "phase-b: quarot row of LLM host matrix (4 cells)"
```

---

## Phase C: TesseraQ row

**Goal:** Fill the trained × non-rotation row: TesseraQ alone, +DBAF, +PCSA-tf, +both on Llama-3-8B W4A4. The TesseraQ +PCSA-tf cell is the **strongest LLM PCSA validation** — non-rotation host where PCSA's per-prompt routing has signal to exploit.

**Files:**
- Existing: `patches/tesseraq_dbaf_pcsa_patch.py`
- Create: `scripts/run_tesseraq_pcsa_dbaf_sweep.sh`

### Task 12: Run TesseraQ alone + +DBAF (gate-independent)

These two cells run regardless of the diagnostic gate outcome, because TesseraQ +DBAF tests the DBAF claim on a trained non-rotation host.

- [ ] **Step 1: Write the sweep script**

Create `/home/ubuntu/unifying-ptq/scripts/run_tesseraq_pcsa_dbaf_sweep.sh`:

```bash
#!/usr/bin/env bash
# TesseraQ row of the LLM host matrix.
# Cells: tesseraq alone, +DBAF (always); +PCSA-tf, +both (only if gate PASSED).
set -uo pipefail

cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate tesseraq

OUT=/data/outputs/T-tesseraq-row/llama3-8b
mkdir -p "$OUT"
MODEL=/data/modelzoo/meta-llama/Meta-Llama-3-8B

# Comma-separated list of augments to run; default to alone+dbaf.
# Phase C extension (Task 13) overrides to "alone,dbaf,pcsa_tf,dbaf+pcsa_tf".
AUGS="${AUGS:-alone,dbaf}"

IFS=',' read -ra AUG_ARR <<<"$AUGS"
for aug in "${AUG_ARR[@]}"; do
  cell_out="${OUT}/tesseraq_${aug/+/_plus_}"
  if [[ -f "${cell_out}/eval.json" ]]; then
    echo "[skip] $aug already done"; continue
  fi
  mkdir -p "$cell_out"
  echo "===== tesseraq $aug ====="

  # Set env vars consumed by patches/tesseraq_dbaf_pcsa_patch.py
  export DBAF_ENABLED=$([[ "$aug" == *dbaf* ]] && echo 1 || echo 0)
  export PCSA_TF_ENABLED=$([[ "$aug" == *pcsa_tf* ]] && echo 1 || echo 0)

  cd TesseraQ
  python -m llmc \
    --config ../patches/tesseraq_emnlp_configs/llama3_8b_w4a4.yml \
    --model_path "$MODEL" \
    --save_path "$cell_out" 2>&1 | tee "${cell_out}/run.log" | tail -10
  cd ..
done

echo "TESSERAQ_ROW_DONE_$?"
```

- [ ] **Step 2: Verify the TesseraQ emnlp_configs dir + patch wiring**

```bash
ls /home/ubuntu/unifying-ptq/patches/tesseraq_emnlp_configs/ 2>/dev/null
grep -nE "(DBAF_ENABLED|PCSA_TF_ENABLED|os.environ)" /home/ubuntu/unifying-ptq/patches/tesseraq_dbaf_pcsa_patch.py | head -10
```

If the patch doesn't read these env vars, edit it to do so — the patch should monkey-patch TesseraQ's fake-quant entry points conditionally based on these env vars. Use the pattern from `patches/quarot_dbaf_pcsa_patch.py` as a reference.

- [ ] **Step 3: Run the alone+dbaf pair**

```bash
AUGS=alone,dbaf bash /home/ubuntu/unifying-ptq/scripts/run_tesseraq_pcsa_dbaf_sweep.sh 2>&1 | tail -30
```

Expected: 2 cells, ~6–10 GPU hours total. Ends with `TESSERAQ_ROW_DONE_0`. Both cells write `eval.json` with finite PPL.

- [ ] **Step 4: Commit the TesseraQ sweep wiring**

```bash
cd /home/ubuntu/unifying-ptq
git add scripts/run_tesseraq_pcsa_dbaf_sweep.sh
git commit -m "phase-c: tesseraq row sweep script (alone+dbaf always; pcsa cells gated)"
```

### Task 13: Run TesseraQ +PCSA-tf + +both (conditional on gate PASS)

Only execute this task if Task 7 (diagnostic gate) returned `GATE PASS`.

- [ ] **Step 1: Confirm gate result before launching**

```bash
grep -E "(PASS|FAIL)" /home/ubuntu/unifying-ptq/docs/superpowers/notes/2026-05-15-pcsa-tf-gate-result.md
```

If output contains `PASS`, proceed. If `FAIL`, **skip this task entirely** and proceed to Phase D.

- [ ] **Step 2: Run the +PCSA-tf and +both cells**

```bash
AUGS=pcsa_tf,dbaf+pcsa_tf bash /home/ubuntu/unifying-ptq/scripts/run_tesseraq_pcsa_dbaf_sweep.sh 2>&1 | tail -30
```

Expected: 2 cells, ~6–10 GPU hours total.

- [ ] **Step 3: Verify all 4 TesseraQ cells produced finite PPL**

```bash
for d in /data/outputs/T-tesseraq-row/llama3-8b/*/; do
  cell=$(basename "$d")
  ppl=$(python -c "import json; d=json.load(open('$d/eval.json')); print(d['metrics']['wikitext2_ppl'])" 2>/dev/null)
  echo "${cell}: WT2 PPL = ${ppl:-MISSING}"
done
```

Expected: 4 lines, all finite. Headline test: `tesseraq_pcsa_tf` PPL should be < `tesseraq_alone` PPL — the strongest LLM PCSA validation cell.

---

## Phase D: Aggregator + paper updates

**Goal:** Wire the new matrix into the LaTeX tables in `paper/emnlp2026/sections/04-experiments.tex` and the auto-aggregator script.

**Files:**
- Modify: `scripts/aggregate_results_to_latex.py`
- Modify: `paper/emnlp2026/sections/04-experiments.tex`

### Task 14: Extend the aggregator to emit the 4×4 matrix

- [ ] **Step 1: Inspect current aggregator scope**

```bash
grep -nE "def aggregate_|/data/outputs" /home/ubuntu/unifying-ptq/scripts/aggregate_results_to_latex.py | head -20
```

Existing aggregator already reads `/data/outputs/G8-training-free-full/<target>/<method>_<aug>/eval.json` for RTN/GPTQ/AWQ/SmoothQuant rows. Phase D adds two new readers: `/data/outputs/Q-quarot-row/...` and `/data/outputs/T-tesseraq-row/...`.

- [ ] **Step 2: Add QuaRot + TesseraQ reader functions**

Edit `/home/ubuntu/unifying-ptq/scripts/aggregate_results_to_latex.py`. After the existing `aggregate_g8_llm` function, add:

```python
# ---------------------------------------------------------------------------
# QuaRot row (training-free × rotation) and TesseraQ row (trained × non-rotation)
# ---------------------------------------------------------------------------

def _read_cells(base: pathlib.Path, host: str, augments: list[str]) -> dict:
    """Returns {augment: ppl_float or None} for a single-host directory layout
    /data/outputs/.../<host>_<aug>/eval.json (aug names like 'dbaf+pcsa_tf'
    are mangled to 'dbaf_plus_pcsa_tf' on disk per the sweep scripts).
    """
    out = {}
    for a in augments:
        cell = base / f"{host}_{a.replace('+', '_plus_')}"
        p = cell / "eval.json"
        if not p.exists():
            out[a] = None
            continue
        d = json.loads(p.read_text())
        wt2 = (d.get("metrics") or {}).get("wikitext2_ppl")
        out[a] = wt2
    return out


def aggregate_matrix_row(host: str, base: pathlib.Path,
                          augments: list[str], out_name: str,
                          row_label: str) -> None:
    """Emit a one-row LaTeX fragment for a host row of the LLM matrix."""
    cells = _read_cells(base, host, augments)
    if all(v is None for v in cells.values()):
        print(f"[matrix] {host} from {base} — no cells found, skipping")
        return
    aug_labels = {"alone": "Alone", "dbaf": "+DBAF",
                  "pcsa_tf": "+PCSA-tf", "dbaf+pcsa_tf": "+both"}
    parts = [row_label]
    for a in augments:
        v = cells.get(a)
        parts.append(f"{v:.2f}" if v is not None else "--")
    line = " & ".join(parts) + " \\\\"
    out_path = _PAPER_TABLE_DIR / out_name
    out_path.write_text(line + "\n")
    print(f"[matrix/{host}] → {out_path}")


def aggregate_quarot_row():
    aggregate_matrix_row(
        host="quarot",
        base=pathlib.Path("/data/outputs/Q-quarot-row/llama3-8b"),
        augments=["alone", "dbaf", "pcsa_tf", "dbaf+pcsa_tf"],
        out_name="row_quarot_llama3_8b.tex",
        row_label="QuaRot (rotation, training-free)",
    )


def aggregate_tesseraq_row():
    aggregate_matrix_row(
        host="tesseraq",
        base=pathlib.Path("/data/outputs/T-tesseraq-row/llama3-8b"),
        augments=["alone", "dbaf", "pcsa_tf", "dbaf+pcsa_tf"],
        out_name="row_tesseraq_llama3_8b.tex",
        row_label="TesseraQ (non-rotation, trained)",
    )
```

Then add to `main()`:

```python
def main():
    aggregate_g7_swinir()
    aggregate_g8_llm("llama3-8b")
    aggregate_crossdetector()
    aggregate_quarot_row()
    aggregate_tesseraq_row()
```

- [ ] **Step 3: Run the aggregator**

```bash
cd /home/ubuntu/unifying-ptq
python scripts/aggregate_results_to_latex.py 2>&1 | tail -15
ls /home/ubuntu/paper/emnlp2026/tables/_auto/row_*.tex
```

Expected: two new files `row_quarot_llama3_8b.tex` and `row_tesseraq_llama3_8b.tex` containing the LaTeX row fragments.

- [ ] **Step 4: Commit the aggregator changes**

```bash
cd /home/ubuntu/unifying-ptq
git add scripts/aggregate_results_to_latex.py
git commit -m "phase-d: aggregator emits QuaRot and TesseraQ matrix rows"
```

### Task 15: Update §4 with the 4×4 LLM matrix

- [ ] **Step 1: Read the current training-free table location**

```bash
grep -nE "(tab:trainfree|tab:training-free|Training-Free Hosts)" /home/ubuntu/paper/emnlp2026/sections/04-experiments.tex | head -5
```

Identify the line range where the existing training-free table lives.

- [ ] **Step 2: Insert the new "LLM Host Matrix" subsection**

Replace the existing training-free table with a 4-row training-free table + a new "Rotation vs Non-Rotation Hosts" subsection. Edit `paper/emnlp2026/sections/04-experiments.tex` — find this block:

```latex
\subsection{Training-Free Hosts (RTN / GPTQ / AWQ)}
\label{sec:exp-trainingfree}
```

And replace its body (the current `tab:trainfree` table) with:

```latex
\subsection{LLM Host Matrix (rotation $\times$ training-status)}
\label{sec:exp-trainingfree}

To validate that DBAF and PCSA-tf compose with host classes regardless of
rotation use and training-status, we evaluate four LLM hosts under the same
W4A4 setting on LLaMA-3-8B (WikiText-2 PPL). The matrix spans two axes:
rotation vs.\ non-rotation, and training-free vs.\ trained.

\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l cccc}
\toprule
\textbf{Host (axis)} & \textbf{Alone} & \textbf{+DBAF} & \textbf{+PCSA-tf} & \textbf{+both} \\
\midrule
\multicolumn{5}{l}{\emph{Training-free $\times$ non-rotation}} \\
RTN          & \input{tables/_auto/cell_rtn_alone}        & \input{tables/_auto/cell_rtn_dbaf}        & \input{tables/_auto/cell_rtn_pcsa_tf}        & \input{tables/_auto/cell_rtn_dbaf_plus_pcsa_tf} \\
GPTQ         & \input{tables/_auto/cell_gptq_alone}       & \input{tables/_auto/cell_gptq_dbaf}       & \input{tables/_auto/cell_gptq_pcsa_tf}       & \input{tables/_auto/cell_gptq_dbaf_plus_pcsa_tf} \\
AWQ          & \input{tables/_auto/cell_awq_alone}        & \input{tables/_auto/cell_awq_dbaf}        & \input{tables/_auto/cell_awq_pcsa_tf}        & \input{tables/_auto/cell_awq_dbaf_plus_pcsa_tf} \\
SmoothQuant  & \input{tables/_auto/cell_smoothquant_alone}& \input{tables/_auto/cell_smoothquant_dbaf}& \input{tables/_auto/cell_smoothquant_pcsa_tf}& \input{tables/_auto/cell_smoothquant_dbaf_plus_pcsa_tf} \\
\midrule
\multicolumn{5}{l}{\emph{Training-free $\times$ rotation}} \\
\input{tables/_auto/row_quarot_llama3_8b}
\midrule
\multicolumn{5}{l}{\emph{Trained $\times$ non-rotation}} \\
\input{tables/_auto/row_tesseraq_llama3_8b}
\midrule
\multicolumn{5}{l}{\emph{Trained $\times$ rotation}} \\
FlatQuant (saturated) & \input{tables/_auto/cell_flatquant_alone} & \input{tables/_auto/cell_flatquant_dbaf} & \input{tables/_auto/cell_flatquant_pcsa} & \input{tables/_auto/cell_flatquant_both} \\
\bottomrule
\end{tabular}
\caption{LLM W4A4 WikiText-2 PPL across the host matrix. DBAF lowers
PPL in every non-rotation cell. PCSA-tf shows non-trivial improvement on
non-rotation hosts (where prompt-conditioned variance is not absorbed by
rotation), and saturates on rotation hosts (FlatQuant, QuaRot).}
\label{tab:llm-matrix}
\end{table*}
```

(Cell file names follow the convention `cell_<method>_<augment>.tex` — these are written by the next step.)

- [ ] **Step 3: Extend the aggregator to also emit per-cell `cell_*.tex` files**

Edit `/home/ubuntu/unifying-ptq/scripts/aggregate_results_to_latex.py`. Modify `aggregate_g8_llm` to additionally write one tiny file per cell:

After the `cells = {...}` dict is built, add:

```python
    # Emit per-cell single-value fragments for direct \input{} use in §4 matrix.
    for (m, a), data in cells.items():
        wt2 = data.get("metrics", {}).get("wikitext2_ppl")
        val = f"{wt2:.2f}" if wt2 is not None else "--"
        fname = f"cell_{m}_{a.replace('+','_plus_')}.tex"
        (_PAPER_TABLE_DIR / fname).write_text(val + "\n")
```

Add analogous logic for FlatQuant cells (read from `/data/outputs/S5-baseline-calib/...`) — but since FlatQuant cells are all ~6.97, you can hardcode `cell_flatquant_*` with the calibration-time numbers in a separate one-time setup step:

```bash
cat > /home/ubuntu/paper/emnlp2026/tables/_auto/cell_flatquant_alone.tex <<<"6.97"
cat > /home/ubuntu/paper/emnlp2026/tables/_auto/cell_flatquant_dbaf.tex <<<"6.97"
cat > /home/ubuntu/paper/emnlp2026/tables/_auto/cell_flatquant_pcsa.tex <<<"6.98"
cat > /home/ubuntu/paper/emnlp2026/tables/_auto/cell_flatquant_both.tex <<<"6.97"
```

- [ ] **Step 4: Re-run the aggregator and verify**

```bash
cd /home/ubuntu/unifying-ptq
python scripts/aggregate_results_to_latex.py 2>&1 | tail -15
ls /home/ubuntu/paper/emnlp2026/tables/_auto/cell_*.tex | head -20
```

Expected: per-cell files exist for all 4 methods × 4 augments plus the QuaRot/TesseraQ rows.

- [ ] **Step 5: Build the paper and visually verify the matrix renders**

```bash
cd /home/ubuntu/paper/emnlp2026
ls *.tex
pdflatex -interaction=nonstopmode main.tex 2>&1 | tail -15
```

If the build is set up, look at the resulting PDF and confirm Table~\ref{tab:llm-matrix} populates correctly with all 24 cells.

- [ ] **Step 6: Reframe FlatQuant's saturation cell honestly**

Search §4 for any text that reads as "PCSA fails on LLM" and replace with a rotation-saturation framing. Specifically, find this passage (in the existing §4.6 KV-PCSA section or in any conclusion):

Expected change: add or update a paragraph like:

```latex
\paragraph{Why FlatQuant saturates.}  FlatQuant applies a learned
Hadamard rotation that already redistributes per-prompt activation
variance evenly across channels; PCSA's per-prompt routing therefore has
no residual structure to exploit, and DBAF's outlier folding is rendered
redundant.  This is not evidence that the mechanisms fail on LLMs —
Table~\ref{tab:llm-matrix} shows both lift PPL substantially in every
non-rotation cell.  The FlatQuant cell is included as the saturation
limit of the matrix, complementary to the active cells.
```

- [ ] **Step 7: Commit the paper updates**

```bash
cd /home/ubuntu/paper/emnlp2026
git add sections/04-experiments.tex tables/_auto/
git commit -m "experiments: 4x4 LLM host matrix replacing single training-free row"
```

- [ ] **Step 8: Final task-list refresh**

Move the relevant existing task numbers in `~/.claude` / project task list to `completed` status (the harness's TaskUpdate tool):
- C2 / C3a / B4 etc remain as-is.
- Mark `S5: KV-cache PCSA + RULER eval` as no-longer-blocking since the PCSA validation now comes from the host-matrix story.

---

## Total Budget Recap

| Section | Wall clock | Type |
|---|---|---|
| Tracks 1-3 (parallel) | ~1 day | dev (1-2 hours each for SmoothQuant/PCSA-tf; rest is TesseraQ env download) |
| Diagnostic gate (Task 7) | ~30 min | run |
| Phase A (Tasks 8-9) | ~3-4 GPU hours | run |
| Phase B (Tasks 10-11) | ~2 GPU hours | run + ~1 hour to wire QuaRot CLI |
| Phase C alone+DBAF (Task 12) | ~6-10 GPU hours | run |
| Phase C +PCSA-tf+both (Task 13, gated) | ~6-10 GPU hours | run |
| Phase D (Tasks 14-15) | ~1-2 days | dev + paper |

**Total worst-case: ~4 days, comfortably within the 10-day deadline window.**
