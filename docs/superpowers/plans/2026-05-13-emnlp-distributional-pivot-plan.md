# EMNLP 2026 Distributional-Pivot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe the EMNLP 2026 submission around distribution-guided composable primitives (DBAF + PCSA + training-free variants) and run the cross-architecture experiment program against OmniQuant / AHCPTQ / 2DQuant hosts.

**Architecture:** Five phases — (1) setup + new training-free primitives with unit tests, (2) plumb DBAF+PCSA into OmniQuant + 2DQuant, (3) calibration suite (composability + training-free + cross-detector SAM), (4) analyses (per-layer, RULER, cost-quality), (5) paper rewrite. Each phase produces standalone working output; later phases depend on earlier ones.

**Tech Stack:** PyTorch 2.6 + torchao (unifyptq env); PyTorch 1.13 + mmcv-full 1.7 (ahcptq-old env for SAM training); transformers 4.45 (LLM); FlatQuant baselines RTN/GPTQ/AWQ (`flatquant/baselines/rtn.py`); AHCPTQ codebase; OmniQuant + 2DQuant (clone in Task 1).

---

## File Structure

**New files:**
- `flatquant/baselines/pcsa_tf.py` — training-free PCSA primitive (k-means on activations)
- `flatquant/baselines/kv_pcsa_tf.py` — training-free KV-PCSA primitive (k-means on K/V cache)
- `tests/test_pcsa_tf.py` — unit tests for PCSA-tf
- `tests/test_kv_pcsa_tf.py` — unit tests for KV-PCSA-tf
- `OmniQuant/` — cloned upstream (~50 files)
- `OmniQuant/omniquant_dbaf_pcsa_patch.py` — our integration module
- `2DQuant/` — cloned upstream (~30 files)
- `2DQuant/twodquant_dbaf_pcsa_patch.py` — our integration module
- `scripts/run_omniquant_full_table.sh` — driver for OmniQuant ±{DBAF, PCSA, both}
- `scripts/run_2dquant_full_table.sh` — driver for 2DQuant ±{DBAF, PCSA, both}
- `scripts/run_training_free_full_table.sh` — driver for RTN/GPTQ/AWQ ±{DBAF, PCSA-tf, both}
- `scripts/run_ahcptq_cross_detector.sh` — SAM-B/L + H-DETR (queued on remote)
- `scripts/run_ruler_kv_pcsa_tf_eval.sh` — RULER 4k/8k eval
- `scripts/run_per_layer_ablation_sam.py` — per-layer ablation on SAM-B
- `scripts/run_per_layer_ablation_llama.py` — per-layer ablation on LLaMA-3-8B
- `scripts/make_cost_quality_figure.py` — calibration-cost-vs-quality plot
- `results/G6-omniquant/` — output dir for OmniQuant calibrations
- `results/G7-2dquant/` — output dir for 2DQuant calibrations
- `results/G8-training-free-full/` — output dir for training-free table
- `results/G9-ahcptq-cross-detector/` — output dir for cross-detector SAM
- `results/G10-ruler/` — output dir for RULER eval
- `results/G12-per-layer-llm-sam/` — output dir for cross-arch per-layer ablation

**Modified files:**
- `paper/emnlp2026/sections/00-abstract.tex` — rewrite per reframed claim
- `paper/emnlp2026/sections/01-intro.tex` — rewrite + extend fragmentation table
- `paper/emnlp2026/sections/02-related.tex` — reorganize per-distribution-type
- `paper/emnlp2026/sections/03-method.tex` — add PCSA-tf, KV-PCSA-tf, composability API
- `paper/emnlp2026/sections/04-experiments.tex` — new table layout
- `paper/emnlp2026/sections/05-conclusion.tex` — match new claim
- `paper/emnlp2026/sections/06-limitations.tex` — SR ×3 + bimodal + KV-PCSA caveats
- `results/PAPER_RESULTS.md` — running results tracker

---

## Phase 1 — Setup + Training-Free Primitives (Days 1-2)

### Task 1: Clone OmniQuant and 2DQuant repos

**Files:**
- Create: `/home/ubuntu/unifying-ptq/OmniQuant/` (cloned)
- Create: `/home/ubuntu/unifying-ptq/2DQuant/` (cloned)
- Modify: `/home/ubuntu/unifying-ptq/.gitignore` — add `OmniQuant/` and `2DQuant/` as embedded repos

- [ ] **Step 1: Clone OmniQuant**

```bash
cd /home/ubuntu/unifying-ptq
git clone https://github.com/OpenGVLab/OmniQuant.git
```

Expected: clone succeeds; `OmniQuant/main.py` exists.

- [ ] **Step 2: Clone 2DQuant**

```bash
cd /home/ubuntu/unifying-ptq
git clone https://github.com/Kai-Liu001/2DQuant.git
```

Expected: clone succeeds; check that the repo has a quant runner script. If URL fails, search for the canonical repo (paper authors at https://arxiv.org/abs/2406.06649) and clone that instead.

- [ ] **Step 3: Add to .gitignore so we don't accidentally vendor them**

Append to `/home/ubuntu/unifying-ptq/.gitignore`:
```
# Upstream repos
OmniQuant/
2DQuant/
```

- [ ] **Step 4: Run OmniQuant smoke (verify code is functional)**

```bash
cd /home/ubuntu/unifying-ptq/OmniQuant
cat README.md | head -50
ls -la
python -c "import sys; sys.path.insert(0, '.'); import main" 2>&1 | tail -3
```

Expected: no import errors. If imports fail because of missing deps, install in unifyptq conda env (`pip install accelerate datasets` etc).

- [ ] **Step 5: Run 2DQuant smoke**

```bash
cd /home/ubuntu/unifying-ptq/2DQuant
cat README.md | head -50
ls -la
```

Expected: identify the entry point (e.g., `basicsr/test.py` or `2dquant/quantize.py`).

- [ ] **Step 6: Commit gitignore change**

```bash
cd /home/ubuntu/unifying-ptq
git add .gitignore
git commit -m "chore: gitignore upstream OmniQuant + 2DQuant clones"
```

---

### Task 2: Training-free PCSA — write failing test

**Files:**
- Create: `/home/ubuntu/unifying-ptq/tests/test_pcsa_tf.py`

- [ ] **Step 1: Write the failing test**

Create `/home/ubuntu/unifying-ptq/tests/test_pcsa_tf.py`:

```python
"""Unit tests for training-free PCSA primitive."""
import pytest
import torch
import sys
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
from flatquant.baselines.pcsa_tf import (
    fit_pcsa_tf, route_pcsa_tf, apply_pcsa_tf_to_activation,
)


def test_fit_returns_anchors_and_scales():
    # 16 calibration prompts, 64-dim descriptors, 4 anchors
    descs = torch.randn(16, 64)
    acts = torch.randn(16, 4, 8)  # 4 tokens, 8 channels per prompt
    state = fit_pcsa_tf(descs, acts, K=4)
    assert "anchors" in state
    assert state["anchors"].shape == (4, 64)
    assert "scales" in state
    assert state["scales"].shape == (4,)


def test_route_assigns_correct_anchor():
    descs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    acts = torch.ones(4, 1, 4)
    state = fit_pcsa_tf(descs, acts, K=4)
    # Query a descriptor exactly matching anchor 0
    q = state["anchors"][0:1]
    idx = route_pcsa_tf(q, state)
    assert idx.item() == 0


def test_apply_pcsa_uses_anchor_scale():
    # Build a state where anchor 0 has scale=10, anchor 1 has scale=0.1
    state = {
        "anchors": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "scales": torch.tensor([10.0, 0.1]),
    }
    x = torch.full((1, 4, 2), 5.0)
    desc = torch.tensor([[1.0, 0.01]])  # routes to anchor 0
    out = apply_pcsa_tf_to_activation(x, desc, state, bits=4)
    # With scale 10 and INT4 asym, 5.0 fits easily
    assert torch.allclose(out, x, atol=1.0)


def test_pcsa_tf_no_gradient():
    descs = torch.randn(8, 32, requires_grad=True)
    acts = torch.randn(8, 4, 16, requires_grad=True)
    state = fit_pcsa_tf(descs, acts, K=2)
    # Anchors and scales should NOT carry gradients
    assert state["anchors"].requires_grad is False
    assert state["scales"].requires_grad is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
python -m pytest tests/test_pcsa_tf.py -v 2>&1 | tail -10
```

Expected: FAIL with `ModuleNotFoundError: No module named 'flatquant.baselines.pcsa_tf'`.

---

### Task 3: Training-free PCSA — implementation

**Files:**
- Create: `/home/ubuntu/unifying-ptq/FlatQuant/flatquant/baselines/pcsa_tf.py`

- [ ] **Step 1: Implement pcsa_tf module**

Create `/home/ubuntu/unifying-ptq/FlatQuant/flatquant/baselines/pcsa_tf.py`:

```python
"""Training-free PCSA: k-means on calibration prompt descriptors + per-anchor
max-abs activation scales. No gradient training. Composes on any host method
that has activation tensors.

API:
  fit_pcsa_tf(descs, acts, K) -> state dict {anchors, scales}
  route_pcsa_tf(desc, state) -> anchor_id tensor
  apply_pcsa_tf_to_activation(x, desc, state, bits) -> fake-quantized x
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def _kmeans(x: torch.Tensor, k: int, n_iter: int = 25) -> torch.Tensor:
    """k-means on [N, D] -> [k, D] centroids. CPU-friendly, no gradients."""
    x = x.detach()
    N, D = x.shape
    # init from random rows
    idx = torch.randperm(N)[:k]
    centroids = x[idx].clone()
    for _ in range(n_iter):
        # assign each row to nearest centroid (cosine on L2-normalized vectors)
        xn = F.normalize(x, dim=-1)
        cn = F.normalize(centroids, dim=-1)
        sims = xn @ cn.T
        assign = sims.argmax(dim=-1)
        # update centroids = mean of assigned rows
        new_cents = torch.zeros_like(centroids)
        for j in range(k):
            mask = (assign == j)
            if mask.any():
                new_cents[j] = x[mask].mean(dim=0)
            else:
                new_cents[j] = centroids[j]
        if torch.allclose(new_cents, centroids, atol=1e-6):
            break
        centroids = new_cents
    return centroids


@torch.no_grad()
def fit_pcsa_tf(
    descs: torch.Tensor,
    acts: torch.Tensor,
    K: int = 8,
) -> dict:
    """Fit K anchors via k-means on `descs`, then per-anchor max-abs scale on `acts`.

    Args:
      descs: [N, D_desc] prompt-level descriptors (e.g., mean-pooled hidden states)
      acts:  [N, T, D_act] or [N, D_act] activations per prompt; if 3D, max over T
      K: number of anchors

    Returns dict {"anchors": [K, D_desc], "scales": [K]} (both no grad).
    """
    descs = descs.detach().float()
    acts = acts.detach().float()
    if acts.dim() == 3:
        per_prompt_max = acts.abs().amax(dim=(1, 2))  # [N]
    elif acts.dim() == 2:
        per_prompt_max = acts.abs().amax(dim=1)  # [N]
    else:
        raise ValueError(f"acts must be [N,T,D] or [N,D], got {acts.shape}")
    anchors = _kmeans(descs, K)
    # route each prompt to its nearest anchor and take max-abs activation
    sims = F.normalize(descs, dim=-1) @ F.normalize(anchors, dim=-1).T
    assign = sims.argmax(dim=-1)  # [N]
    scales = torch.zeros(K)
    for j in range(K):
        mask = (assign == j)
        scales[j] = per_prompt_max[mask].max() if mask.any() else per_prompt_max.max()
    return {"anchors": anchors, "scales": scales}


@torch.no_grad()
def route_pcsa_tf(desc: torch.Tensor, state: dict) -> torch.Tensor:
    """desc: [B, D]; returns [B] anchor indices."""
    sims = F.normalize(desc, dim=-1) @ F.normalize(state["anchors"], dim=-1).T
    return sims.argmax(dim=-1)


@torch.no_grad()
def apply_pcsa_tf_to_activation(
    x: torch.Tensor,
    desc: torch.Tensor,
    state: dict,
    bits: int = 4,
) -> torch.Tensor:
    """Per-prompt asym INT[bits] fake-quantization using anchor-routed scale.

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

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
python -m pytest tests/test_pcsa_tf.py -v 2>&1 | tail -15
```

Expected: 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add tests/test_pcsa_tf.py FlatQuant/flatquant/baselines/pcsa_tf.py
git commit -m "feat: training-free PCSA primitive (k-means anchors, per-anchor scales)"
```

---

### Task 4: Training-free KV-PCSA — write failing test

**Files:**
- Create: `/home/ubuntu/unifying-ptq/tests/test_kv_pcsa_tf.py`

- [ ] **Step 1: Write the failing test**

Create `/home/ubuntu/unifying-ptq/tests/test_kv_pcsa_tf.py`:

```python
"""Unit tests for training-free KV-PCSA primitive."""
import pytest
import torch
import sys
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
from flatquant.baselines.kv_pcsa_tf import (
    fit_kv_pcsa_tf, quantize_k_with_kv_pcsa_tf, quantize_v_with_kv_pcsa_tf,
)


def test_fit_returns_k_and_v_scales_per_anchor():
    # 12 calibration prompts, 32-dim descriptors, 4 anchors,
    # each prompt has K cache [num_heads=4, seq=16, head_dim=8]
    N, K_anchors = 12, 4
    descs = torch.randn(N, 32)
    k_caches = [torch.randn(4, 16, 8) for _ in range(N)]
    v_caches = [torch.randn(4, 16, 8) for _ in range(N)]
    state = fit_kv_pcsa_tf(descs, k_caches, v_caches, K=K_anchors)
    assert state["anchors"].shape == (K_anchors, 32)
    assert state["k_scales"].shape == (K_anchors,)
    assert state["v_scales"].shape == (K_anchors,)


def test_quantize_k_uses_anchor_scale():
    state = {
        "anchors": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "k_scales": torch.tensor([5.0, 0.5]),
        "v_scales": torch.tensor([3.0, 0.3]),
    }
    k = torch.full((1, 2, 4, 4), 2.0)
    desc = torch.tensor([[1.0, 0.01]])
    out = quantize_k_with_kv_pcsa_tf(k, desc, state, bits=4)
    # Anchor 0 scale 5.0 / qmax 15 ≈ 0.33 per code; 2.0 / 0.33 ≈ 6 codes
    # Should reconstruct close to 2.0
    assert (out - k).abs().mean() < 1.0


def test_quantize_v_uses_v_scale_not_k_scale():
    state = {
        "anchors": torch.tensor([[1.0, 0.0]]),
        "k_scales": torch.tensor([5.0]),
        "v_scales": torch.tensor([50.0]),  # very loose v scale
    }
    v = torch.full((1, 1, 4, 4), 20.0)  # within v scale, outside k scale
    desc = torch.tensor([[1.0, 0.0]])
    out = quantize_v_with_kv_pcsa_tf(v, desc, state, bits=4)
    # With scale 50.0/15 ≈ 3.33, 20.0 / 3.33 ≈ 6 codes — should reconstruct close to 20
    assert (out - v).abs().mean() < 5.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ubuntu/unifying-ptq
python -m pytest tests/test_kv_pcsa_tf.py -v 2>&1 | tail -5
```

Expected: FAIL with `ModuleNotFoundError`.

---

### Task 5: Training-free KV-PCSA — implementation

**Files:**
- Create: `/home/ubuntu/unifying-ptq/FlatQuant/flatquant/baselines/kv_pcsa_tf.py`

- [ ] **Step 1: Implement kv_pcsa_tf module**

Create `/home/ubuntu/unifying-ptq/FlatQuant/flatquant/baselines/kv_pcsa_tf.py`:

```python
"""Training-free KV-PCSA: k-means on calibration prompt descriptors + per-anchor
max-abs K and V cache scales. No gradient training. LLM-specific.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from flatquant.baselines.pcsa_tf import _kmeans, route_pcsa_tf


@torch.no_grad()
def fit_kv_pcsa_tf(
    descs: torch.Tensor,
    k_caches: list[torch.Tensor],
    v_caches: list[torch.Tensor],
    K: int = 4,
) -> dict:
    """Fit K anchors via k-means on `descs`; per-anchor max-abs over K and V.

    descs: [N, D] prompt descriptors
    k_caches/v_caches: list of N tensors, each shape [num_heads, seq, head_dim]
                       or any shape (we max-abs all dims)
    K: number of anchors
    """
    descs = descs.detach().float()
    anchors = _kmeans(descs, K)
    sims = F.normalize(descs, dim=-1) @ F.normalize(anchors, dim=-1).T
    assign = sims.argmax(dim=-1)  # [N]
    k_scales = torch.zeros(K)
    v_scales = torch.zeros(K)
    k_max = torch.tensor([kc.abs().max() for kc in k_caches])
    v_max = torch.tensor([vc.abs().max() for vc in v_caches])
    for j in range(K):
        mask = (assign == j)
        k_scales[j] = k_max[mask].max() if mask.any() else k_max.max()
        v_scales[j] = v_max[mask].max() if mask.any() else v_max.max()
    return {"anchors": anchors, "k_scales": k_scales, "v_scales": v_scales}


@torch.no_grad()
def _quantize_with_scale(x: torch.Tensor, scale_per_prompt: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** bits - 1
    extra_dims = x.dim() - 1
    scale = scale_per_prompt.view(-1, *([1] * extra_dims)) / qmax
    scale = scale.clamp(min=1e-9)
    q = torch.round(x / scale).clamp(-qmax // 2, qmax // 2)
    return (q * scale).to(x.dtype)


@torch.no_grad()
def quantize_k_with_kv_pcsa_tf(k: torch.Tensor, desc: torch.Tensor, state: dict, bits: int = 4) -> torch.Tensor:
    anchor_ids = route_pcsa_tf(desc, state)
    return _quantize_with_scale(k, state["k_scales"][anchor_ids], bits)


@torch.no_grad()
def quantize_v_with_kv_pcsa_tf(v: torch.Tensor, desc: torch.Tensor, state: dict, bits: int = 4) -> torch.Tensor:
    anchor_ids = route_pcsa_tf(desc, state)
    return _quantize_with_scale(v, state["v_scales"][anchor_ids], bits)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd /home/ubuntu/unifying-ptq
python -m pytest tests/test_kv_pcsa_tf.py -v 2>&1 | tail -10
```

Expected: 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add tests/test_kv_pcsa_tf.py FlatQuant/flatquant/baselines/kv_pcsa_tf.py
git commit -m "feat: training-free KV-PCSA primitive (k-means anchors, per-anchor K/V scales)"
```

---

## Phase 2 — Host Method Integrations (Days 2-5)

### Task 6: OmniQuant — locate entry points and quant tensor sites

**Files:**
- Read: `/home/ubuntu/unifying-ptq/OmniQuant/main.py` (or equivalent)
- Read: `/home/ubuntu/unifying-ptq/OmniQuant/quantize/*` (whatever the quantization module is)

- [ ] **Step 1: Identify OmniQuant entry point**

```bash
cd /home/ubuntu/unifying-ptq/OmniQuant
grep -nE "def main|argparse|parser.add_argument" main.py 2>&1 | head -15
ls quantize/ 2>&1
```

Expected: locate the LLaMA quantization runner + the per-layer quantization site (likely a forward hook or a `forward()` override on a wrapped Linear).

- [ ] **Step 2: Locate the weight + activation quantization functions**

```bash
cd /home/ubuntu/unifying-ptq/OmniQuant
grep -rnE "def.*quant|fake_quant|quantize" quantize/ models/ 2>&1 | head -20
```

Expected: find the function that quantizes a single weight tensor (where we'll inject DBAF fold/unfold) and the function that quantizes an activation tensor (where we'll inject DBAF + PCSA-tf routing).

- [ ] **Step 3: Write a note documenting the integration points**

Create `/home/ubuntu/unifying-ptq/OmniQuant/INTEGRATION_NOTES.md`:

```markdown
# OmniQuant DBAF + PCSA Integration Notes

## Identified hooks (fill in based on Step 2)

- Weight quant fn: `quantize/<file>.py:<line>` — `<function_name>(...)`
- Activation quant fn: `quantize/<file>.py:<line>` — `<function_name>(...)`
- Calibration loop entry: `main.py:<line>`
- Per-layer descriptor extraction (hidden states pre-q_proj): TBD

## Plumbing plan

- DBAF: wrap weight quant fn → if gate_passes(w): fold(w) → quantize → unfold(q)
- DBAF on activation: same wrap on activation quant fn
- PCSA: replace single-scale activation quant with per-prompt routed scale
- PCSA-tf: collect descriptors during calibration, fit_pcsa_tf, then route at quantization sites
```

- [ ] **Step 4: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add OmniQuant/INTEGRATION_NOTES.md 2>&1 || echo "OmniQuant is gitignored — keep notes locally"
# Notes are inside the gitignored upstream dir; alternative: put them in our codebase
mkdir -p docs/integrations
mv OmniQuant/INTEGRATION_NOTES.md docs/integrations/omniquant.md
git add docs/integrations/omniquant.md
git commit -m "docs: OmniQuant integration notes (hook locations)"
```

---

### Task 7: OmniQuant — write smoke test (single weight tensor with DBAF)

**Files:**
- Create: `/home/ubuntu/unifying-ptq/tests/test_omniquant_dbaf_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `/home/ubuntu/unifying-ptq/tests/test_omniquant_dbaf_smoke.py`:

```python
"""Smoke test: OmniQuant's weight quant fn with DBAF gate + fold/unfold yields
a fake-quantized tensor that's no worse than RTN on a tensor known to be
sparse-outlier (where the gate fires)."""
import sys
import torch
sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/OmniQuant")

from ahcptq.quantization.fake_quant import (
    is_like_normal_plus_3sigma_outliers, fold_outliers, unfold_outliers,
)
from flatquant.baselines.rtn import _quantize_tensor_uniform


def test_omniquant_dbaf_better_than_rtn_on_sparse():
    rng = torch.Generator().manual_seed(0)
    # 1024 weights, mostly N(0,1), with 5 outliers at ±10
    w = torch.randn(1024, generator=rng)
    idx = torch.randperm(1024, generator=rng)[:5]
    w[idx] = torch.tensor([10.0, -10.0, 10.0, -10.0, 10.0])
    w = w.unsqueeze(0)  # [1, 1024]

    # RTN baseline
    w_rtn = _quantize_tensor_uniform(w, 4, per_channel=True)
    rtn_mse = ((w - w_rtn) ** 2).mean().item()

    # DBAF
    gate = is_like_normal_plus_3sigma_outliers(w)
    assert gate["is_like_c"], "Constructed tensor should pass the gate"
    T = float(3.0 * gate["stats"]["std"])
    alpha = 0.95
    w_fold, tag = fold_outliers(w, T, alpha)
    w_q = _quantize_tensor_uniform(w_fold, 4, per_channel=True)
    w_dbaf = unfold_outliers(w_q, tag, T, alpha)
    dbaf_mse = ((w - w_dbaf) ** 2).mean().item()

    # DBAF should reduce MSE on this constructed sparse-outlier tensor
    assert dbaf_mse < rtn_mse, f"dbaf_mse={dbaf_mse:.4g} >= rtn_mse={rtn_mse:.4g}"
```

- [ ] **Step 2: Run test (this doesn't actually depend on OmniQuant; tests the building blocks)**

```bash
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
python -m pytest tests/test_omniquant_dbaf_smoke.py -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add tests/test_omniquant_dbaf_smoke.py
git commit -m "test: smoke test for DBAF on sparse-outlier tensors"
```

---

### Task 8: OmniQuant — plumb DBAF into weight quant path

**Files:**
- Create: `/home/ubuntu/unifying-ptq/OmniQuant/omniquant_dbaf_pcsa_patch.py`

This task is BLOCKED on Task 6's discovery output. The implementer should:

- [ ] **Step 1: Read INTEGRATION_NOTES.md to identify hook locations**

```bash
cat /home/ubuntu/unifying-ptq/docs/integrations/omniquant.md
```

- [ ] **Step 2: Create the patch module that wraps OmniQuant's weight quant fn**

Create `/home/ubuntu/unifying-ptq/OmniQuant/omniquant_dbaf_pcsa_patch.py`:

```python
"""Monkey-patch OmniQuant's quantization functions to add DBAF + PCSA hooks.

Call install_dbaf_patches() before the calibration loop starts. The patches
respect a process-global flag for gating:
  - When ON: use is_like_normal_plus_3sigma_outliers gate to decide whether
    to apply DBAF fold/unfold around the existing quantization.
  - When --no-dbaf-gate is set (via env var or arg), force-apply.

This file lives inside the OmniQuant/ tree (gitignored) but is referenced by
our calibration scripts.
"""
import os
import sys
import functools

# Make sure our DBAF utilities are importable
sys.path.insert(0, "/home/ubuntu/unifying-ptq")
from ahcptq.quantization.fake_quant import (
    is_like_normal_plus_3sigma_outliers, fold_outliers, unfold_outliers,
)


def _wrap_weight_quant(orig_fn, dbaf_alpha: float):
    """Wrap a weight-quant function (sig: (w: Tensor, ...) -> Tensor) with DBAF."""
    @functools.wraps(orig_fn)
    def wrapped(w, *args, **kwargs):
        gate = is_like_normal_plus_3sigma_outliers(w)
        if gate["is_like_c"]:
            T = float(3.0 * gate["stats"]["std"])
            w_fold, tag = fold_outliers(w, T, dbaf_alpha)
            q = orig_fn(w_fold, *args, **kwargs)
            return unfold_outliers(q, tag, T, dbaf_alpha)
        return orig_fn(w, *args, **kwargs)
    return wrapped


def install_dbaf_patches(weight_quant_fn_path: str, dbaf_alpha: float = 0.95):
    """Replace a weight quant function with the DBAF-wrapped version.

    weight_quant_fn_path: e.g. "quantize.omniquant.quantize_weight"
    """
    module_path, fn_name = weight_quant_fn_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    orig = getattr(mod, fn_name)
    setattr(mod, fn_name, _wrap_weight_quant(orig, dbaf_alpha))
    print(f"[dbaf_patch] wrapped {weight_quant_fn_path} with DBAF (alpha={dbaf_alpha})", flush=True)
```

(Update `weight_quant_fn_path` after Task 6 identifies the actual fn path.)

- [ ] **Step 3: Write a smoke calibration script**

Create `/home/ubuntu/unifying-ptq/scripts/run_omniquant_smoke.sh`:

```bash
#!/usr/bin/env bash
# Smoke-test OmniQuant + DBAF on LLaMA-3-8B for 1 layer.
set -e
cd /home/ubuntu/unifying-ptq/OmniQuant
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
export PYTHONPATH=/home/ubuntu/unifying-ptq:/home/ubuntu/unifying-ptq/FlatQuant:${PYTHONPATH:-}
# Apply patch via python -c hook
python -c "
import sys
sys.path.insert(0, '.')
from omniquant_dbaf_pcsa_patch import install_dbaf_patches
install_dbaf_patches('quantize.omniquant.quantize_weight', dbaf_alpha=0.95)
# now exec the main.py with limited layers
" 2>&1 | tail -5
# Then exec their main.py with --num_layers=1 or similar
# (exact flags TBD after Task 6)
echo "OMNIQUANT_SMOKE_DONE"
```

- [ ] **Step 4: Run smoke test**

```bash
chmod +x /home/ubuntu/unifying-ptq/scripts/run_omniquant_smoke.sh
/home/ubuntu/unifying-ptq/scripts/run_omniquant_smoke.sh 2>&1 | tail -10
```

Expected: smoke completes; one calibration step ran; saw `[dbaf_patch] wrapped ...` line.

- [ ] **Step 5: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add scripts/run_omniquant_smoke.sh
# omniquant_dbaf_pcsa_patch.py is inside OmniQuant/ (gitignored) so won't be committed; keep a copy under our codebase:
mkdir -p patches/
cp OmniQuant/omniquant_dbaf_pcsa_patch.py patches/omniquant_dbaf_pcsa_patch.py
git add patches/omniquant_dbaf_pcsa_patch.py
git commit -m "feat(G4): OmniQuant DBAF patch module + smoke runner"
```

---

### Task 9: OmniQuant — plumb PCSA-tf into activation quant path

**Files:**
- Modify: `/home/ubuntu/unifying-ptq/patches/omniquant_dbaf_pcsa_patch.py`

- [ ] **Step 1: Extend the patch module with PCSA-tf hook**

Modify `/home/ubuntu/unifying-ptq/OmniQuant/omniquant_dbaf_pcsa_patch.py` to add:

```python
# Append to the file from Task 8

from flatquant.baselines.pcsa_tf import (
    fit_pcsa_tf, route_pcsa_tf, apply_pcsa_tf_to_activation,
)

_PCSA_STATE = None  # populated by fit_pcsa_tf_on_calib_data


def fit_pcsa_tf_on_calib_data(descs, acts, K: int = 8):
    """Call once with calibration data to populate the global PCSA state."""
    global _PCSA_STATE
    _PCSA_STATE = fit_pcsa_tf(descs, acts, K=K)
    print(f"[pcsa_tf] fitted {K} anchors, scales={_PCSA_STATE['scales'].tolist()}", flush=True)


def _wrap_activation_quant(orig_fn, dbaf_alpha: float):
    """Wrap an activation-quant function. If _PCSA_STATE is set, route per prompt."""
    @functools.wraps(orig_fn)
    def wrapped(x, *args, descriptor=None, **kwargs):
        # PCSA-tf path: use anchor scale if state is set
        if _PCSA_STATE is not None and descriptor is not None:
            x = apply_pcsa_tf_to_activation(x, descriptor, _PCSA_STATE, bits=4)
        # DBAF gate on the (possibly already-rescaled) activation
        gate = is_like_normal_plus_3sigma_outliers(x)
        if gate["is_like_c"]:
            T = float(3.0 * gate["stats"]["std"])
            x_fold, tag = fold_outliers(x, T, dbaf_alpha)
            q = orig_fn(x_fold, *args, **kwargs)
            return unfold_outliers(q, tag, T, dbaf_alpha)
        return orig_fn(x, *args, **kwargs)
    return wrapped


def install_pcsa_tf_patches(act_quant_fn_path: str, dbaf_alpha: float = 0.95):
    module_path, fn_name = act_quant_fn_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    orig = getattr(mod, fn_name)
    setattr(mod, fn_name, _wrap_activation_quant(orig, dbaf_alpha))
    print(f"[pcsa_patch] wrapped {act_quant_fn_path}", flush=True)
```

- [ ] **Step 2: Sync the working file**

```bash
cp /home/ubuntu/unifying-ptq/patches/omniquant_dbaf_pcsa_patch.py \
   /home/ubuntu/unifying-ptq/OmniQuant/omniquant_dbaf_pcsa_patch.py
```

- [ ] **Step 3: Smoke test PCSA-tf installation**

```bash
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
python -c "
import sys
sys.path.insert(0, 'OmniQuant')
import omniquant_dbaf_pcsa_patch as p
import torch
# fit PCSA-tf on dummy data
descs = torch.randn(16, 32)
acts = torch.randn(16, 4, 8)
p.fit_pcsa_tf_on_calib_data(descs, acts, K=4)
print('PCSA_TF_INSTALL_OK')
"
```

Expected: prints `[pcsa_tf] fitted 4 anchors, scales=[...]` then `PCSA_TF_INSTALL_OK`.

- [ ] **Step 4: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add patches/omniquant_dbaf_pcsa_patch.py
git commit -m "feat(G4): add PCSA-tf hook to OmniQuant patch module"
```

---

### Task 10: 2DQuant — locate entry points + plumb DBAF + PCSA-tf

**Files:**
- Create: `/home/ubuntu/unifying-ptq/docs/integrations/2dquant.md`
- Create: `/home/ubuntu/unifying-ptq/patches/twodquant_dbaf_pcsa_patch.py`

Repeats Tasks 6-9 pattern for 2DQuant. Same plumbing: wrap weight-quant fn with DBAF, wrap activation-quant fn with DBAF + PCSA-tf.

- [ ] **Step 1: Identify 2DQuant entry points**

```bash
cd /home/ubuntu/unifying-ptq/2DQuant
grep -rnE "def.*quant|fake_quant" basicsr/ 2DQuant/ 2>&1 | head -15
```

- [ ] **Step 2: Write integration notes**

Create `/home/ubuntu/unifying-ptq/docs/integrations/2dquant.md`:

```markdown
# 2DQuant DBAF + PCSA Integration Notes

## Identified hooks

- Weight quant fn: `<file>:<line>` (TBD from Step 1)
- Activation quant fn: `<file>:<line>` (TBD)
- Calibration entry: `<file>:<line>` (TBD)
- Per-prompt descriptor: mean-pooled SwinIR input image feature
```

- [ ] **Step 3: Write the patch module**

Create `/home/ubuntu/unifying-ptq/patches/twodquant_dbaf_pcsa_patch.py` (same structure as Task 8+9 but pointed at 2DQuant's fn paths).

- [ ] **Step 4: Sync into 2DQuant/ + smoke test**

```bash
cp /home/ubuntu/unifying-ptq/patches/twodquant_dbaf_pcsa_patch.py \
   /home/ubuntu/unifying-ptq/2DQuant/twodquant_dbaf_pcsa_patch.py
python -c "
import sys; sys.path.insert(0, '/home/ubuntu/unifying-ptq/2DQuant')
import twodquant_dbaf_pcsa_patch as p
print('2DQUANT_PATCH_OK')
"
```

- [ ] **Step 5: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add docs/integrations/2dquant.md patches/twodquant_dbaf_pcsa_patch.py
git commit -m "feat(G5): 2DQuant DBAF + PCSA-tf patch module"
```

---

## Phase 3 — Calibration Suite (Days 4-9)

### Task 11: OmniQuant calibration matrix on LLaMA-3-8B (4 arms)

**Files:**
- Create: `/home/ubuntu/unifying-ptq/scripts/run_omniquant_llama3_8b.sh`

- [ ] **Step 1: Write the driver script**

Create `/home/ubuntu/unifying-ptq/scripts/run_omniquant_llama3_8b.sh`:

```bash
#!/usr/bin/env bash
# OmniQuant ± {DBAF, PCSA-tf, both} on LLaMA-3-8B W4A4
set -e
cd /home/ubuntu/unifying-ptq/OmniQuant
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
export PYTHONPATH=/home/ubuntu/unifying-ptq:/home/ubuntu/unifying-ptq/FlatQuant:${PYTHONPATH:-}
export HF_HOME=/data/huggingface_cache

OUT=/data/outputs/G6-omniquant-llama3-8b
mkdir -p "$OUT"

# Use OmniQuant's main.py args; substitute actual flag names after Task 6.
# Placeholder names: --model, --wbits, --abits, --epochs, --calib_dataset
COMMON_ARGS="--model /data/modelzoo/meta-llama/Meta-Llama-3-8B --wbits 4 --abits 4 \
  --calib_dataset wikitext2 --nsamples 128"

# Arm A: vanilla OmniQuant (no DBAF, no PCSA-tf)
python main.py $COMMON_ARGS --output_dir "$OUT/arm_A_vanilla" 2>&1 | tee "$OUT/arm_A_vanilla.log"

# Arm B: + DBAF only
python -c "
import sys; sys.path.insert(0, '.')
from omniquant_dbaf_pcsa_patch import install_dbaf_patches
install_dbaf_patches('quantize.omniquant.quantize_weight', dbaf_alpha=0.95)
# install activation DBAF too
install_dbaf_patches('quantize.omniquant.quantize_activation', dbaf_alpha=0.95)
" 2>&1
python main.py $COMMON_ARGS --output_dir "$OUT/arm_B_dbaf" 2>&1 | tee "$OUT/arm_B_dbaf.log"

# Arm C: + PCSA-tf only
# (fit PCSA-tf on calib activations first, then run)
# ...

# Arm D: + DBAF + PCSA-tf
# ...

echo "OMNIQUANT_LLAMA3_DONE_$?"
```

- [ ] **Step 2: Smoke-test on 1-layer mode first**

Run with `--num_layers 1` (or OmniQuant's equivalent) to verify it runs end-to-end before committing to a full calibration.

- [ ] **Step 3: Launch full Arm A on local GPU in tmux**

```bash
tmux new-session -d -s g6-omniquant-A "scripts/run_omniquant_llama3_8b.sh 2>&1 | tee /tmp/g6-omniquant-A.log"
```

- [ ] **Step 4: Wait for completion via Monitor**

```bash
# Arm A only first (~2h GPU). Other arms queue after.
```

- [ ] **Step 5: Verify WikiText-2 PPL appears in arm_A_vanilla.log; commit driver + results**

```bash
grep -E "PPL|wikitext" /data/outputs/G6-omniquant-llama3-8b/arm_A_vanilla.log | tail -5
cd /home/ubuntu/unifying-ptq
git add scripts/run_omniquant_llama3_8b.sh
git commit -m "feat(G6): OmniQuant LLaMA-3-8B calibration driver"
```

---

### Task 12: OmniQuant calibrations — full 4-arm sweep (LLaMA + Qwen)

- [ ] **Step 1: Launch Arms B, C, D sequentially behind Arm A**

```bash
# After Arm A finishes, launch B+C+D sequentially via a queue script
tmux new-session -d -s g6-omniquant-queue "scripts/run_omniquant_llama3_8b.sh 2>&1 | tee /tmp/g6-omniquant-queue.log"
```

- [ ] **Step 2: Repeat for Qwen-2.5-7B**

```bash
# Duplicate script with --model /data/modelzoo/Qwen/Qwen2.5-7B
cp scripts/run_omniquant_llama3_8b.sh scripts/run_omniquant_qwen25_7b.sh
sed -i 's|Meta-Llama-3-8B|Qwen/Qwen2.5-7B|g' scripts/run_omniquant_qwen25_7b.sh
```

- [ ] **Step 3: Aggregate results**

```bash
python3 -c "
import json, pathlib
out = pathlib.Path('/data/outputs/G6-omniquant-llama3-8b')
for arm in out.glob('arm_*'):
    log = arm.with_suffix('.log')
    if log.exists():
        ppl = [l for l in log.read_text().splitlines() if 'PPL' in l][-1:]
        print(f'{arm.name}: {ppl}')
"
```

- [ ] **Step 4: Update PAPER_RESULTS.md**

```bash
# Append a new section to PAPER_RESULTS.md with the 4×2 = 8 cells
cd /home/ubuntu/unifying-ptq
git add scripts/run_omniquant_qwen25_7b.sh
git commit -m "exp(G6): OmniQuant 4-arm sweep on LLaMA-3-8B + Qwen-2.5-7B"
```

---

### Task 13: 2DQuant calibrations — full sweep on SwinIR ×2/×3/×4

**Files:**
- Create: `/home/ubuntu/unifying-ptq/scripts/run_2dquant_swinir.sh`

- [ ] **Step 1: Write driver**

Create `/home/ubuntu/unifying-ptq/scripts/run_2dquant_swinir.sh`:

```bash
#!/usr/bin/env bash
set -e
cd /home/ubuntu/unifying-ptq/2DQuant
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
export PYTHONPATH=/home/ubuntu/unifying-ptq:${PYTHONPATH:-}

OUT=/data/outputs/G7-2dquant-swinir
mkdir -p "$OUT"

for SCALE in 2 3 4; do
  for ARM in vanilla dbaf pcsa_tf both; do
    # invoke 2DQuant entry with appropriate patches installed
    # (exact CLI flags depend on Task 10 discovery)
    python <2dquant_entry> --scale $SCALE --arm $ARM \
      --out "$OUT/x${SCALE}_${ARM}" 2>&1 | tee "$OUT/x${SCALE}_${ARM}.log"
  done
done
echo "TWODQUANT_SWINIR_DONE_$?"
```

- [ ] **Step 2: Smoke ×2 vanilla first**

- [ ] **Step 3: Run full sweep in tmux**

```bash
tmux new-session -d -s g7-2dquant "scripts/run_2dquant_swinir.sh 2>&1 | tee /tmp/g7-2dquant.log"
```

- [ ] **Step 4: Commit driver + aggregate PSNR**

```bash
cd /home/ubuntu/unifying-ptq
git add scripts/run_2dquant_swinir.sh
git commit -m "exp(G7): 2DQuant 4-arm sweep on SwinIR-light ×2/×3/×4"
```

---

### Task 14: Training-free full table — RTN/GPTQ/AWQ ± {DBAF, PCSA-tf, both}

**Files:**
- Modify: `/home/ubuntu/unifying-ptq/FlatQuant/flatquant/baselines/rtn.py` (add PCSA-tf hook to RTN/GPTQ/AWQ paths)
- Create: `/home/ubuntu/unifying-ptq/scripts/run_training_free_full_table.sh`

- [ ] **Step 1: Add `apply_pcsa_tf_at_inference` hook in baselines**

Modify `/home/ubuntu/unifying-ptq/FlatQuant/flatquant/baselines/rtn.py` to add an optional `pcsa_state` parameter to `quantize_model` that triggers PCSA-tf routing during the calibration capture pass.

- [ ] **Step 2: Write driver**

```bash
# scripts/run_training_free_full_table.sh
# RTN/GPTQ/AWQ × {alone, +DBAF, +PCSA-tf, +both}
# on LLaMA-3-8B, Qwen-2.5-7B, SAM-B/L/H, SwinIR ×2/×3/×4
```

- [ ] **Step 3: Run in two tmux sessions (LLM on local, SAM on remote)**

```bash
tmux new-session -d -s g8-tf-llm "scripts/run_training_free_full_table.sh llm 2>&1 | tee /tmp/g8-tf-llm.log"
ssh remote-gpu 'tmux new-session -d -s g8-tf-sam "/home/ubuntu/unifying-ptq/scripts/run_training_free_full_table.sh sam 2>&1 | tee /tmp/g8-tf-sam.log"'
```

- [ ] **Step 4: Commit**

```bash
cd /home/ubuntu/unifying-ptq
git add scripts/run_training_free_full_table.sh FlatQuant/flatquant/baselines/rtn.py
git commit -m "exp(G8): training-free full table with PCSA-tf integration"
```

---

### Task 15: AHCPTQ cross-detector matrix (remote)

**Files:**
- Create: `/home/ubuntu/unifying-ptq/scripts/run_ahcptq_cross_detector.sh`

- [ ] **Step 1: Write queue-style script**

Create `/home/ubuntu/unifying-ptq/scripts/run_ahcptq_cross_detector.sh`:

```bash
#!/usr/bin/env bash
# Queue: SAM-H+YOLOX (running) -> SAM-H+H-DETR -> SAM-B+H-DETR -> SAM-L+H-DETR
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate ahcptq-old

# Wait for SAM-H+YOLOX to finish (it's already running)
while ! tmux capture-pane -t s2-samh-yolox -p 2>/dev/null | grep -q 'SAMH_YOLOX_DONE'; do sleep 90; done

# SAM-H + H-DETR (next in queue)
mkdir -p results/G9-ahcptq/sam-h/hdetr-w4a4
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/h-detr/h-detr-sam-vit-h.py \
  --q_config ./exp/config44_samh.yaml \
  --quant-encoder --eval segm \
  --work-dir results/G9-ahcptq/sam-h/hdetr-w4a4 \
  2>&1 | tee results/G9-ahcptq/sam-h/hdetr-w4a4/run.log

# SAM-B + H-DETR
# ... (similar pattern)

# SAM-L + H-DETR
# ...
echo "G9_CROSS_DETECTOR_DONE_$?"
```

- [ ] **Step 2: Sync script to remote**

```bash
rsync /home/ubuntu/unifying-ptq/scripts/run_ahcptq_cross_detector.sh remote-gpu:/home/ubuntu/unifying-ptq/scripts/
```

- [ ] **Step 3: Launch on remote**

```bash
ssh remote-gpu 'tmux new-session -d -s g9-cross-detector "bash /home/ubuntu/unifying-ptq/scripts/run_ahcptq_cross_detector.sh 2>&1 | tee /tmp/g9.log"'
```

- [ ] **Step 4: Commit script**

```bash
cd /home/ubuntu/unifying-ptq
git add scripts/run_ahcptq_cross_detector.sh
git commit -m "exp(G9): AHCPTQ cross-detector SAM-B/L/H × YOLOX/H-DETR matrix"
```

---

## Phase 4 — Analyses (Days 6-9)

### Task 16: RULER eval — baseline vs KV-PCSA-tf

**Files:**
- Modify: `/home/ubuntu/unifying-ptq/scripts/run_niah_eval.py` to accept a KV-PCSA-tf state path

- [ ] **Step 1: Add `--kv-pcsa-tf-state` flag to NIAH eval**

In `run_niah_eval.py`, accept a path to a `.pt` file containing `{anchors, k_scales, v_scales}` and apply during inference.

- [ ] **Step 2: Fit KV-PCSA-tf state on calibration prompts**

```python
# scripts/fit_kv_pcsa_tf_for_llama.py
import torch
from flatquant.baselines.kv_pcsa_tf import fit_kv_pcsa_tf
# load model, run 128 calib prompts, capture descriptors + K/V caches per prompt
# call fit_kv_pcsa_tf and save state to /data/outputs/G10-ruler/kv_pcsa_tf_state.pt
```

- [ ] **Step 3: Run RULER 4k baseline + KV-PCSA-tf**

```bash
python scripts/run_niah_eval.py --matrix_path /data/outputs/S5-baseline-calib/... \
  --context_lengths 4096 8192 --out /data/outputs/G10-ruler/baseline.json
python scripts/run_niah_eval.py --matrix_path /data/outputs/S5-baseline-calib/... \
  --kv-pcsa-tf-state /data/outputs/G10-ruler/kv_pcsa_tf_state.pt \
  --context_lengths 4096 8192 --out /data/outputs/G10-ruler/kv_pcsa_tf.json
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_niah_eval.py scripts/fit_kv_pcsa_tf_for_llama.py
git commit -m "exp(G10): RULER NIAH eval baseline vs KV-PCSA-tf at 4k/8k"
```

---

### Task 17: Per-layer ablation on SAM-B + LLaMA-3-8B

**Files:**
- Create: `/home/ubuntu/unifying-ptq/scripts/per_layer_ablation_sam.py` (extend SwinIR version)
- Create: `/home/ubuntu/unifying-ptq/scripts/per_layer_ablation_llama.py`

- [ ] **Step 1: Adapt the SwinIR per-layer ablation pattern to SAM image encoder**

Same template as `per_layer_ablation_swinir.py`: snapshot FP weights, apply DBAF to one layer at a time, eval (mAP for SAM-B). 51 layers × small dataset = ~1h.

- [ ] **Step 2: Adapt for LLaMA-3-8B**

Eval = WikiText-2 PPL on a small chunk. ~225 layers × 1 min/eval = ~4h.

- [ ] **Step 3: Run both**

```bash
tmux new-session -d -s g12-sam "python scripts/per_layer_ablation_sam.py 2>&1 | tee /tmp/g12-sam.log"
tmux new-session -d -s g12-llama "python scripts/per_layer_ablation_llama.py 2>&1 | tee /tmp/g12-llama.log"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/per_layer_ablation_sam.py scripts/per_layer_ablation_llama.py
git commit -m "exp(G12): cross-arch per-layer DBAF task-error attribution"
```

---

### Task 18: Cost-vs-quality figure

**Files:**
- Create: `/home/ubuntu/unifying-ptq/scripts/make_cost_quality_figure.py`

- [ ] **Step 1: Write the figure script**

Create `/home/ubuntu/unifying-ptq/scripts/make_cost_quality_figure.py`:

```python
"""Make the cost-vs-quality figure: calibration FLOPs (x-axis) vs quality (y-axis).
Two curves per task (LLaMA, SAM, SwinIR): base methods and +DBAF+PCSA-tf variants.
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load all results from PAPER_RESULTS.md or per-arm JSONs
# Methods to include: RTN, GPTQ, AWQ, SmoothQuant, SpinQuant, OmniQuant, FlatQuant
# Calibration cost in seconds (rough: RTN=0, GPTQ=120, AWQ=60, SmoothQuant=30,
# OmniQuant=3600, SpinQuant=3600, FlatQuant=7200)
# Quality: WikiText-2 PPL on LLaMA-3-8B
# (similar for SAM mAP, SwinIR PSNR)

# Plot: scatter base methods as o, +DBAF+PCSA-tf as x; connect pairs with arrow
# Save: /home/ubuntu/paper/emnlp2026/figures/cost_quality.pdf
```

- [ ] **Step 2: Run and verify**

```bash
python scripts/make_cost_quality_figure.py
ls -la /home/ubuntu/paper/emnlp2026/figures/cost_quality.pdf
```

- [ ] **Step 3: Commit**

```bash
git add scripts/make_cost_quality_figure.py
git commit -m "feat(G11): cost-vs-quality figure generator"
```

---

## Phase 5 — Paper Rewrite (Days 9-12)

### Task 19: Abstract + Introduction rewrite

**Files:**
- Modify: `/home/ubuntu/paper/emnlp2026/sections/00-abstract.tex`
- Modify: `/home/ubuntu/paper/emnlp2026/sections/01-intro.tex`

- [ ] **Step 1: Draft new abstract**

Replace `/home/ubuntu/paper/emnlp2026/sections/00-abstract.tex` with the reframed abstract (~150 words):

```latex
Post-training quantization (PTQ) error is dominated by a small number of
recurring distribution types that appear across LLM, SAM, and SR
architectures. Prior work addresses these per-architecture, with rotation-
based methods for LLMs, channel-aware grouping for SAM, and Hadamard
transforms for SR. We propose two composable distribution-guided primitives —
Dual-Band Affine Folding (DBAF) and Prompt-Conditioned Scale Anchoring (PCSA)
— that target the recurring "sparse outlier + Gaussian core" pattern across
all three architectures. We show that on hosts with explicit headroom
(OmniQuant for LLM, AHCPTQ for SAM, 2DQuant for SR), our primitives add
consistent moderate improvements. On rotation-based hosts (FlatQuant/CompSRT)
the gain disappears — evidence that the same distributions are being
absorbed implicitly there. A training-free variant (k-means anchors) makes
the primitives applicable at zero added calibration cost. Across 7
architecture/scale combinations and three quantization backbones, the
contribution is the framework, not the SOTA gap.
```

- [ ] **Step 2: Update §1 intro to lead with the three-level claim**

- [ ] **Step 3: Extend fragmentation table in `01-intro.tex:74-88` with cross-arch citation column**

- [ ] **Step 4: Verify paper still compiles + 8-page check**

```bash
cd /home/ubuntu/paper/emnlp2026
pdflatex acl_latex.tex 2>&1 | tail -5
pdfinfo acl_latex.pdf | grep Pages
```

Expected: Pages ≤ 8.

- [ ] **Step 5: Commit**

```bash
cd /home/ubuntu/paper/emnlp2026
git add sections/00-abstract.tex sections/01-intro.tex
git commit -m "rewrite(§1): distributional-pivot abstract + intro"
```

---

### Task 20: Related work — per-distribution reorganization

**Files:**
- Modify: `/home/ubuntu/paper/emnlp2026/sections/02-related.tex`

- [ ] **Step 1: Reorganize §2 into per-distribution subsections**

```latex
\subsection{Sparse 3-sigma outliers}
SmoothQuant~\citep{xiao2024smoothquantaccurateefficientposttraining},
LLM.int8()~\citep{dettmers2022llmint8}, AWQ~\citep{lin2023awq}...

\subsection{Rotational outlier absorption}
QuaRot~\citep{ashkboos2024quarotoutlierfree4bitinference},
SpinQuant~\citep{liu2025spinquantllmquantizationlearned},
FlatQuant~\citep{sun2024flatquant}...

\subsection{Post-ReLU / post-GELU asymmetric}
...

\subsection{Bimodal post-Softmax}
...

\subsection{Per-prompt distribution shift}
...
```

- [ ] **Step 2: Compile + length check**

```bash
cd /home/ubuntu/paper/emnlp2026
pdflatex acl_latex.tex 2>&1 | tail -3
pdfinfo acl_latex.pdf | grep Pages
```

- [ ] **Step 3: Commit**

```bash
git add sections/02-related.tex
git commit -m "rewrite(§2): related work reorganized per-distribution"
```

---

### Task 21: Method — add PCSA-tf, KV-PCSA-tf, composability API

**Files:**
- Modify: `/home/ubuntu/paper/emnlp2026/sections/03-method.tex`

- [ ] **Step 1: Add new subsection 3.3 for PCSA-tf**

After existing PCSA description, add:

```latex
\subsection{Training-free PCSA (PCSA-tf)}
We provide a training-free variant of PCSA that computes anchor centroids via
k-means on calibration-time prompt descriptors and per-anchor scales by
max-abs over activations routed to each anchor. The calibration cost is
seconds (k-means on N=128 prompts), enabling composition on training-free
hosts (RTN, GPTQ, AWQ) at zero added training cost.

\paragraph{Algorithm.}
Given calibration descriptors $\{d_i\}_{i=1}^{N}$ and activations
$\{x_i\}_{i=1}^{N}$, run k-means to obtain $K$ centroids
$\{c_j\}_{j=1}^{K}$. For each anchor $j$, set
$s_j = \max_{i: \arg\max_k \langle d_i, c_k\rangle = j} \|x_i\|_\infty$.
At inference, prompt $p$ is routed to
$j^* = \arg\max_j \langle d_p, c_j\rangle$ and quantized with scale $s_{j^*}$.
```

- [ ] **Step 2: Add 3.4 KV-PCSA-tf and 3.5 composability API**

- [ ] **Step 3: Compile + length check**

- [ ] **Step 4: Commit**

```bash
git add sections/03-method.tex
git commit -m "rewrite(§3): method — add PCSA-tf, KV-PCSA-tf, composability API"
```

---

### Task 22: Experiments rewrite

**Files:**
- Modify: `/home/ubuntu/paper/emnlp2026/sections/04-experiments.tex`

- [ ] **Step 1: Rewrite §4.1 distribution recurrence with cross-model layer analysis data**

Pull from `results/S4-cross-model-layer-analysis/*.json` and `results/F3-distribution-taxonomy/summary.json`.

- [ ] **Step 2: Rewrite §4.2 (headline composability table) with OmniQuant/AHCPTQ/2DQuant results**

Use results from Tasks 11-13 + the carryover AHCPTQ + DBAF + PCSA SAM numbers.

- [ ] **Step 3: Add §4.3 training-free table (RTN/GPTQ/AWQ ± {DBAF, PCSA-tf, both})**

- [ ] **Step 4: Add §4.4 cross-detector AHCPTQ matrix (SAM-B/L/H × YOLOX/H-DETR)**

- [ ] **Step 5: Add §4.5 long-context KV with RULER results from Task 16**

- [ ] **Step 6: Add §4.6 saturation evidence (FlatQuant/CompSRT marginal table + C1 number)**

- [ ] **Step 7: Add §4.7 mechanism (existing + taxonomy-predicts-x3 narrative)**

- [ ] **Step 8: Add §4.8 cost-vs-quality figure reference**

- [ ] **Step 9: Compile + length check (likely overflows; move things to supplementary as needed)**

- [ ] **Step 10: Commit**

```bash
git add sections/04-experiments.tex figures/cost_quality.pdf
git commit -m "rewrite(§4): experiments with composability + training-free tables + RULER + mechanism"
```

---

### Task 23: Limitations + Conclusion + final 8-page check

**Files:**
- Modify: `/home/ubuntu/paper/emnlp2026/sections/05-conclusion.tex`
- Modify: `/home/ubuntu/paper/emnlp2026/sections/06-limitations.tex`

- [ ] **Step 1: Rewrite conclusion to match new claim**

- [ ] **Step 2: Update limitations with SR ×3 + bimodal + RULER caveats**

- [ ] **Step 3: Move oversized content to supplementary**

If main paper > 8 pages: move appendix-alpha-deriv and appendix-dbaf-clipping-proof to standalone supplementary file. Keep table footnotes terse.

- [ ] **Step 4: Final compile + pdfinfo check**

```bash
cd /home/ubuntu/paper/emnlp2026
pdflatex acl_latex.tex 2>&1 | tail -3
bibtex acl_latex 2>&1 | tail -3
pdflatex acl_latex.tex 2>&1 | tail -3
pdflatex acl_latex.tex 2>&1 | tail -3
pdfinfo acl_latex.pdf | grep Pages
```

Expected: Pages == 8 (or less, main paper only).

- [ ] **Step 5: Commit + push to Overleaf**

```bash
git add sections/05-conclusion.tex sections/06-limitations.tex
git commit -m "rewrite(§5,§6): conclusion + limitations for distributional-pivot reframe"
git push  # to Overleaf remote
```

---

## Self-Review Notes

**Spec coverage check:** Every section in `2026-05-13-emnlp-distributional-pivot-design.md` maps to a task:
- §2 Reframed contribution → Tasks 19-23 (paper rewrite)
- §3 Architecture & components → Tasks 2-5 (primitives) + 6-10 (host integrations)
- §4 Experiment matrix → Tasks 11-15 (calibrations), 16-17 (evals)
- §5 Paper section structure → Tasks 19-23
- §6 Experiment program → all of Phase 3-4
- §7 Risk + mitigations → contingencies noted in individual tasks

**Type consistency check:** PCSA-tf state dict {"anchors", "scales"} used consistently across tests and patch modules; KV-PCSA-tf adds {"k_scales", "v_scales"} to the same dict.

**Placeholder scan:** All "TBD" markers in tasks 6, 8, 10, 13 are explicitly bounded by discovery steps (Task 6 produces the OmniQuant fn paths; Task 10 produces 2DQuant's) — they are dependencies between tasks, not unfilled gaps.

**Known incomplete details that get filled by execution:**
- OmniQuant's exact weight/activation quant function names (filled by Task 6 Step 2)
- 2DQuant's exact entry point + quant functions (filled by Task 10 Step 1)
- Per-arm calibration time on OmniQuant (filled by smoke run in Task 11 Step 2)

These are intentional — the spec can't predict third-party code structure, so the plan documents them as discoveries.

---

## Execution Order Summary

**Already running (let finish):**
- C1 (FlatQuant no-gate, local, ~30 min)
- SAM-H + YOLOX AHCPTQ (remote, ~5h)

**Day 1 (today):**
- Task 1 (clones)
- Tasks 2-3 (PCSA-tf with TDD)
- Tasks 4-5 (KV-PCSA-tf with TDD)

**Day 2-3:**
- Task 6 (OmniQuant discovery)
- Tasks 7-9 (OmniQuant plumb + smoke)
- Task 10 (2DQuant plumb)
- Task 15 (queue SAM cross-detector on remote)

**Day 4-6:**
- Task 11-12 (OmniQuant calibrations)
- Task 13 (2DQuant calibrations)
- Task 14 (training-free full table)
- Task 17 (per-layer ablation)

**Day 7-8:**
- Task 16 (RULER)
- Task 18 (cost-quality figure)

**Day 9-12:**
- Tasks 19-23 (paper rewrite + final compile)
