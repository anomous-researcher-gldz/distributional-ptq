"""RTN + DBAF + PCSA-tf with FOLDED calibration on LLaMA-3-8B W4A4.

Variant 7 of the PCSA-tf failure catalog. Variants 1-6 calibrated
PCSA-tf scales on the FP model's UN-folded activations, then at
inference saw the FOLDED activations (after DBAF at alpha=0.25)
divided by an un-folded-magnitude scale -- bulk quantized to ~0.

Variant 7 calibrates on the FOLDED distribution: each captured
calibration activation is DBAF-folded at the same (T_sigma=3, alpha)
before fit_pcsa_tf computes the per-anchor 99th-percentile scale.
Now the calibrated scale matches the inference distribution.

Wiring:
  1) FP forward + pre-hooks on each decoder block -> capture x (un-folded)
  2) FOLD x per token at T=3*sigma, alpha=0.25
  3) Descriptors = mean over tokens of UN-folded x (matches inference)
     Acts        = folded x (used by fit_pcsa_tf for scales)
  4) RTN quantize weights with DBAF alpha=0.25
  5) Apply PCSA-tf hooks with use_dbaf=True, dbaf_alpha=0.25 at inference
     (fold-around-PCSA, matching the calibration fold).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "FlatQuant"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="llama3-8b")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--T_sigma", type=float, default=3.0)
    p.add_argument("--pcsa_k", type=int, default=8)
    p.add_argument("--out_root", default="/data/outputs/PCSA-tf-folded-calib")
    return p.parse_args(argv)


def _dbaf_fold_per_token(x, alpha, T_sigma):
    import torch
    sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-9)
    T = T_sigma * sigma
    sgn = torch.sign(x)
    mask = x.abs() > T
    return torch.where(mask, sgn * T + alpha * (x - sgn * T), x)


def _collect_pcsa_state_folded(model, tok, alpha: float, T_sigma: float,
                                K: int = 8) -> dict:
    """Per-Linear PCSA-tf calibration with FOLDED activations.

    Hooks each q/k/v/o/gate/up_proj individually (matches the application
    path in _apply_pcsa_tf_to_llm). For each, fold the captured input at
    (alpha, T_sigma) before fit_pcsa_tf, so the calibrated scale matches
    the post-fold distribution PCSA-tf actually sees at inference.

    Per-block calibration (pre-2026-05-23) gave catastrophic PPL because
    scales fit on the residual stream were applied to post-RMSNorm Linear
    inputs with different magnitudes.
    """
    import re
    import torch
    import torch.nn as nn
    from flatquant.baselines.pcsa_tf import fit_pcsa_tf
    from run_training_free_full_table import _calib_batch_llm

    print("[v7] capturing per-Linear activations on FP model ...", flush=True)

    hidden_size = model.config.hidden_size
    pat = re.compile(r"model\.layers\.\d+\.")
    target_names: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "lm_head" in name:
            continue
        if mod.in_features != hidden_size:
            continue
        if not pat.search(name):
            continue
        target_names.append(name)

    per_linear_inputs: dict[str, list] = {n: [] for n in target_names}
    hooks = []

    def _make_pre_hook(linear_name):
        def _h(module, args):
            per_linear_inputs[linear_name].append(args[0].detach().float().cpu())
        return _h

    name_to_mod = dict(model.named_modules())
    for n in target_names:
        hooks.append(name_to_mod[n].register_forward_pre_hook(_make_pre_hook(n)))

    calib = _calib_batch_llm(tok)
    with torch.no_grad():
        ids = calib.to(next(model.parameters()).device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        model(ids)

    for h in hooks:
        h.remove()

    state: dict = {}
    for n in target_names:
        bufs = per_linear_inputs[n]
        if not bufs:
            continue
        x = bufs[0]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        # Descriptors: UN-folded mean over tokens (matches inference routing,
        # which routes on the un-folded input then folds inside PCSA-tf).
        descs = x.mean(dim=1)
        # Acts: FOLDED at the same alpha/T_sigma the inference path uses.
        acts_folded = _dbaf_fold_per_token(x, alpha, T_sigma)
        state[n] = fit_pcsa_tf(descs, acts_folded, K=K)

    if state:
        first = next(iter(state))
        last = list(state)[-1]
        print(f"[v7] fitted PCSA-tf state for {len(state)} Linears "
              f"(folded calib, alpha={alpha})", flush=True)
        print(f"[v7] scales[{first}]={state[first]['scales'].tolist()}", flush=True)
        print(f"[v7] scales[{last}]={state[last]['scales'].tolist()}", flush=True)
    return state


def main():
    import torch
    args = _parse_args()
    from flatquant.baselines.rtn import quantize_model as rtn_quantize_model
    from run_training_free_full_table import (
        _load_llm, _eval_ppl_wikitext2, _eval_ppl_c4,
        _apply_pcsa_tf_to_llm,
    )

    out_dir = pathlib.Path(args.out_root) / args.target / \
              f"rtn_dbaf_pcsa_tf_foldedcalib_a{args.alpha:.2f}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval.json"

    print(f"[v7] target={args.target} alpha={args.alpha} K={args.pcsa_k}",
          flush=True)
    t0 = time.time()
    model, tok = _load_llm(args.target)

    pcsa_state = _collect_pcsa_state_folded(model, tok, alpha=args.alpha,
                                            T_sigma=args.T_sigma, K=args.pcsa_k)

    print(f"[v7] RTN quantize (W4, DBAF alpha={args.alpha}) ...", flush=True)
    model = rtn_quantize_model(model, bits=4, use_dbaf=True, alpha=args.alpha)

    print(f"[v7] applying PCSA-tf hooks (DBAF alpha={args.alpha}) ...",
          flush=True)
    model = _apply_pcsa_tf_to_llm(model, pcsa_state, use_dbaf=True,
                                  dbaf_alpha=args.alpha)

    print("[v7] eval wt2 ...", flush=True)
    wt2_ppl = _eval_ppl_wikitext2(model, tok)
    print(f"[v7] wt2 = {wt2_ppl:.3f}", flush=True)
    try:
        c4_ppl = _eval_ppl_c4(model, tok)
        print(f"[v7] c4 = {c4_ppl:.3f}", flush=True)
    except Exception as exc:
        print(f"[v7] WARNING: c4 eval failed: {exc}", flush=True)
        c4_ppl = float("nan")

    result = {
        "target": args.target,
        "method": "rtn",
        "augments": "dbaf+pcsa_tf_foldedcalib",
        "alpha": args.alpha,
        "pcsa_k": args.pcsa_k,
        "metrics": {"wikitext2_ppl": wt2_ppl, "c4_ppl": c4_ppl},
        "wallclock_seconds": time.time() - t0,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
