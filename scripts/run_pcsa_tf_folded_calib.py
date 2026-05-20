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
    """Like _collect_llm_pcsa_state but folds captured acts before fit."""
    import torch
    from flatquant.baselines.pcsa_tf import fit_pcsa_tf
    from run_training_free_full_table import _calib_batch_llm

    print("[v7] capturing per-block activations on FP model ...", flush=True)
    per_layer_inputs: dict[int, list] = {}
    hooks = []

    def _make_hook(layer_idx: int):
        def _h(module, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states")
            if x is not None and x.dim() == 3:
                per_layer_inputs.setdefault(layer_idx, []).append(
                    x.detach().float().cpu()
                )
        return _h

    decoder_layers = model.model.layers
    for i, blk in enumerate(decoder_layers):
        h = blk.register_forward_pre_hook(_make_hook(i), with_kwargs=True)
        hooks.append(h)

    # Same calibration set as the un-folded variant for apples-to-apples
    calib = _calib_batch_llm(tok)
    with torch.no_grad():
        for batch in calib:
            ids = batch.to(next(model.parameters()).device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            model(ids)

    for h in hooks:
        h.remove()

    state: dict = {}
    for i, x_chunks in per_layer_inputs.items():
        if not x_chunks:
            continue
        x = torch.cat(x_chunks, dim=0)  # [N, T, D]
        # Descriptors: un-folded mean over tokens (matches inference routing)
        descs = x.mean(dim=1)
        # Acts: FOLDED at the same alpha/T_sigma the inference path uses
        acts_folded = _dbaf_fold_per_token(x, alpha, T_sigma)
        state[i] = fit_pcsa_tf(descs, acts_folded, K=K)

    last = max(state)
    print(f"[v7] fitted PCSA-tf state for {len(state)} blocks "
          f"(folded calib, alpha={alpha})", flush=True)
    print(f"[v7] scales[0]={state[0]['scales'].tolist()}", flush=True)
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
