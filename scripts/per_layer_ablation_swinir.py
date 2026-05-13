"""Per-layer DBAF ablation for training-free SwinIR.

For a chosen scale (×2/×3/×4) and dataset (Set5/Urban100):
  baseline   = no-DBAF (RTN per-channel on all Conv2d/Linear)
  per_layer  = DBAF applied to EXACTLY ONE layer at a time; RTN on the others.

For each layer L, report the PSNR delta vs baseline. Layers with positive
delta are where DBAF is task-positive; negative deltas are task-harmful.
Cross-reference each layer's `is_like_normal_plus_3sigma_outliers(weight)`
gate decision and frac>3σ to test whether the gate predicts task-level
gain (where MSE-only predictivity was inverted).
"""
from __future__ import annotations
import sys, json, pathlib, glob, argparse, time, copy
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
from ahcptq.quantization.fake_quant import is_like_normal_plus_3sigma_outliers
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf
sys.path.insert(0, "/home/ubuntu/unifying-ptq/scripts")
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("rtfs", "/home/ubuntu/unifying-ptq/scripts/run_training_free_swinir.py")
_rtfs = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_rtfs)


def _rtn_layer(w: torch.Tensor, bits: int) -> torch.Tensor:
    out_c = w.shape[0]
    w_flat = w.view(out_c, -1) if w.dim() > 2 else w
    return _quantize_tensor_uniform(w_flat, bits, per_channel=True).view_as(w).to(w.dtype)


def _dbaf_layer(w: torch.Tensor, bits: int, alpha: float) -> torch.Tensor:
    out_c = w.shape[0]
    w_flat = w.view(out_c, -1) if w.dim() > 2 else w
    return _quantize_per_channel_with_dbaf(w_flat.float(), bits, alpha=alpha).view_as(w).to(w.dtype)


def list_target_layers(model) -> list[str]:
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and mod.weight.numel() >= 64:
            names.append(name)
    return names


def apply_quant_with_one_dbaf_layer(model, fp_weights, dbaf_layer: str | None, bits: int, alpha: float):
    """Reload FP weights, then quantize all layers RTN except `dbaf_layer` which gets DBAF."""
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and mod.weight.numel() >= 64:
            w_fp = fp_weights[name]
            w_q = _dbaf_layer(w_fp, bits, alpha) if name == dbaf_layer else _rtn_layer(w_fp, bits)
            mod.weight.data = w_q.to(mod.weight.dtype).clone()


def evaluate_psnr(model, scale: int, dataset_dir: str) -> float:
    psnrs = []
    hr_paths = sorted(glob.glob(f"{dataset_dir}/*.png") + glob.glob(f"{dataset_dir}/*.PNG"))
    if not hr_paths:
        hr_paths = sorted(glob.glob(f"{dataset_dir}/HR/*.png"))
    for hr_path in hr_paths:
        hr = np.array(Image.open(hr_path).convert("RGB"))
        h, w = hr.shape[:2]; h -= h % scale; w -= w % scale
        hr = hr[:h, :w]
        lr = np.array(Image.fromarray(hr).resize((w // scale, h // scale), Image.BICUBIC))
        lh, lw = lr.shape[:2]; lh -= lh % 8; lw -= lw % 8
        lr = lr[:lh, :lw]
        hr_crop = hr[:lh*scale, :lw*scale]
        x = torch.from_numpy(lr).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
        with torch.no_grad():
            sr = model(x).clamp(0, 1)
        sr_np = (sr[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        psnrs.append(_rtfs.psnr(sr_np, hr_crop, crop=scale))
    return float(np.mean(psnrs)) if psnrs else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=int, required=True)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="Limit number of layers (for testing)")
    args = p.parse_args()

    model = _rtfs.load_swinir(args.scale, args.pretrained).cuda().eval()
    # Snapshot FP weights so we can reset between trials.
    fp_weights = {name: mod.weight.data.clone() for name, mod in model.named_modules()
                  if isinstance(mod, (nn.Conv2d, nn.Linear)) and mod.weight.numel() >= 64}
    layers = list_target_layers(model)
    if args.limit:
        layers = layers[:args.limit]
    print(f"[per-layer-ablation] scale={args.scale} dataset={pathlib.Path(args.dataset).name} "
          f"target_layers={len(layers)}", flush=True)

    # Baseline: no DBAF anywhere.
    apply_quant_with_one_dbaf_layer(model, fp_weights, dbaf_layer=None, bits=args.bits, alpha=args.alpha)
    psnr_baseline = evaluate_psnr(model, args.scale, args.dataset)
    print(f"[baseline] no-DBAF PSNR = {psnr_baseline:.4f}", flush=True)

    rows = []
    for i, name in enumerate(layers):
        # Gate / outlier stats for the FP weight.
        fp = fp_weights[name]
        flat = fp.detach().float().reshape(-1)
        mu, sd = flat.mean(), flat.std().clamp_min(1e-8)
        z = (flat - mu) / sd
        frac3 = float((z.abs() > 3.0).float().mean().item())
        gate = bool(is_like_normal_plus_3sigma_outliers(fp)["is_like_c"])

        apply_quant_with_one_dbaf_layer(model, fp_weights, dbaf_layer=name,
                                        bits=args.bits, alpha=args.alpha)
        psnr_l = evaluate_psnr(model, args.scale, args.dataset)
        delta = psnr_l - psnr_baseline
        rows.append({"layer": name, "shape": list(fp.shape), "frac3": frac3,
                     "gate": gate, "psnr": psnr_l, "psnr_delta": delta})
        if (i + 1) % 10 == 0 or i == len(layers) - 1:
            print(f"  [{i+1:3d}/{len(layers)}] {name}: gate={gate} frac3={frac3:.4f} "
                  f"PSNR={psnr_l:.4f} Δ={delta:+.4f}", flush=True)

    # Aggregate: split by gate, compute mean PSNR delta on each side.
    gate_pass = [r["psnr_delta"] for r in rows if r["gate"]]
    gate_fail = [r["psnr_delta"] for r in rows if not r["gate"]]
    summary = {
        "scale": args.scale, "dataset": pathlib.Path(args.dataset).name,
        "baseline_psnr": psnr_baseline,
        "n_layers": len(rows),
        "n_gate_pass": len(gate_pass), "n_gate_fail": len(gate_fail),
        "mean_delta_gate_pass": float(np.mean(gate_pass)) if gate_pass else None,
        "mean_delta_gate_fail": float(np.mean(gate_fail)) if gate_fail else None,
        "max_delta": max(r["psnr_delta"] for r in rows),
        "min_delta": min(r["psnr_delta"] for r in rows),
        "n_positive_delta": sum(1 for r in rows if r["psnr_delta"] > 0),
        "n_negative_delta": sum(1 for r in rows if r["psnr_delta"] < 0),
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
