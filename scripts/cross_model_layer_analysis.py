"""Per-layer weight AND activation analysis across all evaluated models.

For each Linear/Conv2d module of each model:
  - Weight side: kurt, skew, frac>3sigma, DBAF gate, INT4-RTN MSE, INT4-RTN+DBAF MSE, gain.
  - Activation side: same fingerprints + quantization errors, computed on
    activations captured via forward hooks during a small calibration pass
    (model-appropriate inputs).

Goal: provide the direct link between distributional pattern -> quantization
error reduction by DBAF -> end-to-end gain. The gate is then justified as a
predictor of where the gain actually comes from.

Models:
  llama       - /data/modelzoo/meta-llama/Meta-Llama-3-8B
  qwen        - /data/modelzoo/Qwen/Qwen2.5-7B
  sam-b/-l/-h - /home/ubuntu/unifying-ptq/ckpt/sam_vit_{b,l,h}_*.pth
  swinir-x2/x3/x4 - SwinIR-light checkpoints

Activation calibration inputs:
  - LLM:    WikiText-2 train, 2 sequences x 1024 tokens
  - SAM:    2 Set5 HR images (used as encoder input via resize)
  - SwinIR: 2 Set5 LR images at the right scale
"""
from __future__ import annotations
import sys, json, pathlib, os, glob, argparse, time, gc
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/projects/instance_segment_anything/models")
from ahcptq.quantization.fake_quant import (
    profile_with_3sigma_outliers, is_like_normal_plus_3sigma_outliers,
)
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf
# is_like_normal_plus_3sigma_outliers is already imported above; just re-bind for clarity.
_w_gate_check = is_like_normal_plus_3sigma_outliers


# ----------- Per-tensor analysis -----------
@torch.no_grad()
def tensor_fingerprint(x: torch.Tensor) -> dict:
    s = profile_with_3sigma_outliers(x)
    flat = x.detach().float().reshape(-1)
    mu, sd = flat.mean(), flat.std().clamp_min(1e-8)
    z = (flat - mu) / sd
    frac4 = float((z.abs() > 4.0).float().mean().item())
    gate = is_like_normal_plus_3sigma_outliers(x)["is_like_c"]
    return {"kurt": s["kurtosis"], "skew": s["skew"],
            "frac3": s["frac_out_3"], "frac4": frac4,
            "dbaf_gate": bool(gate)}


@torch.no_grad()
def weight_quant_errors(w_2d: torch.Tensor, bits=4, alpha=0.95) -> dict:
    """w_2d: [out, fan_in_flat]. Returns three modes: RTN, DBAF gated, DBAF forced.

    - rtn: per-channel RTN, no DBAF.
    - dbaf_gated: DBAF only if is_like_normal_plus_3sigma_outliers(w) passes,
                  else fall back to plain per-channel RTN (mirrors flat_linear.py:61).
    - dbaf_force: DBAF on every layer regardless of gate.
    """
    w_f = w_2d.float()
    w_rtn = _quantize_tensor_uniform(w_f, bits, per_channel=True)
    gate = _w_gate_check(w_f)["is_like_c"]
    if gate:
        w_dbaf_gated = _quantize_per_channel_with_dbaf(w_f, bits, alpha=alpha)
    else:
        w_dbaf_gated = w_rtn
    w_dbaf_force = _quantize_per_channel_with_dbaf(w_f, bits, alpha=alpha)
    norm = (w_f ** 2).mean().clamp_min(1e-12).item()
    rtn_mse = ((w_f - w_rtn) ** 2).mean().item()
    dbaf_gated_mse = ((w_f - w_dbaf_gated) ** 2).mean().item()
    dbaf_force_mse = ((w_f - w_dbaf_force) ** 2).mean().item()
    return {
        "rtn_mse": rtn_mse, "rtn_nmse": rtn_mse / norm,
        "dbaf_gated_mse": dbaf_gated_mse, "dbaf_gated_nmse": dbaf_gated_mse / norm,
        "dbaf_force_mse": dbaf_force_mse, "dbaf_force_nmse": dbaf_force_mse / norm,
        "gain_gated_pct": ((rtn_mse - dbaf_gated_mse) / max(rtn_mse, 1e-12) * 100),
        "gain_force_pct": ((rtn_mse - dbaf_force_mse) / max(rtn_mse, 1e-12) * 100),
    }


from flatquant.quant_utils import ActivationQuantizer as _ActQ
from ahcptq.quantization.fake_quant import fold_outliers as _fold, unfold_outliers as _unfold


@torch.no_grad()
def activation_quant_errors(x: torch.Tensor, bits=4, alpha=0.99) -> dict:
    """Use the codebase's ActivationQuantizer for INT4 asym per-token quant.

    Three measurements per layer's captured activations:
      - rtn_mse:        ActivationQuantizer(bits=bits, sym=False, dbaf_alpha=None)
      - dbaf_gated_mse: ActivationQuantizer(bits=bits, sym=False, dbaf_alpha=alpha)
                        (DBAF applies iff is_like_normal_plus_3sigma_outliers)
      - dbaf_force_mse: same as above but with DBAF *forced* via direct fold/unfold
                        (skip the gate so we measure DBAF's effect when it would
                        otherwise self-disable)
    """
    x_f = x.detach().float()
    # 1) RTN-only (no DBAF).
    qf = _ActQ(bits=bits, sym=False, dbaf_alpha=None)
    rtn = qf.fake_quant(x_f.clone())
    rtn_mse = ((x_f - rtn) ** 2).mean().item()
    # 2) RTN + DBAF with the codebase's gate-on-by-default behavior.
    qg = _ActQ(bits=bits, sym=False, dbaf_alpha=alpha)
    dbaf_gated = qg.fake_quant(x_f.clone())
    dbaf_gated_mse = ((x_f - dbaf_gated) ** 2).mean().item()
    # 3) RTN + DBAF force-applied (no gate).
    T = float(3.0 * x_f.std().clamp_min(1e-8))
    x_fold, tag = _fold(x_f, T, alpha)
    forced_q = _ActQ(bits=bits, sym=False, dbaf_alpha=None).fake_quant(x_fold.clone())
    forced = _unfold(forced_q, tag, T, alpha)
    dbaf_force_mse = ((x_f - forced) ** 2).mean().item()
    norm = (x_f ** 2).mean().clamp_min(1e-12).item()
    return {
        "rtn_mse": rtn_mse, "rtn_nmse": rtn_mse / norm,
        "dbaf_gated_mse": dbaf_gated_mse, "dbaf_gated_nmse": dbaf_gated_mse / norm,
        "dbaf_force_mse": dbaf_force_mse, "dbaf_force_nmse": dbaf_force_mse / norm,
        "gain_gated_pct": ((rtn_mse - dbaf_gated_mse) / max(rtn_mse, 1e-12) * 100),
        "gain_force_pct": ((rtn_mse - dbaf_force_mse) / max(rtn_mse, 1e-12) * 100),
    }


# ----------- Per-model orchestration -----------
def analyze_model(model_name, model_loader, calib_inputs_fn, output_dir, device="cuda"):
    print(f"\n=== {model_name} ===", flush=True)
    t0 = time.time()
    model = model_loader().to(device).eval()
    rows = []
    # 1) Per-layer weight analysis.
    name_to_mod = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)) and mod.weight.numel() >= 64:
            if "lm_head" in name:  # match RTN baseline
                continue
            name_to_mod[name] = mod
            w = mod.weight.data
            w2 = w if w.dim() == 2 else w.view(w.shape[0], -1)
            wfp = tensor_fingerprint(w)
            wqe = weight_quant_errors(w2)
            rows.append({"layer": name, "type": type(mod).__name__,
                         "shape": list(w.shape),
                         "w_kurt": wfp["kurt"], "w_skew": wfp["skew"],
                         "w_frac3": wfp["frac3"], "w_frac4": wfp["frac4"],
                         "w_gate": wfp["dbaf_gate"],
                         "w_rtn_mse": wqe["rtn_mse"], "w_rtn_nmse": wqe["rtn_nmse"],
                         "w_dbaf_gated_mse": wqe["dbaf_gated_mse"], "w_dbaf_gated_nmse": wqe["dbaf_gated_nmse"],
                         "w_dbaf_force_mse": wqe["dbaf_force_mse"], "w_dbaf_force_nmse": wqe["dbaf_force_nmse"],
                         "w_gain_gated_pct": wqe["gain_gated_pct"],
                         "w_gain_force_pct": wqe["gain_force_pct"]})
    print(f"[{model_name}] {len(rows)} weight layers analyzed", flush=True)

    # 2) Activation hooks.
    act_acc = {r["layer"]: [] for r in rows}
    handles = []
    for r in rows:
        m = name_to_mod[r["layer"]]
        is_conv = isinstance(m, nn.Conv2d)
        def make_hook(nm, is_conv2d):
            def hook(mod, inp, out):
                x = inp[0] if isinstance(inp, tuple) else inp
                xf = x.detach().float()
                # For Conv2d input [B, C, H, W] permute to [B, H, W, C] so each
                # row is a (spatial-position, batch) "token" across C channels;
                # for Linear input [B, T, D] or [B, D], last dim is already the
                # feature dim.
                if is_conv2d and xf.dim() == 4:
                    xf = xf.permute(0, 2, 3, 1).contiguous()
                f = xf.reshape(-1, xf.shape[-1])
                if f.shape[0] > max(64, 200_000 // max(f.shape[-1], 1)):
                    idx = torch.randperm(f.shape[0], device=f.device)[:max(64, 200_000 // max(f.shape[-1], 1))]
                    f = f[idx]
                act_acc[nm].append(f.cpu())
            return hook
        handles.append(m.register_forward_hook(make_hook(r["layer"], is_conv)))

    # 3) Calibration forward.
    print(f"[{model_name}] running calibration forward", flush=True)
    calib_inputs_fn(model, device)
    for h in handles:
        h.remove()

    # 4) Activation analysis.
    for r in rows:
        chunks = act_acc[r["layer"]]
        if not chunks:
            r.update({"a_kurt": None, "a_skew": None, "a_frac3": None, "a_frac4": None,
                      "a_gate": None, "a_rtn_mse": None, "a_dbaf_gated_mse": None,
                      "a_dbaf_force_mse": None, "a_gain_gated_pct": None, "a_gain_force_pct": None})
            continue
        t = torch.cat(chunks)
        # Trim if huge.
        if t.numel() > 500_000:
            idx = torch.randperm(t.shape[0])[:max(64, 500_000 // t.shape[-1])]
            t = t[idx]
        afp = tensor_fingerprint(t)
        aqe = activation_quant_errors(t)
        r.update({"a_kurt": afp["kurt"], "a_skew": afp["skew"],
                  "a_frac3": afp["frac3"], "a_frac4": afp["frac4"],
                  "a_gate": afp["dbaf_gate"],
                  "a_rtn_mse": aqe["rtn_mse"], "a_rtn_nmse": aqe["rtn_nmse"],
                  "a_dbaf_gated_mse": aqe["dbaf_gated_mse"], "a_dbaf_gated_nmse": aqe["dbaf_gated_nmse"],
                  "a_dbaf_force_mse": aqe["dbaf_force_mse"], "a_dbaf_force_nmse": aqe["dbaf_force_nmse"],
                  "a_gain_gated_pct": aqe["gain_gated_pct"],
                  "a_gain_force_pct": aqe["gain_force_pct"]})

    # 5) Aggregate.
    def safe_mean(vals): vals = [v for v in vals if v is not None]; return float(np.mean(vals)) if vals else None
    def pct_true(vals): vals = [v for v in vals if v is not None]; return float(np.mean(vals)) if vals else None

    n_obs_act = sum(1 for r in rows if r["a_gate"] is not None)
    summary = {
        "model": model_name,
        "n_layers": len(rows),
        "n_act_layers_observed": n_obs_act,
        # Weight aggregates
        "w_pct_gated": pct_true([r["w_gate"] for r in rows]),
        "w_mean_kurt": safe_mean([r["w_kurt"] for r in rows]),
        "w_mean_frac3": safe_mean([r["w_frac3"] for r in rows]),
        "w_mean_rtn_mse": safe_mean([r["w_rtn_mse"] for r in rows]),
        "w_mean_dbaf_gated_mse": safe_mean([r["w_dbaf_gated_mse"] for r in rows]),
        "w_mean_dbaf_force_mse": safe_mean([r["w_dbaf_force_mse"] for r in rows]),
        "w_mean_gain_gated_pct": safe_mean([r["w_gain_gated_pct"] for r in rows]),
        "w_mean_gain_force_pct": safe_mean([r["w_gain_force_pct"] for r in rows]),
        "w_mean_rtn_mse_when_gated":   safe_mean([r["w_rtn_mse"] for r in rows if r["w_gate"]]),
        "w_mean_rtn_mse_when_notgated":safe_mean([r["w_rtn_mse"] for r in rows if not r["w_gate"]]),
        "w_mean_gain_gated_when_gated":    safe_mean([r["w_gain_gated_pct"] for r in rows if r["w_gate"]]),
        "w_mean_gain_gated_when_notgated": safe_mean([r["w_gain_gated_pct"] for r in rows if not r["w_gate"]]),
        "w_mean_gain_force_when_gated":    safe_mean([r["w_gain_force_pct"] for r in rows if r["w_gate"]]),
        "w_mean_gain_force_when_notgated": safe_mean([r["w_gain_force_pct"] for r in rows if not r["w_gate"]]),
        # Activation aggregates (only over observed layers)
        "a_pct_gated": pct_true([r["a_gate"] for r in rows]),
        "a_mean_kurt": safe_mean([r["a_kurt"] for r in rows]),
        "a_mean_frac3": safe_mean([r["a_frac3"] for r in rows]),
        "a_mean_rtn_mse": safe_mean([r["a_rtn_mse"] for r in rows]),
        "a_mean_dbaf_gated_mse": safe_mean([r["a_dbaf_gated_mse"] for r in rows]),
        "a_mean_dbaf_force_mse": safe_mean([r["a_dbaf_force_mse"] for r in rows]),
        "a_mean_gain_gated_pct": safe_mean([r["a_gain_gated_pct"] for r in rows]),
        "a_mean_gain_force_pct": safe_mean([r["a_gain_force_pct"] for r in rows]),
        "a_mean_rtn_mse_when_gated":    safe_mean([r["a_rtn_mse"] for r in rows if r["a_gate"]]),
        "a_mean_rtn_mse_when_notgated": safe_mean([r["a_rtn_mse"] for r in rows if r["a_gate"] is False]),
        "a_mean_gain_gated_when_gated":       safe_mean([r["a_gain_gated_pct"] for r in rows if r["a_gate"]]),
        "a_mean_gain_gated_when_notgated":    safe_mean([r["a_gain_gated_pct"] for r in rows if r["a_gate"] is False]),
        "a_mean_gain_force_when_gated":       safe_mean([r["a_gain_force_pct"] for r in rows if r["a_gate"]]),
        "a_mean_gain_force_when_notgated":    safe_mean([r["a_gain_force_pct"] for r in rows if r["a_gate"] is False]),
        "wallclock": time.time() - t0,
    }
    outdir = pathlib.Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{model_name}.json").write_text(json.dumps({"summary": summary, "layers": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    del model; gc.collect(); torch.cuda.empty_cache()
    return summary


# ----------- Calibration input functions per model family -----------
def llm_calib_fn(model, device):
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok_path = getattr(model.config, "_name_or_path", None) or model.config._name_or_path
    tok = AutoTokenizer.from_pretrained(tok_path)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"][:200])
    ids = tok(text, return_tensors="pt").input_ids.to(device)[:, :1024]
    with torch.no_grad():
        _ = model(ids)


def sam_calib_fn():
    def f(model, device):
        imgs = sorted(glob.glob("/home/ubuntu/unifying-ptq/data/sr_testsets/Set5_HR/*.png"))[:2]
        with torch.no_grad():
            for ip in imgs:
                arr = np.array(Image.open(ip).convert("RGB"))
                arr2 = np.array(Image.fromarray(arr).resize((1024, 1024), Image.BILINEAR))
                # SAM ViT expects [B, 3, 1024, 1024] in pixel space; the encoder
                # subtracts its own pixel mean/std internally.
                x = torch.from_numpy(arr2).permute(2, 0, 1).float().unsqueeze(0).to(device)
                _ = model(x)
    return f


import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("rtfs", "/home/ubuntu/unifying-ptq/scripts/run_training_free_swinir.py")
_rtfs = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_rtfs)
def swinir_calib_fn(scale):
    def f(model, device):
        imgs = sorted(glob.glob("/home/ubuntu/unifying-ptq/data/sr_testsets/Set5_HR/*.png"))[:2]
        with torch.no_grad():
            for ip in imgs:
                hr = np.array(Image.open(ip).convert("RGB"))
                h, w = hr.shape[:2]
                h -= h % scale; w -= w % scale
                hr = hr[:h, :w]
                lr = np.array(Image.fromarray(hr).resize((w // scale, h // scale), Image.BICUBIC))
                lh, lw = lr.shape[:2]
                lh -= lh % 8; lw -= lw % 8
                lr = lr[:lh, :lw]
                x = torch.from_numpy(lr).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                _ = model(x).clamp(0, 1)
    return f


# ----------- Loaders -----------
def llama_loader():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        "/data/modelzoo/meta-llama/Meta-Llama-3-8B",
        torch_dtype=torch.float16, low_cpu_mem_usage=True)


def qwen_loader():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        "/data/modelzoo/Qwen/Qwen2.5-7B",
        torch_dtype=torch.float16, low_cpu_mem_usage=True)


def sam_loader(variant):
    from segment_anything import sam_model_registry
    ck = f"/home/ubuntu/unifying-ptq/ckpt/sam_vit_{variant}_" + {
        "b": "01ec64", "l": "0b3195", "h": "4b8939",
    }[variant] + ".pth"
    sam = sam_model_registry[f"vit_{variant}"](checkpoint=ck)
    return sam.image_encoder  # analyze only image encoder


def swinir_loader(scale):
    def f():
        return _rtfs.load_swinir(scale,
            f"/home/ubuntu/unifying-ptq/ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{scale}.pth")
    return f


# ----------- main -----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset to run: any of llama qwen sam-b sam-l sam-h swinir-x2 swinir-x3 swinir-x4 (default: all)")
    p.add_argument("--out", default="/home/ubuntu/unifying-ptq/results/S4-cross-model-layer-analysis")
    args = p.parse_args()

    ALL = {
        "swinir-x2": (swinir_loader(2), swinir_calib_fn(2)),
        "swinir-x3": (swinir_loader(3), swinir_calib_fn(3)),
        "swinir-x4": (swinir_loader(4), swinir_calib_fn(4)),
        "sam-b": (lambda: sam_loader("b"), sam_calib_fn()),
        "sam-l": (lambda: sam_loader("l"), sam_calib_fn()),
        "sam-h": (lambda: sam_loader("h"), sam_calib_fn()),
        "llama": (llama_loader, llm_calib_fn),
        "qwen": (qwen_loader, llm_calib_fn),
    }
    if args.models:
        items = [(k, ALL[k]) for k in args.models if k in ALL]
    else:
        items = list(ALL.items())
    summaries = []
    for name, (loader, calib) in items:
        try:
            s = analyze_model(name, loader, calib, args.out)
            summaries.append(s)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{name}] FAILED: {e}", flush=True)
    # Write combined table.
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out, "summary.json").write_text(json.dumps(summaries, indent=2))
    # Pretty print.
    cols = ["model", "n_layers",
            "w_pct_gated", "w_mean_gain_gated_pct", "w_mean_gain_force_pct",
            "w_mean_gain_force_when_gated", "w_mean_gain_force_when_notgated",
            "a_pct_gated", "a_mean_gain_gated_pct", "a_mean_gain_force_pct",
            "a_mean_gain_force_when_gated", "a_mean_gain_force_when_notgated"]
    print("\n=== SUMMARY ACROSS MODELS ===")
    print(" | ".join(f"{c:>26}" for c in cols))
    for s in summaries:
        cells = []
        for c in cols:
            v = s.get(c)
            if isinstance(v, float):
                cells.append(f"{v:>26.4f}")
            else:
                cells.append(f"{str(v):>26}")
        print(" | ".join(cells))


if __name__ == "__main__":
    main()
