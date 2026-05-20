"""INT4 end-to-end latency: baseline INT4 vs +DBAF, multi-model.

Extends Table~\\ref{tab:int4} (LLaMA-3-8B via FlatQuant kernels) to
Qwen-2.5-7B, SAM-B, and SwinIR-light.

For each model:
  1. Load in FP16 (FP32 for SwinIR; its attention softmax mixes dtypes
     under .half() and crashes).
  2. Apply torchao W4A16 (Int4WeightOnlyConfig, group_size=128) on every
     nn.Linear except lm_head. W4A16 = INT4 weights, FP16 activations,
     INT4 dequant + FP16 matmul. Standard real-INT4 production setup
     for non-fused servers (AWQ, GPTQ defaults).
  3. Time forward: W4A16 baseline ms.
  4. Attach forward_pre_hooks that DBAF-fold the layer input at the
     paper's recommended alpha=0.25, T=3 sigma.
  5. Time again: W4A16 + DBAF.
  6. Report absolute ms + % overhead.

Why W4A16 not W4A4: torchao's W4A4 path uses cutlass kernels with row-tile
size assumptions that mismatch SAM-B's 768-dim QKV (cutlass error at
runtime). W4A16 is universally supported and is the configuration most
production servers actually deploy for sub-8B / non-LLM models.

LLaMA-3-8B INT4 numbers in Table~\\ref{tab:int4} are kept separately
(FlatQuant fused INT4 kernels); this script's job is to add Qwen, SAM, SR.

Output: scripts/_out/int4_e2e_latency_multimodel.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "FlatQuant"))

import torch
import torch.nn as nn


def _dbaf_hook_factory(alpha: float, T_sigma: float):
    def hook(module, inputs):
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return inputs
        x = inputs[0]
        sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-9)
        T = T_sigma * sigma
        sgn = torch.sign(x)
        mask = x.abs() > T
        x_folded = torch.where(mask, sgn * T + alpha * (x - sgn * T), x)
        return (x_folded,) + inputs[1:]
    return hook


def _attach_dbaf(layers, alpha, T_sigma):
    h = _dbaf_hook_factory(alpha, T_sigma)
    return [layer.register_forward_pre_hook(h) for layer in layers]


def _detach(handles):
    for h in handles:
        h.remove()


@torch.no_grad()
def _time_forward(fwd, build_input, n_warmup: int, n_iters: int) -> float:
    inp = build_input()
    for _ in range(n_warmup):
        _ = fwd(inp)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = fwd(inp)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / n_iters


def _apply_w4a16(model, label: str):
    """torchao W4A16 (Int4WeightOnlyConfig). Group size 128 is safe."""
    from torchao.quantization import quantize_, Int4WeightOnlyConfig

    def _filter_linear(mod, fqn):
        return isinstance(mod, nn.Linear) and "lm_head" not in fqn

    print(f"[{label}] applying torchao W4A16 (group_size=128) ...", flush=True)
    quantize_(model, Int4WeightOnlyConfig(group_size=128), filter_fn=_filter_linear)
    return model


def _model_from_fwd(fwd) -> nn.Module:
    for cell in (fwd.__closure__ or ()):
        cand = cell.cell_contents
        if isinstance(cand, nn.Module):
            return cand
    raise RuntimeError("could not recover model object from fwd closure")


def bench_one(name: str, loader_fn, alpha: float, T_sigma: float,
              n_warmup: int, n_iters: int,
              llm_seq_len: int | None = None) -> dict:
    print(f"\n=== {name} ===", flush=True)
    fwd, layers, d, build_input = loader_fn()
    if llm_seq_len is not None:
        orig = build_input
        build_input = lambda: orig(llm_seq_len)

    # FP precision baseline (precision-invariant cost reference for DBAF)
    t_fp = _time_forward(fwd, build_input, n_warmup, n_iters)
    print(f"  FP baseline:           {t_fp:9.2f} ms/forward", flush=True)

    # Attach DBAF hooks for FP+DBAF measurement
    handles = _attach_dbaf(layers, alpha, T_sigma)
    t_fp_dbaf = _time_forward(fwd, build_input, n_warmup, n_iters)
    _detach(handles)
    print(f"  FP + DBAF:             {t_fp_dbaf:9.2f} ms/forward  "
          f"(+{100.0 * (t_fp_dbaf - t_fp) / t_fp:.2f}%)", flush=True)

    # W4A16 baseline + DBAF
    int4_baseline = int4_dbaf = None
    int4_err = None
    try:
        root = _model_from_fwd(fwd)
        _apply_w4a16(root, name)
        int4_baseline = _time_forward(fwd, build_input, n_warmup, n_iters)
        print(f"  W4A16 baseline:        {int4_baseline:9.2f} ms/forward",
              flush=True)
        handles = _attach_dbaf(layers, alpha, T_sigma)
        int4_dbaf = _time_forward(fwd, build_input, n_warmup, n_iters)
        _detach(handles)
        print(f"  W4A16 + DBAF:          {int4_dbaf:9.2f} ms/forward  "
              f"(+{100.0 * (int4_dbaf - int4_baseline) / int4_baseline:.2f}%)",
              flush=True)
    except Exception as exc:
        int4_err = f"{type(exc).__name__}: {exc}"
        print(f"  WARNING: W4A16 path failed: {int4_err}", flush=True)

    result = {
        "model": name, "d": d, "n_layers": len(layers),
        "fp_baseline_ms": t_fp,
        "fp_dbaf_ms": t_fp_dbaf,
        "fp_dbaf_overhead_pct": 100.0 * (t_fp_dbaf - t_fp) / t_fp,
        "w4a16_baseline_ms": int4_baseline,
        "w4a16_dbaf_ms": int4_dbaf,
        "w4a16_dbaf_overhead_pct": (
            None if int4_baseline is None or int4_dbaf is None
            else 100.0 * (int4_dbaf - int4_baseline) / int4_baseline
        ),
        "w4a16_error": int4_err,
    }
    torch.cuda.empty_cache()
    return result


def _load_swinir_fp32(scale: int):
    """SwinIR-light at FP32 (avoids attention dtype mixing under .half())."""
    sys.path.insert(0, "/home/ubuntu/unifying-ptq/2DQuant")
    from basicsr.archs.swinir_arch import SwinIR
    model = SwinIR(
        upscale=scale, in_chans=3, img_size=64, window_size=8, img_range=1.,
        depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv'
    )
    ckpt_path = (
        f"/home/ubuntu/unifying-ptq/ckpt/swinir/"
        f"002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{scale}.pth"
    )
    sd = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    if "params" in sd:
        sd = sd["params"]
    model.load_state_dict(sd, strict=False)
    model = model.float().cuda().eval()

    layers = []
    for rstb in model.layers:
        layers.extend(list(rstb.residual_group.blocks))
    d = layers[0].dim

    sr_lr_size = {2: 512, 3: 384, 4: 256}[scale]
    def build_input():
        return torch.randn(1, 3, sr_lr_size, sr_lr_size,
                           dtype=torch.float32, device="cuda")
    def fwd(lr):
        with torch.no_grad():
            return model(lr)
    return fwd, layers, d, build_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default="qwen25-7b,sam-b,swinir-x2",
        help="comma-separated subset of {llama3-8b, qwen25-7b, sam-b, sam-l, "
             "sam-h, swinir-x2, swinir-x3, swinir-x4}",
    )
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--T_sigma", type=float, default=3.0)
    ap.add_argument("--n_warmup", type=int, default=3)
    ap.add_argument("--n_iters", type=int, default=5)
    ap.add_argument("--llm_seq_len", type=int, default=2048)
    ap.add_argument("--out", type=pathlib.Path,
                    default=_REPO / "scripts" / "_out" / "int4_e2e_latency_multimodel.json")
    args = ap.parse_args()

    from end_to_end_latency import _load_llm, _load_sam  # type: ignore

    LLM_PATHS = {
        "llama3-8b": "/data/modelzoo/meta-llama/Meta-Llama-3-8B",
        "qwen25-7b": "/data/modelzoo/Qwen/Qwen2.5-7B",
    }
    SAM_VARIANTS = {"sam-b": "b", "sam-l": "l", "sam-h": "h"}
    SR_SCALES = {"swinir-x2": 2, "swinir-x3": 3, "swinir-x4": 4}

    targets = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    for target in targets:
        if target in LLM_PATHS:
            path = LLM_PATHS[target]
            r = bench_one(target, lambda p=path: _load_llm(p),
                          args.alpha, args.T_sigma,
                          args.n_warmup, args.n_iters,
                          llm_seq_len=args.llm_seq_len)
        elif target in SAM_VARIANTS:
            v = SAM_VARIANTS[target]
            r = bench_one(target, lambda v=v: _load_sam(v),
                          args.alpha, args.T_sigma,
                          args.n_warmup, args.n_iters)
        elif target in SR_SCALES:
            s = SR_SCALES[target]
            r = bench_one(target, lambda s=s: _load_swinir_fp32(s),
                          args.alpha, args.T_sigma,
                          args.n_warmup, args.n_iters)
        else:
            print(f"WARNING: unknown target {target}; skipping", flush=True)
            continue
        rows.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": rows, "config": vars(args) | {
        "models": targets,
    }}, indent=2, default=str))
    print(f"\n -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
