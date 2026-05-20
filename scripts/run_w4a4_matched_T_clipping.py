"""W4A4 matched-T clipping baseline: clamp |w| AND |a| to ±T=3σ per row/token, then RTN.

This is the W4A4 analogue of run_matched_T_clipping.py (which was W4 weight-only).
It establishes that the catastrophic PPL of weight-only clipping at α=0 (PPL=79256)
also holds in the W4A4 setting our paper actually targets, ruling out the
"clipping might work at W4A4 even though it doesn't at W4" objection.

Comparison: RTN+DBAF α=0.25 W4A4 LLaMA-3-8B = 16.31 wt2.
Expectation: matched-T clipping (α=0, hard clamp) at W4A4 should be much worse,
because clipping discards outlier ordering both on weights and activations.
"""
import argparse, json, pathlib, sys, time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/scripts")
from run_S4 import wikitext_ppl


@torch.no_grad()
def quantize_weights_clipping(model, bits=4, T_sigma=3.0):
    """Clip-then-RTN on weights only (per-row)."""
    qmax = 2 ** (bits - 1) - 1
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "lm_head" in name:
            continue
        w = mod.weight.data.float()
        sigma = w.std(dim=1, keepdim=True)
        T = T_sigma * sigma
        w_clip = torch.clamp(w, min=-T, max=T)
        scale = w_clip.abs().amax(dim=1, keepdim=True) / qmax
        scale = scale.clamp(min=1e-9)
        q = torch.round(w_clip / scale).clamp(-qmax, qmax)
        mod.weight.data = (q * scale).to(mod.weight.dtype)


def make_act_clip_hook(bits=4, T_sigma=3.0):
    """Forward pre-hook: clip-then-RTN activations per-token."""
    qmax = 2 ** (bits - 1) - 1
    def hook(mod, args):
        if not args or not isinstance(args[0], torch.Tensor):
            return args
        x = args[0]
        # Per-token (last dim is hidden): compute sigma over hidden dim
        sigma = x.std(dim=-1, keepdim=True)
        T = T_sigma * sigma
        x_clip = torch.clamp(x, min=-T, max=T)
        scale = x_clip.abs().amax(dim=-1, keepdim=True) / qmax
        scale = scale.clamp(min=1e-9)
        q = torch.round(x_clip / scale).clamp(-qmax, qmax)
        x_q = (q * scale).to(x.dtype)
        return (x_q,) + args[1:]
    return hook


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/data/modelzoo/meta-llama/Meta-Llama-3-8B")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--T-sigma", type=float, default=3.0)
    p.add_argument("--out", required=True)
    p.add_argument("--ppl-samples", type=int, default=64)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    print("[w4a4-clip] quantizing weights (clip-then-RTN)...", flush=True)
    t0 = time.time()
    quantize_weights_clipping(model, bits=args.bits, T_sigma=args.T_sigma)

    print("[w4a4-clip] attaching activation clip hooks...", flush=True)
    handles = []
    hook = make_act_clip_hook(bits=args.bits, T_sigma=args.T_sigma)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            handles.append(mod.register_forward_pre_hook(hook))
    t_quant = time.time() - t0
    print(f"[w4a4-clip] {len(handles)} Linears wrapped (W{args.bits}A{args.bits}, T={args.T_sigma}σ)", flush=True)

    print(f"[w4a4-clip] eval wt2 (n={args.ppl_samples})...", flush=True)
    ppl = wikitext_ppl(model, tok, n_samples=args.ppl_samples)

    out = {
        "model": "llama3-8b",
        "method": f"RTN+matched-T-clipping({args.T_sigma}sigma)",
        "bits_w": args.bits,
        "bits_a": args.bits,
        "wikitext2_ppl": float(ppl),
        "quant_seconds": t_quant,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[w4a4-clip] wt2 PPL = {ppl:.3f}", flush=True)
    print(f"[w4a4-clip] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
