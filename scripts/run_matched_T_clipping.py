"""Matched-T clipping baseline: clamp |w| to ±T=3σ per row, then RTN.

Compared to DBAF: both reduce dynamic range. DBAF *preserves* outlier ordering
(via affine fold); matched-T clipping discards it (hard clamp).

If RTN+DBAF beats RTN+clipping at the same T, this isolates the value of
folding (preserving outlier order) from the value of dynamic-range reduction.
"""
import argparse, json, pathlib, sys, time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
from scripts.run_S4 import wikitext_ppl


@torch.no_grad()
def quantize_clipping(model: nn.Module, bits: int = 4, T_sigma: float = 3.0):
    qmax = 2 ** (bits - 1) - 1
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "lm_head" in name:
            continue
        w = mod.weight.data
        sigma = w.std(dim=1, keepdim=True)
        T = T_sigma * sigma
        # Hard clamp (matched-T clipping)
        w_clip = torch.clamp(w, min=-T, max=T)
        scale = w_clip.abs().amax(dim=1, keepdim=True) / qmax
        scale = scale.clamp(min=1e-9)
        q = torch.round(w_clip / scale).clamp(-qmax, qmax)
        mod.weight.data = (q * scale).to(w.dtype)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/data/modelzoo/meta-llama/Meta-Llama-3-8B")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--T-sigma", type=float, default=3.0)
    p.add_argument("--out", required=True)
    p.add_argument("--ppl-samples", type=int, default=64)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    t0 = time.time()
    model = quantize_clipping(model, bits=args.bits, T_sigma=args.T_sigma)
    t_quant = time.time() - t0

    ppl = wikitext_ppl(model, tok, n_samples=args.ppl_samples)
    out = {
        "model": "llama3-8b",
        "method": f"RTN+matched-T-clipping({args.T_sigma}sigma)",
        "bits": args.bits,
        "wikitext2_ppl": ppl,
        "quant_seconds": t_quant,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
