"""Recompute closed-form alpha* for the three cross-architecture families.

Weights only -- these hosts are weight-only RTN, so the folded tensor is the
weight. No calibration data and no task evaluation are needed; alpha* is a
function of (T, M, p_out) alone.

Uses astar_common.alpha_star_model, the same code path as the sweep scripts, so
the released number and the released code cannot drift apart. Also reports the
pre-fix value (T taken as 3*std(|w|)) so the delta is explicit.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astar_common

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "alpha_star_recompute.json")
PREV = {"clip": 0.447, "whisper": 0.432, "dit": 0.416}


@torch.no_grad()
def alpha_star_prefix(model, skip=(), n_rows=6):
    """The pre-fix computation: .abs() before .std(), i.e. T ~ 1.8 sigma."""
    vals = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or mod.weight.dim() != 2:
            continue
        if any(s in name for s in skip):
            continue
        w = mod.weight.data.float()
        for r in range(0, w.shape[0], max(1, w.shape[0] // n_rows)):
            row = w[r].abs()
            T = 3 * row.std()
            M = torch.quantile(row, 0.999)
            p_out = (row > T).float().mean()
            p_in = 1 - p_out
            if p_out > 0 and M > T:
                a = torch.pow(T * p_out / ((M - T) * p_in + 1e-12), 1 / 3.0)
                if torch.isfinite(a):
                    vals.append(a.item())
    return float(np.median(vals))


def load(which):
    if which == "clip":
        from transformers import CLIPModel
        return CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                         torch_dtype=torch.float32).eval(), (), 6
    if which == "whisper":
        from transformers import WhisperForConditionalGeneration
        return WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small", torch_dtype=torch.float32).eval(), ("proj_out",), 6
    if which == "dit":
        from diffusers import DiTTransformer2DModel
        return DiTTransformer2DModel.from_pretrained(
            "facebook/DiT-XL-2-256", subfolder="transformer",
            torch_dtype=torch.float32).eval(), (), 8
    raise ValueError(which)


res = {}
for which in sys.argv[1:] or ["clip", "whisper", "dit"]:
    model, skip, n_rows = load(which)
    new, n = astar_common.alpha_star_model(model, skip=skip, n_rows=n_rows)
    old = alpha_star_prefix(model, skip=skip, n_rows=n_rows)
    res[which] = {"alpha_star": round(new, 3), "alpha_star_prefix_bug": round(old, 3),
                  "committed_value": PREV[which], "n_row_samples": n}
    print(f"{which:8s} corrected={new:.3f}  pre-fix={old:.3f}  committed={PREV[which]}", flush=True)
    del model
    torch.cuda.empty_cache()

if len(res) == 3:
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nsaved ->", OUT)
print(json.dumps(res, indent=1))
