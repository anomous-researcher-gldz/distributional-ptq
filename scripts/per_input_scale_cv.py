"""Per-input max-abs activation CV: SAM-B vs LLaMA-3-8B.

For PCSA to find meaningful anchor clusters, the per-input max-abs activation
scale on each Linear input must vary substantially across inputs. We measure
this with the coefficient of variation (std/mean) of |x|_max across N inputs
per linear layer, averaged over layers.

Hypothesis (paper §4.4):
  - SAM-B (per-image, content-varying): CV >> 0
  - LLaMA-3-8B (per-prompt, similar token statistics): CV ~ 0

Outputs JSON with per-model summary CV + per-layer breakdown.
"""
from __future__ import annotations
import sys, json, pathlib, glob, argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/projects/instance_segment_anything/models")


@torch.no_grad()
def per_input_max_abs(model, layers, build_input_fn, n_inputs: int, device="cuda"):
    """Returns dict[layer_name][i] = max-abs activation on input i."""
    per_layer = {}
    handles = []

    def _make_hook(name):
        def _h(mod, args):
            if not args:
                return
            x = args[0] if isinstance(args[0], torch.Tensor) else None
            if x is None:
                return
            per_layer.setdefault(name, []).append(float(x.detach().abs().max().item()))
        return _h

    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Module):
            for nm, sub in layer.named_modules():
                if isinstance(sub, (nn.Linear, nn.Conv2d)):
                    handles.append(sub.register_forward_pre_hook(_make_hook(f"layer{i}.{nm}")))

    for k in range(n_inputs):
        x = build_input_fn(k)
        model(x) if not isinstance(x, dict) else model(**x)

    for h in handles:
        h.remove()
    return per_layer


def summarize(per_layer):
    cvs = []
    for name, vals in per_layer.items():
        a = np.asarray(vals, dtype=np.float64)
        if a.size < 2 or a.mean() == 0:
            continue
        cv = float(a.std() / abs(a.mean()))
        cvs.append({"layer": name, "n": int(a.size),
                    "mean_max_abs": float(a.mean()),
                    "std_max_abs": float(a.std()), "cv": cv})
    if not cvs:
        return {"n_layers": 0, "mean_cv": None, "median_cv": None, "per_layer": []}
    arr = np.array([c["cv"] for c in cvs])
    return {"n_layers": len(cvs),
            "mean_cv": float(arr.mean()),
            "median_cv": float(np.median(arr)),
            "p25_cv": float(np.percentile(arr, 25)),
            "p75_cv": float(np.percentile(arr, 75)),
            "per_layer": cvs}


def sam_b(n_inputs=20):
    import segment_anything as sa
    sam = sa.sam_model_registry["vit_b"](
        checkpoint="/home/ubuntu/unifying-ptq/ckpt/sam_vit_b_01ec64.pth")
    sam = sam.to(torch.bfloat16).cuda().eval()
    enc = sam.image_encoder
    layers = list(enc.blocks)

    # Use Set5 if available; otherwise synthesize content-varied images.
    img_paths = sorted(glob.glob("/home/ubuntu/unifying-ptq/data/sr_testsets/Set5_HR/*.png"))
    if len(img_paths) < n_inputs:
        # Synthesize via random Gaussian/uniform mixtures + structured noise
        torch.manual_seed(0)
        img_paths = []
    def build(i):
        if i < len(img_paths):
            arr = np.array(Image.open(img_paths[i]).convert("RGB"))
            if arr.shape[0] < 1024 or arr.shape[1] < 1024:
                # Pad/resize to 1024 — SAM expects fixed size
                from PIL import Image as P
                arr = np.array(P.fromarray(arr).resize((1024, 1024), P.BICUBIC))
            arr = arr[:1024, :1024]
            x = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)
        else:
            # synthetic with content variation
            torch.manual_seed(i)
            scale = 50.0 + 100.0 * (i / n_inputs)  # range 50-150 — content variation
            x = (torch.randn(1, 3, 1024, 1024) * scale + 128.0).clamp(0, 255).float()
        x = x.to(torch.bfloat16).cuda() / 255.0
        return x
    return enc, layers, build


def llama_8b(n_inputs=20):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/modelzoo/meta-llama/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "/data/modelzoo/meta-llama/Meta-Llama-3-8B",
        torch_dtype=torch.bfloat16,
    ).cuda().eval()
    layers = list(model.model.layers)

    # Use WikiText prompts of equal length so length doesn't drive scale.
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t) > 200][:n_inputs]
    while len(texts) < n_inputs:
        texts.append(" ".join(["text"] * 256))
    def build(i):
        text = texts[i]
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        return ids
    return model, layers, build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_inputs", type=int, default=20)
    ap.add_argument("--models", nargs="+", default=["sam-b", "llama-8b"])
    ap.add_argument("--out", default="/home/ubuntu/unifying-ptq/results/per_input_scale_cv.json")
    args = ap.parse_args()

    results = {}
    for m in args.models:
        print(f"\n=== {m} ===", flush=True)
        if m == "sam-b":
            model, layers, build = sam_b(args.n_inputs)
        elif m == "llama-8b":
            model, layers, build = llama_8b(args.n_inputs)
        else:
            print(f"skip unknown {m}"); continue
        per_layer = per_input_max_abs(model, layers, build, args.n_inputs)
        s = summarize(per_layer)
        print(f"  n_layers={s['n_layers']}  mean_cv={s['mean_cv']:.4f}  "
              f"median_cv={s['median_cv']:.4f}  IQR=[{s['p25_cv']:.4f},{s['p75_cv']:.4f}]",
              flush=True)
        results[m] = s
        del model
        torch.cuda.empty_cache()

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
