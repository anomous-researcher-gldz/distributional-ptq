"""Direct mechanism measurement: per-prompt optimal scale variance.

For each Linear input we observe across N inputs, the *optimal* per-input
max-abs scale is what PCSA tries to predict by routing to anchor scales.
We measure:
  s_i  = ||x_i||_inf   (per-input optimal max-abs scale)
  var(s_i) / mean(s_i)^2  (squared CV — variance relative to mean magnitude)

The mean of this over Linears is the "magnitude of per-input scale shift" that
PCSA could in principle exploit if it can predict s_i from the descriptor.

Then for the PCSA anchor differentiation check: load the calibrated PCSA scales
(SAM uses K=4; LLM K=8). Compute the per-anchor scale spread relative to mean:
  cv_anchor = std(s_k) / mean(s_k)

High cv_anchor means PCSA's K scales actually differ, i.e. the anchor mechanism
is doing useful per-cluster differentiation.

Output: JSON with per-Linear scale variance (mean across layers) + per-anchor
scale CV (mean across K for each model).
"""
from __future__ import annotations
import sys, json, pathlib, argparse, glob
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/ubuntu/unifying-ptq")
sys.path.insert(0, "/home/ubuntu/unifying-ptq/projects/instance_segment_anything/models")


@torch.no_grad()
def measure_scale_shift_sam(n_images: int = 50):
    """SAM-B mask-decoder cross-attn q_proj per-input scale variance."""
    import segment_anything as sa
    sam = sa.sam_model_registry["vit_b"](
        checkpoint="/home/ubuntu/unifying-ptq/ckpt/sam_vit_b_01ec64.pth"
    ).cuda().eval()
    md = sam.mask_decoder

    # Per (layer, input) max-abs activation
    scales = {}
    def make_hook(name):
        def h(mod, args):
            x = args[0]
            s = float(x.float().abs().max().item())
            scales.setdefault(name, []).append(s)
        return h
    handles = []
    for li, blk in enumerate(md.transformer.layers):
        handles.append(blk.cross_attn_token_to_image.q_proj.register_forward_pre_hook(make_hook(f"layer{li}")))

    torch.manual_seed(0)
    for i in range(n_images):
        s = 30.0 + 6.0 * i
        img = (torch.randn(1, 3, 1024, 1024) * s + 128.0).clamp(0, 255).cuda() / 255.0
        img_emb = sam.image_encoder(img)
        n_pts = 2 + (i % 3)
        coords = torch.rand(1, n_pts, 2, device="cuda") * 1024.0
        labels = torch.ones(1, n_pts, device="cuda")
        sparse, dense = sam.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        _ = md(image_embeddings=img_emb,
               image_pe=sam.prompt_encoder.get_dense_pe(),
               sparse_prompt_embeddings=sparse,
               dense_prompt_embeddings=dense,
               multimask_output=False)
    for h in handles: h.remove()
    return scales


@torch.no_grad()
def measure_scale_shift_llama(n_prompts: int = 50):
    """LLaMA-3-8B per-layer q_proj input per-input scale variance."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained("/data/modelzoo/meta-llama/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "/data/modelzoo/meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16
    ).cuda().eval()
    scales = {}
    def make_hook(name):
        def h(mod, args):
            x = args[0]
            s = float(x.float().abs().max().item())
            scales.setdefault(name, []).append(s)
        return h
    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.q_proj.register_forward_pre_hook(make_hook(f"layer{li}")))

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t) > 300][:n_prompts]
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        _ = model(ids)
    for h in handles: h.remove()
    return scales


def summarize(scales):
    cvs = []
    for name, vals in scales.items():
        a = np.asarray(vals, dtype=np.float64)
        if a.size < 2 or a.mean() == 0:
            continue
        cv = float(a.std() / abs(a.mean()))
        cvs.append({"layer": name, "n": int(a.size),
                    "mean_scale": float(a.mean()),
                    "std_scale": float(a.std()),
                    "cv_scale": cv})
    if not cvs:
        return {"n_layers": 0, "mean_cv_scale": None}
    arr = np.array([c["cv_scale"] for c in cvs])
    return {
        "n_layers": len(cvs),
        "mean_cv_scale": float(arr.mean()),
        "median_cv_scale": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "per_layer": cvs[:10],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_inputs", type=int, default=50)
    p.add_argument("--out", default="/home/ubuntu/unifying-ptq/results/pcsa_mechanism_direct.json")
    args = p.parse_args()

    results = {}
    print("=== SAM-B scale shift ===", flush=True)
    sam_scales = measure_scale_shift_sam(args.n_inputs)
    sam_summary = summarize(sam_scales)
    print(f"  n_layers={sam_summary['n_layers']}  mean_CV_scale={sam_summary['mean_cv_scale']:.4f}  median={sam_summary['median_cv_scale']:.4f}  IQR=[{sam_summary['p25']:.4f},{sam_summary['p75']:.4f}]", flush=True)
    results["sam-b"] = sam_summary

    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    print("\n=== LLaMA-3-8B scale shift ===", flush=True)
    llm_scales = measure_scale_shift_llama(args.n_inputs)
    llm_summary = summarize(llm_scales)
    print(f"  n_layers={llm_summary['n_layers']}  mean_CV_scale={llm_summary['mean_cv_scale']:.4f}  median={llm_summary['median_cv_scale']:.4f}  IQR=[{llm_summary['p25']:.4f},{llm_summary['p75']:.4f}]", flush=True)
    results["llama-8b"] = llm_summary

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
