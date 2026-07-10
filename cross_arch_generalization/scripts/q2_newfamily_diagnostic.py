"""Q2 / W2 -- does the paper's outlier gate behave as claimed on architecture
families NOT in the paper (diffusion transformer, multimodal, speech)?

We run ONLY the diagnostic (the paper's gate function, copied verbatim from
ahcptq/quantization/fake_quant.py) -- no quantization, no retraining. For every
Linear/Conv weight tensor we compute the weight-side gate (input-independent,
the rigorous claim: Table outlier_stats reports 94-100% weight gate-pass on the
paper's 8 configs). For activations we hook the same modules over a short
forward pass with model-appropriate inputs and report the activation-side gate
rate too.

Families: DiT-XL-2 (diffusion transformer), CLIP-ViT-L/14 (multimodal
vision+text), Whisper-small (speech encoder-decoder).
"""
import sys, json, math, warnings
import numpy as np
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- gate functions copied VERBATIM from the paper's repo ----------
def profile_with_3sigma_outliers(x, eps=1e-8):
    flat = x.detach().float().reshape(-1)
    n = flat.numel()
    if n == 0:
        return None
    if n > 3_000_000:                      # subsample huge tensors for speed
        idx = torch.randperm(n, device=flat.device)[:3_000_000]
        flat = flat[idx]
    mean = flat.mean(); std = flat.std().clamp_min(eps)
    z = (flat - mean) / std
    return {"n": int(n), "skew": (z**3).mean().item(),
            "kurtosis": (z**4).mean().item(),
            "frac_out_3": (z.abs() > 3.0).float().mean().item()}

def gate(x, skew_thresh=0.7, kurt_min=3.0, kurt_max=30.0,
         frac3_min=1e-4, frac3_max=2e-2):
    s = profile_with_3sigma_outliers(x)
    if s is None:
        return None, None
    ok = (abs(s["skew"]) <= skew_thresh and kurt_min <= s["kurtosis"] <= kurt_max
          and frac3_min <= s["frac_out_3"] <= frac3_max)
    return bool(ok), s

# ---------- generic analyzer ----------
def analyze(model, forward_fn, name, max_layers=400):
    mods = [(n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d))
            and hasattr(m, "weight") and m.weight is not None][:max_layers]
    # weight side
    w_gates, w_stats = [], []
    for n, m in mods:
        w = m.weight.data
        w2d = w.reshape(w.shape[0], -1) if w.dim() > 1 else w.reshape(1, -1)
        g, s = gate(w2d)
        if g is not None:
            w_gates.append(g); w_stats.append(s)
    # activation side via hooks
    act = {}
    def mk(nm):
        def h(mod, inp, out):
            if isinstance(inp, tuple) and len(inp) and torch.is_tensor(inp[0]):
                act.setdefault(nm, inp[0].detach().float().reshape(-1).cpu())
        return h
    handles = [m.register_forward_hook(mk(n)) for n, m in mods]
    try:
        forward_fn(model)
    except Exception as e:
        print(f"  [{name}] forward failed: {type(e).__name__}: {e}")
    for hd in handles:
        hd.remove()
    a_gates = []
    for n, m in mods:
        if n in act and act[n].numel() > 100:
            g, _ = gate(act[n].unsqueeze(0))
            if g is not None:
                a_gates.append(g)
    res = {
        "model": name,
        "n_weight_tensors": len(w_gates),
        "weight_gate_pass_pct": round(100 * np.mean(w_gates), 1) if w_gates else None,
        "mean_weight_kurt": round(float(np.mean([s["kurtosis"] for s in w_stats])), 2) if w_stats else None,
        "mean_weight_frac3": round(float(np.mean([s["frac_out_3"] for s in w_stats])), 5) if w_stats else None,
        "n_act_tensors": len(a_gates),
        "act_gate_pass_pct": round(100 * np.mean(a_gates), 1) if a_gates else None,
    }
    print(f"  [{name}] weight gate-pass {res['weight_gate_pass_pct']}% "
          f"(n={res['n_weight_tensors']}, mean kurt {res['mean_weight_kurt']}, "
          f"frac3 {res['mean_weight_frac3']}); "
          f"act gate-pass {res['act_gate_pass_pct']}% (n={res['n_act_tensors']})")
    return res

results = []

# ================= 1) DiT-XL-2 (diffusion transformer) =================
def run_dit():
    from diffusers import DiTTransformer2DModel
    t = DiTTransformer2DModel.from_pretrained(
        "facebook/DiT-XL-2-256", subfolder="transformer",
        torch_dtype=torch.float32).to(DEV).eval()
    def fwd(model):
        # DiT denoises 32x32x4 latents; feed a realistic mid-schedule latent
        lat = torch.randn(2, 4, 32, 32, device=DEV)
        ts = torch.tensor([500, 500], device=DEV)
        cls = torch.tensor([207, 360], device=DEV)   # two ImageNet classes
        model(lat, timestep=ts, class_labels=cls)
    return analyze(t, fwd, "DiT-XL-2 (diffusion transformer)")

# ================= 2) CLIP-ViT-L/14 (multimodal) =================
def run_clip():
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                  torch_dtype=torch.float32).to(DEV).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    from PIL import Image
    imgs = [Image.fromarray((np.random.default_rng(i).integers(0,255,(224,224,3))
            ).astype(np.uint8)) for i in range(3)]
    txt = ["a photo of a cat", "a diagram of a transformer", "an aerial city view"]
    def fwd(model):
        b = proc(text=txt, images=imgs, return_tensors="pt", padding=True)
        b = {k: v.to(DEV) for k, v in b.items()}
        model(**b)
    return analyze(m, fwd, "CLIP-ViT-L/14 (multimodal)")

# ================= 3) Whisper-small (speech) =================
def run_whisper():
    from transformers import WhisperForConditionalGeneration
    m = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small", torch_dtype=torch.float32).to(DEV).eval()
    def fwd(model):
        # 80-mel x 3000 frames log-mel; feed a plausible speech-like spectrogram
        feats = torch.randn(1, 80, 3000, device=DEV) * 0.5 - 0.5
        dec = torch.tensor([[model.config.decoder_start_token_id]], device=DEV)
        model(input_features=feats, decoder_input_ids=dec)
    return analyze(m, fwd, "Whisper-small (speech)")

for fn, tag in [(run_dit, "DiT"), (run_clip, "CLIP"), (run_whisper, "Whisper")]:
    print(f"\n=== {tag} ===")
    try:
        results.append(fn())
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  {tag} FAILED: {e}")

out = "/tmp/claude-1000/-home-ubuntu/629c24a1-ecff-4b3c-8a83-a9277305528e/scratchpad/q2_results.json"
json.dump(results, open(out, "w"), indent=2)
print(f"\nSaved -> {out}")
print("\nFor reference, paper's in-scope weight gate-pass: 94-100% across 8 configs.")
