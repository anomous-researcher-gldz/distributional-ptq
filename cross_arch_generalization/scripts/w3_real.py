"""W3 on REAL LLaMA-3-8B weights (CPU, no forward) -- why analytic alpha* fails.

For each gate-pass Linear weight row, sweep alpha and measure the TRUE
fold->INT4-quantize->unfold error, decomposed into GLOBAL MSE (what the proxy
approximates) and OUTLIER-POSITION MSE (the |x|>3sigma entries the alpha^-1
noise concentrates on, and that end-to-end PPL depends on). The proxy/global
MSE bottoms out at small alpha, but outlier-position MSE is minimized at high
alpha and blows up as alpha->0 -- explaining why the analytic optimum
underclips catastrophically while the reconstruction-selected alpha is high.
"""
import sys, json
import numpy as np, torch
torch.set_grad_enabled(False)
REPO="/home/ubuntu/distributional-ptq"; sys.path.insert(0,REPO)
from ahcptq.quantization.fake_quant import is_like_normal_plus_3sigma_outliers
QMAX=7
ALPHAS=[0.02,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.75,0.95]

print("loading LLaMA weights on CPU...", flush=True)
from transformers import AutoModelForCausalLM
m=AutoModelForCausalLM.from_pretrained("NousResearch/Meta-Llama-3-8B",
    torch_dtype=torch.float32, device_map="cpu")
import torch.nn as nn
lins=[(n,mod) for n,mod in m.named_modules() if isinstance(mod,nn.Linear) and "lm_head" not in n]

def fold_quant_unfold(w, alpha):
    # per-row T=3sigma, symmetric int4 per row
    sigma=w.std(dim=1,keepdim=True).clamp_min(1e-9); T=3.0*sigma
    sgn=torch.sign(w); mask=w.abs()>T
    wf=torch.where(mask, sgn*T+alpha*(w-sgn*T), w)
    scale=(wf.abs().amax(dim=1,keepdim=True)/QMAX).clamp_min(1e-9)
    wq=torch.round(wf/scale).clamp(-QMAX,QMAX)*scale
    sgnq=torch.sign(wq); maskq=wq.abs()>T
    wu=torch.where(maskq, sgnq*T+(1.0/alpha)*(wq-sgnq*T), wq)
    err2=(wu-w)**2
    return err2.mean().item(), err2[mask].mean().item() if mask.any() else 0.0, float(mask.float().mean())

# sample gate-pass layers across depth
gate_layers=[]
for n,mod in lins:
    w=mod.weight.data
    if is_like_normal_plus_3sigma_outliers(w)["is_like_c"]:
        gate_layers.append((n,w))
print(f"{len(gate_layers)} gate-pass Linear tensors; sampling 24 across depth", flush=True)
idx=np.linspace(0,len(gate_layers)-1,24).astype(int)
sample=[gate_layers[i] for i in idx]

glob=np.zeros(len(ALPHAS)); outl=np.zeros(len(ALPHAS)); pout=[]
for name,w in sample:
    for ai,a in enumerate(ALPHAS):
        g,o,po=fold_quant_unfold(w,a)
        glob[ai]+=g; outl[ai]+=o
    pout.append(po)
glob/=len(sample); outl/=len(sample)

a_g=ALPHAS[int(np.argmin(glob))]; a_o=ALPHAS[int(np.argmin(outl))]
print(f"\nAvg over {len(sample)} real gate-pass tensors (mean outlier frac {np.mean(pout):.4f}):")
print(f"{'alpha':>6} {'GLOBAL-MSE':>12} {'OUTLIER-MSE':>12} {'out/global':>11}")
for a,g,o in zip(ALPHAS,glob,outl):
    print(f"{a:>6.2f} {g:>12.4g} {o:>12.4g} {o/max(g,1e-12):>11.1f}")
print(f"\nGLOBAL-MSE optimum (what proxy/alpha* tracks): alpha={a_g}")
print(f"OUTLIER-MSE optimum (what PPL tracks):          alpha={a_o}")
print(f"global-MSE penalty at outlier-opt vs its own opt: "
      f"{glob[ALPHAS.index(a_o)]/glob.min():.2f}x (tiny -> proxy can't see it)")
print(f"outlier-MSE penalty at analytic alpha*~0.07 vs outlier-opt: "
      f"{outl[ALPHAS.index(0.07)]/outl.min():.1f}x (huge -> catastrophic underclip)")

out="/tmp/claude-1000/-home-ubuntu/629c24a1-ecff-4b3c-8a83-a9277305528e/scratchpad/w3_real_results.json"
json.dump(dict(alphas=ALPHAS, global_mse=glob.tolist(), outlier_mse=outl.tolist(),
    n_tensors=len(sample), mean_out_frac=float(np.mean(pout)),
    global_opt=a_g, outlier_opt=a_o), open(out,"w"), indent=2)
print(f"\nSaved -> {out}")
