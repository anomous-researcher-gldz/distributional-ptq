"""Why does the paper's reconstruction-based alpha selection FAIL on DiT?

Two hypotheses, both measured here:

(H1) Bulk-vs-outlier tension. DBAF sets the quant scale from the max FOLDED
     magnitude, so low alpha -> fine scale -> bulk (|w|<=T) very accurate (low
     GLOBAL MSE) but outliers squashed into a thin band and their error amplified
     by 1/alpha on unfold (high OUTLIER-position MSE). The one-block reconstruction
     loss averages over all positions, ~99% of which are bulk, so it is
     bulk-dominated and picks LOW alpha. Prediction: global-wMSE minimized at low
     alpha, outlier-wMSE minimized at HIGH alpha.

(H2) Trajectory accumulation is the DiT-specific reason. A discriminative model
     (CLIP/Whisper) makes ONE forward pass, so single-pass reconstruction MSE is
     the eval-relevant error and selection works. Diffusion runs T sequential
     passes; the outlier-position error injected each step COMPOUNDS along the
     sampling trajectory and dominates the terminal latent (hence FID), which the
     single-pass bulk-dominated proxy cannot see. Prediction: at low alpha the
     per-step latent error GROWS over steps; at alpha=0.75 it stays bounded, so
     the terminal-trajectory error ordering matches FID, not single-step NMSE.
"""
import sys, copy, json
import os as _os_repo
_REPO_ROOT = _os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
torch.set_grad_enabled(False)
REPO=_REPO_ROOT; sys.path.insert(0,REPO); sys.path.insert(0,REPO+"/FlatQuant")
from flatquant.baselines.rtn import _quantize_per_channel_with_dbaf, _quantize_tensor_uniform
DEV="cuda"
OUT=_REPO_ROOT + "/cross_arch_generalization/results/alpha_dit_diagnosis_results.json"
GRID=[0.25,0.3,0.5,0.75,0.95]
BITS=4

# ---------------- H1: weight-reconstruction bulk vs outlier decomposition ----------------
from diffusers import DiTTransformer2DModel
t=DiTTransformer2DModel.from_pretrained("facebook/DiT-XL-2-256",subfolder="transformer",
    torch_dtype=torch.float32).to(DEV).eval()
orig=copy.deepcopy(t.state_dict())

def dbaf_recon(w,bits,alpha,T_sigma=3.0):
    qmax=2**(bits-1)-1
    sigma=w.std(dim=1,keepdim=True); T=T_sigma*sigma; sgn=torch.sign(w)
    mask=w.abs()>T
    w_fold=torch.where(mask,sgn*T+alpha*(w-sgn*T),w)
    scale=(w_fold.abs().amax(dim=1,keepdim=True)/qmax).clamp(min=1e-9)
    q=torch.round(w_fold/scale).clamp(-qmax,qmax); w_q=q*scale
    sgn_q=torch.sign(w_q); mask_q=w_q.abs()>T
    w_out=torch.where(mask_q,sgn_q*T+(1.0/alpha)*(w_q-sgn_q*T),w_q)
    return w_out, mask

print("[H1] weight recon: bulk vs outlier MSE across alpha", flush=True)
h1={}
lins=[(n,m) for n,m in t.named_modules() if isinstance(m,nn.Linear) and m.weight.dim()==2]
for a in GRID:
    se_bulk=se_out=n_bulk=n_out=0.0
    for _,m in lins:
        w=m.weight.data.float()
        wr,mask=dbaf_recon(w,BITS,a)
        err2=((wr-w)**2)
        se_out+=err2[mask].sum().item();  n_out+=mask.sum().item()
        se_bulk+=err2[~mask].sum().item(); n_bulk+=(~mask).sum().item()
    # normalize each by the energy of its own region for comparability
    h1[str(a)]={"bulk_mse":se_bulk/max(n_bulk,1),"outlier_mse":se_out/max(n_out,1),
                "outlier_frac":n_out/(n_out+n_bulk)}
    print(f"  alpha={a:<5} bulk_MSE={h1[str(a)]['bulk_mse']:.3e}  "
          f"outlier_MSE={h1[str(a)]['outlier_mse']:.3e}", flush=True)
bulk_best=min(GRID,key=lambda a:h1[str(a)]["bulk_mse"])
out_best=min(GRID,key=lambda a:h1[str(a)]["outlier_mse"])
print(f"  -> bulk-MSE minimized at alpha={bulk_best}; outlier-MSE minimized at alpha={out_best}", flush=True)
del t; torch.cuda.empty_cache()

# ---------------- H2: trajectory accumulation over the sampler ----------------
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
pipe=DiTPipeline.from_pretrained("facebook/DiT-XL-2-256",torch_dtype=torch.float32).to(DEV)
pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
torig=copy.deepcopy(pipe.transformer.state_dict())
STEPS=25; NIMG=16
labels=list(range(NIMG))  # fixed classes

def quant(alpha):
    for _,m in pipe.transformer.named_modules():
        if isinstance(m,nn.Linear) and m.weight.dim()==2:
            m.weight.data=_quantize_per_channel_with_dbaf(m.weight.data,BITS,alpha=alpha).to(m.weight.dtype)

def gen_traj():
    traj=[]
    orig_step=pipe.scheduler.step
    def patched(*a,**k):
        out=orig_step(*a,**k)
        ps=out.prev_sample if hasattr(out,"prev_sample") else out[0]
        traj.append(ps.detach().float().cpu().clone()); return out
    pipe.scheduler.step=patched
    try:
        g=torch.Generator(device=DEV).manual_seed(1234)
        pipe(class_labels=labels,num_inference_steps=STEPS,generator=g)
    finally:
        pipe.scheduler.step=orig_step
    return traj  # list ~length STEPS of (N,C,H,W)

print("\n[H2] trajectory: FP reference...", flush=True)
pipe.transformer.load_state_dict(torig); fp_traj=gen_traj()
def nmse(a,b): return float(((a-b)**2).mean()/(b**2).mean())
h2={"steps":STEPS,"per_step_nmse":{}, "single_step_nmse":{}, "terminal_nmse":{}}
for a in [0.25,0.5,0.75]:
    pipe.transformer.load_state_dict(torig); quant(a)
    qt=gen_traj()
    per=[nmse(qt[k],fp_traj[k]) for k in range(STEPS)]
    h2["per_step_nmse"][str(a)]=[round(x,5) for x in per]
    h2["single_step_nmse"][str(a)]=round(per[0],5)      # first step ~ the selection proxy
    h2["terminal_nmse"][str(a)]=round(per[-1],5)         # end of trajectory ~ what FID sees
    print(f"  alpha={a}: step0 NMSE={per[0]:.4f}  ->  terminal NMSE={per[-1]:.4f}  "
          f"(x{per[-1]/max(per[0],1e-9):.1f})", flush=True)
pipe.transformer.load_state_dict(torig)

res={"bits":BITS,"H1_weight_bulk_vs_outlier":h1,
     "H1_bulk_min_alpha":bulk_best,"H1_outlier_min_alpha":out_best,
     "H2_trajectory":h2,
     "known_FID_to_FP":{"rtn":242.8,"alpha0.25":275.1,"alpha0.75":185.7}}
json.dump(res,open(OUT,"w"),indent=2)
print("\nsaved ->",OUT)
