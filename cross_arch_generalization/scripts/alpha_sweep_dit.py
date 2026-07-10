"""One-block reconstruction sweep for DBAF alpha on DiT-XL (paper's selection rule).

Paper (S3): "We pick alpha from {alpha*, 0.25, 0.3, 0.5, 0.75, 0.95, 0.99} by
one-block reconstruction loss." We replicate that rule here for DiT and report
which alpha the reconstruction loss selects, alongside the per-step NMSE-vs-FP
(the reconstruction proxy) so we can see whether the proxy aligns with the FID
task metric (which for diffusion is known to diverge from MSE).

alpha* is the closed-form lower bound (Eq. alpha-star), computed per row from the
same T=3sigma, M=percentile_99.9, p_out=Pr(|x|>T) statistics and aggregated
(median) across Linear weights.
"""
import sys, copy, json
import numpy as np, torch, torch.nn as nn
torch.set_grad_enabled(False)
REPO="/home/ubuntu/distributional-ptq"; sys.path.insert(0,REPO); sys.path.insert(0,REPO+"/FlatQuant")
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf
DEV="cuda"
OUT="/home/ubuntu/distributional-ptq/cross_arch_generalization/results/alpha_sweep_dit_results.json"
from diffusers import DiTTransformer2DModel
t=DiTTransformer2DModel.from_pretrained("facebook/DiT-XL-2-256", subfolder="transformer",
    torch_dtype=torch.float32).to(DEV).eval()
orig_sd=copy.deepcopy(t.state_dict())

g=torch.Generator(device=DEV).manual_seed(0); B=64
lat=torch.randn(B,4,32,32,device=DEV,generator=g)
ts=torch.randint(0,1000,(B,),device=DEV,generator=g)
cls=torch.randint(0,1000,(B,),device=DEV,generator=g)
def predict(model): return model(lat,timestep=ts,class_labels=cls).sample.float()
fp_out=predict(t)

def alpha_star_estimate():
    vals=[]
    for name,mod in t.named_modules():
        if not isinstance(mod,nn.Linear) or mod.weight.dim()!=2: continue
        w=mod.weight.data.float()
        for r in range(0, w.shape[0], max(1,w.shape[0]//8)):  # subsample rows
            row=w[r].abs()
            T=3*row.std(); M=torch.quantile(row,0.999)
            p_out=(row>T).float().mean(); p_in=1-p_out
            if p_out>0 and M>T:
                a=torch.pow(T*p_out/((M-T)*p_in+1e-12),1/3.0)
                if torch.isfinite(a): vals.append(a.item())
    return float(np.median(vals)) if vals else 0.07

def quant_forced(bits,alpha):
    n=0
    for name,mod in t.named_modules():
        if not isinstance(mod,nn.Linear) or mod.weight.dim()!=2: continue
        w=mod.weight.data; n+=1
        wq=_quantize_per_channel_with_dbaf(w,bits,alpha=alpha)
        mod.weight.data=wq.to(mod.weight.dtype)
    return n
def recon_nmse():
    o=predict(t); return float(((o-fp_out)**2).mean()/(fp_out**2).mean())

astar=round(alpha_star_estimate(),3)
grid=[astar,0.25,0.3,0.5,0.75,0.95,0.99]
print(f"alpha* estimate = {astar}", flush=True)
res={"alpha_star_est":astar,"bits":4,"grid":grid,"metric":"per_step_NMSE_vs_FP","sweep":{}}
for a in grid:
    t.load_state_dict(orig_sd); quant_forced(4,a)
    nm=recon_nmse(); res["sweep"][str(a)]=round(nm,5)
    print(f"  alpha={a:<6} recon-NMSE={nm:.5f}", flush=True)
t.load_state_dict(orig_sd)
best=min(res["sweep"],key=lambda k:res["sweep"][k])
res["selected_alpha"]=float(best); res["selected_nmse"]=res["sweep"][best]
print(f"\n[DiT] reconstruction-selected alpha = {best} (NMSE {res['sweep'][best]})", flush=True)
json.dump(res,open(OUT,"w"),indent=2); print("saved ->",OUT)
