"""Find a legitimate PCSA-FIRE site on a new family: DiT under a CLUSTERED
(few-class) deployment regime.

PCSA fires where per-input descriptors cluster (compactness c<=0.4) AND the
per-cluster activation SCALE differs (so a global scale over-ranges some
clusters). DiT is class-conditioned by design, so a few-class deployment is a
genuine clustered regime (not cherry-picking: we ALSO report that the diffuse
many-class/random regime self-disables, c~0.76, in dit_flagship.py).

Test (activation-only, weights FP, to isolate PCSA):
  - descriptor: per-sample mean-pooled input to block0 attention (paper's
    hidden-state descriptor).
  - quantize every transformer-block Linear INPUT to symmetric INT4 with
    (a) a single GLOBAL per-site scale vs (b) PCSA per-CLUSTER per-site scale
    routed by the descriptor.
  - metric: NMSE of the final noise prediction vs the FP model.
If PCSA-NMSE < global-NMSE at a firing site, PCSA fires and helps on diffusion.
"""
import sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_grad_enabled(False); DEV="cuda"
from diffusers import DiTTransformer2DModel
t=DiTTransformer2DModel.from_pretrained("facebook/DiT-XL-2-256",subfolder="transformer",
    torch_dtype=torch.float32).to(DEV).eval()
QMAX=7

# --- CLUSTERED regime: K=4 classes x 40 noises = 160 samples ---
K=4
CLASSES=[207,360,933,980]           # golden retriever, otter, cheeseburger, volcano
g=torch.Generator(device=DEV).manual_seed(0)
cls=torch.tensor(CLASSES,device=DEV).repeat_interleave(40)
lat=torch.randn(160,4,32,32,device=DEV,generator=g)
ts=torch.full((160,),100,device=DEV)
gt_cluster=np.repeat(np.arange(K),40)          # ground-truth class cluster
n_cal=120; n_ev=40
cal=np.arange(120); ev=np.arange(120,160)

lin_sites=[(n,m) for n,m in t.named_modules() if isinstance(m,nn.Linear) and "transformer_blocks" in n]
desc_site=t.transformer_blocks[0].attn1.to_q

# ---- Pass 1: FP, capture descriptor + per-site per-sample max-abs + FP output ----
descs=[]; site_absmax={n:[] for n,_ in lin_sites}
def desc_hook(mod,args):
    if torch.is_tensor(args[0]):
        descs.append(F.normalize(args[0].float().mean(1),dim=-1).cpu().numpy())  # [B,D]
def mk_abs(n):
    def h(mod,args):
        if torch.is_tensor(args[0]):
            site_absmax[n].append(args[0].detach().float().abs().amax(dim=tuple(range(1,args[0].dim()))).cpu().numpy())
    return h
hs=[desc_site.register_forward_pre_hook(desc_hook)]+[m.register_forward_pre_hook(mk_abs(n)) for n,m in lin_sites]
fp_out=t(lat,timestep=ts,class_labels=cls).sample.float().cpu()
for h in hs: h.remove()
D=np.concatenate(descs,0)                        # [160, d]
for n in site_absmax: site_absmax[n]=np.concatenate(site_absmax[n],0)   # [160]

# ---- compactness of descriptors (clustered regime) ----
def compactness(X,k=4,nb=15):
    X=X.astype(np.float64)
    def km(A,seed):
        rng=np.random.default_rng(seed); C=A[rng.choice(len(A),k,replace=False)].copy()
        for _ in range(140):
            a=((A[:,None]-C[None])**2).sum(-1).argmin(1)
            for j in range(k):
                if (a==j).any(): C[j]=A[a==j].mean(0)
        return np.linalg.norm(A-C[a],axis=1).mean(),a
    dr,assign=km(X,0); rs=[]
    for s in range(nb):
        Xp=X.copy(); rng=np.random.default_rng(s+7)
        for j in range(X.shape[1]): Xp[:,j]=rng.permutation(Xp[:,j])
        rs.append(dr/max(km(Xp,s)[0],1e-8))
    return float(np.mean(rs)),float(np.std(rs)),assign
c,cs,assign=compactness(D,k=K)
# cluster purity vs ground-truth class
from collections import Counter
purity=np.mean([Counter(gt_cluster[assign==j]).most_common(1)[0][1]/max((assign==j).sum(),1)
                for j in range(K)]) if len(set(assign))==K else float('nan')
print(f"[DiT clustered K={K}] descriptor compactness c={c:.3f}±{cs:.3f} -> "
      f"{'FIRE' if c<=0.4 else 'SKIP'}  (cluster purity vs class ~{purity:.2f})",flush=True)

# ---- route eval samples by nearest calibration-cluster centroid ----
def kmeans_fit(X,k,seed=0):
    rng=np.random.default_rng(seed); C=X[rng.choice(len(X),k,replace=False)].copy()
    for _ in range(140):
        a=((X[:,None]-C[None])**2).sum(-1).argmin(1)
        for j in range(k):
            if (a==j).any(): C[j]=X[a==j].mean(0)
    return C
Cc=kmeans_fit(D[cal].astype(np.float64),K)
route=lambda x: ((x[:,None]-Cc[None])**2).sum(-1).argmin(1)
cal_id=route(D[cal].astype(np.float64)); ev_id=route(D[ev].astype(np.float64))

# ---- per-site scales: global vs per-cluster (from calib) ----
global_scale={}; cluster_scale={}; scale_cv=[]
for n in site_absmax:
    a=site_absmax[n]
    global_scale[n]=a[cal].max()/QMAX
    cs_=np.array([a[cal][cal_id==j].max() if (cal_id==j).any() else a[cal].max() for j in range(K)])/QMAX
    cluster_scale[n]=cs_
    scale_cv.append(cs_.std()/max(cs_.mean(),1e-9))
print(f"  mean per-cluster scale CV across sites = {np.mean(scale_cv):.3f} "
      f"(>0 means clusters need different scales -> PCSA has headroom)",flush=True)

# ---- Pass 2: eval batch with activation INT4 quant, global vs PCSA ----
lat_e=lat[ev]; ts_e=ts[ev]; cls_e=cls[ev]
def run_quant(mode):
    hs=[]
    def mk(n):
        def h(mod,args):
            x=args[0]
            if not torch.is_tensor(x): return
            if mode=="global": s=torch.tensor(global_scale[n],device=x.device)
            else:
                s=torch.tensor(cluster_scale[n][ev_id],device=x.device)  # [B]
                s=s.view(-1,*([1]*(x.dim()-1)))
            s=s.clamp(min=1e-9)
            xq=torch.round(x/s).clamp(-QMAX,QMAX)*s
            return (xq,)+tuple(args[1:])
        return h
    for n,m in lin_sites: hs.append(m.register_forward_pre_hook(mk(n)))
    out=t(lat_e,timestep=ts_e,class_labels=cls_e).sample.float().cpu()
    for h in hs: h.remove()
    return float(((out-fp_out[ev])**2).mean()/(fp_out[ev]**2).mean())

nmse_global=run_quant("global")
nmse_pcsa=run_quant("pcsa")
print(f"  A4 activation quant NMSE-to-FP: global={nmse_global:.5f}  "
      f"PCSA={nmse_pcsa:.5f}  ({100*(1-nmse_pcsa/nmse_global):+.1f}% vs global)",flush=True)

res=dict(regime="clustered_K4_classes", compactness=round(c,3), compactness_std=round(cs,3),
    decision="FIRE" if c<=0.4 else "SKIP", cluster_purity=round(float(purity),3),
    mean_scale_cv=round(float(np.mean(scale_cv)),3),
    nmse_global=round(nmse_global,5), nmse_pcsa=round(nmse_pcsa,5),
    pcsa_improvement_pct=round(100*(1-nmse_pcsa/nmse_global),2))
json.dump(res,open("results/dit_pcsa_fire_results.json","w"),indent=2)
print(f"\n{json.dumps(res,indent=2)}\nsaved.")
