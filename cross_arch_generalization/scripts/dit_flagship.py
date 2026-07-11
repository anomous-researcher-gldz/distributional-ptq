"""Flagship end-to-end on DiT-XL-2 (diffusion transformer, NOT in paper).

DiT weight gate-pass is 69.8% (heavy-tailed 30% self-disable), so GATED DBAF is
the correct test. Metric: denoising fidelity -- deviation of the quantized
noise prediction from the FP model's prediction, averaged over timesteps/classes
(standard cheap diffusion-PTQ quality proxy; lower = closer to FP). DBAF should
reduce this deviation where the gate fires.

PCSA site: DiT is class-conditioned, so per-class activation clusters should be
tight -> low compactness -> PCSA predicted to FIRE. This is the one new family
where the compactness gate should fire (positive prediction).
"""
import sys, copy, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_grad_enabled(False)
REPO="/home/ubuntu/distributional-ptq"; sys.path.insert(0,REPO); sys.path.insert(0,REPO+"/FlatQuant")
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf
DEV="cuda"
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25   # `python dit_flagship.py [alpha]`
from diffusers import DiTTransformer2DModel
t=DiTTransformer2DModel.from_pretrained("facebook/DiT-XL-2-256", subfolder="transformer",
    torch_dtype=torch.float32).to(DEV).eval()
orig_sd=copy.deepcopy(t.state_dict())

# fixed eval batch: varied classes, timesteps, noise
g=torch.Generator(device=DEV).manual_seed(0)
B=64
lat=torch.randn(B,4,32,32,device=DEV,generator=g)
ts=torch.randint(0,1000,(B,),device=DEV,generator=g)
cls=torch.randint(0,1000,(B,),device=DEV,generator=g)

def predict(model):
    return model(lat,timestep=ts,class_labels=cls).sample.float()

fp_out=predict(t)

def quantize(model,bits,mode):
    n=0
    for name,mod in model.named_modules():
        if not isinstance(mod,nn.Linear) or mod.weight.dim()!=2: continue
        w=mod.weight.data; n+=1
        if mode=="rtn": wq=_quantize_tensor_uniform(w,bits,per_channel=True)
        elif mode=="dbaf_forced": wq=_quantize_per_channel_with_dbaf(w,bits,alpha=ALPHA)
        elif mode=="dbaf_gated": wq=_quantize_per_channel_with_dbaf(w,bits,alpha=ALPHA,gate_frac3_max=2e-2)
        mod.weight.data=wq.to(mod.weight.dtype)
    return n

def dev_from_fp(model):
    o=predict(model)
    # relative deviation from FP prediction (NMSE)
    return float(((o-fp_out)**2).mean()/ (fp_out**2).mean())

results={}
for bits in [4,3]:
    for mode in ["rtn","dbaf_forced","dbaf_gated"]:
        t.load_state_dict(orig_sd)
        nl=quantize(t,bits,mode)
        d=dev_from_fp(t); results[f"W{bits}_{mode}"]=round(d,5)
        print(f"  W{bits}_{mode:12s} NMSE-vs-FP={d:.5f}  ({nl} linears)",flush=True)
t.load_state_dict(orig_sd)

# ---- compactness at DiT: per-sample mean-pooled attn q input, varied classes ----
def compactness(X,k=4,nb=12):
    X=X.astype(np.float64)
    def km(A,seed):
        rng=np.random.default_rng(seed); C=A[rng.choice(len(A),min(k,len(A)),replace=False)].copy()
        for _ in range(120):
            a=((A[:,None]-C[None])**2).sum(-1).argmin(1)
            for j in range(len(C)):
                if (a==j).any(): C[j]=A[a==j].mean(0)
        return np.linalg.norm(A-C[a],axis=1).mean()
    dr=km(X,0); rs=[]
    for s in range(nb):
        Xp=X.copy(); rng=np.random.default_rng(s+7)
        for j in range(X.shape[1]): Xp[:,j]=rng.permutation(Xp[:,j])
        rs.append(dr/max(km(Xp,s),1e-8))
    return float(np.mean(rs)),float(np.std(rs))

# descriptor: mean-pooled hidden entering block-0 attention, one per sample.
blk=t.transformer_blocks[0].attn1.to_q
descs=[]
def h(mod,args):
    x=args[0]
    if torch.is_tensor(x):
        d=F.normalize(x.float().mean(1),dim=-1); descs.append(d.cpu().numpy())
hd=blk.register_forward_pre_hook(h)
# 8 classes x 8 noises -> descriptors that should cluster by class
gg=torch.Generator(device=DEV).manual_seed(1)
cl8=torch.arange(8,device=DEV).repeat_interleave(8)
lt=torch.randn(64,4,32,32,device=DEV,generator=gg)
tt=torch.full((64,),200,device=DEV)
t(lt,timestep=tt,class_labels=cl8)
hd.remove()
X=np.concatenate(descs,0)
c,cs=compactness(X)
results["pcsa_site_compactness_attn_q"]=dict(c=round(c,3),std=round(cs,3),
    decision="FIRE" if c<=0.4 else "SKIP", n=int(X.shape[0]))
print(f"\n[DiT] PCSA-site compactness (block0 attn q, class-conditioned): "
      f"c={c:.3f}±{cs:.3f} -> {'FIRE' if c<=0.4 else 'SKIP'}",flush=True)

_tag = "_a025" if abs(ALPHA-0.25)<1e-9 else ""
out=f"/home/ubuntu/distributional-ptq/cross_arch_generalization/results/dit_flagship{_tag}_results.json"
json.dump(results,open(out,"w"),indent=2)
print(f"\nSaved -> {out}\n{json.dumps(results,indent=2)}")
