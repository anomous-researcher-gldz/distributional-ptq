"""Option 2: reproduce PCSA FIRING on SAM (the paper's in-scope fire site) to
confirm the primitive genuinely fires when its precondition IS met -- the honest
counterpart to the new-family self-disable results.

Paper reports SAM-B mask-decoder cross-attention q-projection at c=0.189 (FIRE).
We load SAM-B (public checkpoint), run point prompts on real images, hook that
exact site, and measure compactness + per-cluster activation scale CV + INT4
quant NMSE (global vs per-cluster). If c is low and PCSA helps, the mechanism is
real; contrast with new families where c stays high and PCSA self-disables.
"""
import sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_grad_enabled(False); DEV="cuda"; QMAX=7
from segment_anything import sam_model_registry, SamPredictor
from skimage import data as skdata
from skimage.transform import resize

sam=sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth").to(DEV).eval()
pred=SamPredictor(sam)

# real RGB images
imgs=[]
for fn in ["astronaut","coffee","chelsea","rocket","cat"]:
    try:
        im=getattr(skdata,fn)()
        if im.ndim==2: im=np.stack([im]*3,-1)
        imgs.append(im[...,:3].astype(np.uint8))
    except Exception: pass
print(f"{len(imgs)} real images",flush=True)

# hook the mask-decoder cross-attn (token->image) q_proj INPUT
site=sam.mask_decoder.transformer.layers[0].cross_attn_token_to_image.q_proj
descs=[]; absmax=[]; img_id=[]
def h(mod,args):
    if torch.is_tensor(args[0]):
        x=args[0].float()                         # [B, tokens, D]
        descs.append(F.normalize(x.mean(1),dim=-1).cpu().numpy())
        absmax.append(x.abs().amax(dim=(1,2)).cpu().numpy())
hd=site.register_forward_pre_hook(h)

for ii,im in enumerate(imgs):
    pred.set_image(im)
    H,W=im.shape[:2]
    # grid of point prompts across the image
    gx=np.linspace(0.15,0.85,6)*W; gy=np.linspace(0.15,0.85,6)*H
    for x in gx:
        for y in gy:
            pc=np.array([[x,y]]); pl=np.array([1])
            descs_before=len(descs)
            pred.predict(point_coords=pc,point_labels=pl,multimask_output=False)
            for _ in range(len(descs)-descs_before): img_id.append(ii)
hd.remove()
D=np.concatenate(descs,0); A=np.concatenate(absmax,0); img_id=np.array(img_id)
print(f"{len(D)} prompt descriptors at mask-decoder cross-attn q-proj",flush=True)

def compactness(X,k=4,nb=15):
    X=X.astype(np.float64)
    def km(A_,seed):
        rng=np.random.default_rng(seed); C=A_[rng.choice(len(A_),k,replace=False)].copy()
        for _ in range(140):
            a=((A_[:,None]-C[None])**2).sum(-1).argmin(1)
            for j in range(k):
                if (a==j).any(): C[j]=A_[a==j].mean(0)
        return np.linalg.norm(A_-C[a],axis=1).mean(),a
    dr,assign=km(X,0); rs=[]
    for s in range(nb):
        Xp=X.copy(); rng=np.random.default_rng(s+7)
        for j in range(X.shape[1]): Xp[:,j]=rng.permutation(Xp[:,j])
        rs.append(dr/max(km(Xp,s)[0],1e-8))
    return float(np.mean(rs)),float(np.std(rs)),assign
c,cs,assign=compactness(D,k=4)
print(f"[SAM mask-decoder cross-attn q] compactness c={c:.3f}±{cs:.3f} -> "
      f"{'FIRE' if c<=0.4 else 'SKIP'}  (paper reports 0.189)",flush=True)

K=4
gl=A.max()/QMAX
cl=np.array([A[assign==j].max() if (assign==j).any() else A.max() for j in range(K)])/QMAX
scale_cv=cl.std()/max(cl.mean(),1e-9)
def q(vals,s): return np.mean((np.round(vals/s).clip(-QMAX,QMAX)*s-vals)**2/np.maximum(vals**2,1e-12))
nmse_g=q(A,gl); nmse_p=np.mean([q(A[assign==j],cl[j]) for j in range(K) if (assign==j).any()])
print(f"  per-cluster scale CV = {scale_cv:.3f}  (new families were 0.01-0.09)",flush=True)
print(f"  abs-max INT4 NMSE: global={nmse_g:.5f} PCSA={nmse_p:.5f} "
      f"({100*(1-nmse_p/nmse_g):+.1f}% vs global)",flush=True)

res=dict(model="SAM-B", site="mask_decoder.cross_attn_token_to_image.q_proj",
    n_prompts=int(len(D)), compactness=round(c,3),std=round(cs,3),
    decision="FIRE" if c<=0.4 else "SKIP", paper_reported=0.189,
    scale_cv=round(float(scale_cv),3), nmse_global=round(float(nmse_g),5),
    nmse_pcsa=round(float(nmse_p),5), pcsa_improvement_pct=round(100*(1-nmse_p/nmse_g),2))
json.dump(res,open("results/sam_pcsa_fire_results.json","w"),indent=2)
print(f"\n{json.dumps(res,indent=2)}\nsaved.")
