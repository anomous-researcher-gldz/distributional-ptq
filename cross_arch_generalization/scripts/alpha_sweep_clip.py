"""One-block reconstruction sweep for DBAF alpha on CLIP-ViT-L/14 (paper's rule).

Grid {alpha*,0.25,0.3,0.5,0.75,0.95,0.99}; selection = argmin reconstruction NMSE
on the model's image-text logits vs FP over a calibration batch. We also report
CIFAR-100 zero-shot top-1 at each alpha to confirm the reconstruction-selected
alpha coincides with the task-optimal alpha (expected for a discriminative model).
"""
import sys, copy, json
import os as _os_repo
_REPO_ROOT = _os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_grad_enabled(False)
REPO=_REPO_ROOT; sys.path.insert(0,REPO); sys.path.insert(0,REPO+"/FlatQuant")
from flatquant.baselines.rtn import _quantize_per_channel_with_dbaf
DEV="cuda"
OUT=_REPO_ROOT + "/cross_arch_generalization/results/alpha_sweep_clip_results.json"
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
MID="openai/clip-vit-large-patch14"
model=CLIPModel.from_pretrained(MID,torch_dtype=torch.float32).to(DEV).eval()
proc=CLIPProcessor.from_pretrained(MID)
orig=copy.deepcopy(model.state_dict())
ds=load_dataset("uoft-cs/cifar100",split="test").select(range(2500))
classes=ds.features["fine_label"].names
imgs=[e["img"] for e in ds]; labels=np.array([e["fine_label"] for e in ds])
_txt=proc(text=[f"a photo of a {c}" for c in classes],return_tensors="pt",padding=True).to(DEV)

def logits(bs=250):
    outs=[]
    for i in range(0,len(imgs),bs):
        ib=proc(images=imgs[i:i+bs],return_tensors="pt").to(DEV)
        o=model(input_ids=_txt["input_ids"],attention_mask=_txt["attention_mask"],
                pixel_values=ib["pixel_values"]).logits_per_image
        outs.append(o.float())
    return torch.cat(outs,0)
fp_logits=logits()
fp_top1=100.0*(fp_logits.argmax(1).cpu().numpy()==labels).mean()

def astar_est():
    vals=[]
    for _,mod in model.named_modules():
        if not isinstance(mod,nn.Linear) or mod.weight.dim()!=2: continue
        w=mod.weight.data.float()
        for r in range(0,w.shape[0],max(1,w.shape[0]//6)):
            row=w[r].abs(); T=3*row.std(); M=torch.quantile(row,0.999)
            p_out=(row>T).float().mean(); p_in=1-p_out
            if p_out>0 and M>T:
                a=torch.pow(T*p_out/((M-T)*p_in+1e-12),1/3.0)
                if torch.isfinite(a): vals.append(a.item())
    return float(np.median(vals)) if vals else 0.07
def quant_forced(bits,alpha):
    for _,mod in model.named_modules():
        if not isinstance(mod,nn.Linear) or mod.weight.dim()!=2: continue
        mod.weight.data=_quantize_per_channel_with_dbaf(mod.weight.data,bits,alpha=alpha).to(mod.weight.dtype)

astar=round(astar_est(),3); grid=[astar,0.25,0.3,0.5,0.75,0.95,0.99]
print(f"FP top-1={fp_top1:.2f}  alpha*={astar}",flush=True)
res={"alpha_star_est":astar,"bits":4,"grid":grid,"FP_top1":round(float(fp_top1),2),"sweep":{}}
for a in grid:
    model.load_state_dict(orig); quant_forced(4,a)
    lg=logits(); nm=float(((lg-fp_logits)**2).mean()/(fp_logits**2).mean())
    top1=100.0*(lg.argmax(1).cpu().numpy()==labels).mean()
    res["sweep"][str(a)]={"recon_nmse":round(nm,5),"top1":round(float(top1),2)}
    print(f"  alpha={a:<6} recon-NMSE={nm:.5f}  top1={top1:.2f}",flush=True)
model.load_state_dict(orig)
best=min(res["sweep"],key=lambda k:res["sweep"][k]["recon_nmse"])
res["selected_alpha"]=float(best); res["selected"]=res["sweep"][best]
print(f"\n[CLIP] reconstruction-selected alpha={best} -> top1={res['sweep'][best]['top1']}",flush=True)
json.dump(res,open(OUT,"w"),indent=2); print("saved ->",OUT)
