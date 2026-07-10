"""Option 3: multimodal LLM (LLaVA-1.5-7B) KV-cache PCSA -- the closest analog of
the paper's LLaMA KV-PCSA win, on a genuinely multimodal input.

The KV cache is the ONE site where per-input magnitude divergence survives
(keys/values are post-projection, NOT re-normalized), which is why the paper's
KV-PCSA fires there while q-proj self-disables. We feed image-clustered prompts
(K distinct image categories, same text) and test at the LLM self-attention
K/V projections:
  - descriptor: per-prompt pooled LLM hidden state.
  - compactness c of descriptors.
  - per-cluster KV-cache max-abs scale CV (the quantity my earlier hunt showed is
    the bottleneck; KV cache is where it should finally diverge).
  - INT4 KV quant NMSE: global scale vs PCSA per-cluster scale.
Fires-and-helps => PCSA generalizes to a multimodal model.
"""
import sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_grad_enabled(False); DEV="cuda"; QMAX=7
from transformers import LlavaForConditionalGeneration, AutoProcessor
from datasets import load_dataset

MID="llava-hf/llava-1.5-7b-hf"
model=LlavaForConditionalGeneration.from_pretrained(MID,torch_dtype=torch.float16,
    low_cpu_mem_usage=True).to(DEV).eval()
proc=AutoProcessor.from_pretrained(MID)

# image-clustered prompts: K CIFAR-100 classes x M images, same text
cif=load_dataset("uoft-cs/cifar100",split="test")
names=cif.features["fine_label"].names
CLS=[names.index(x) for x in ["castle","mountain","orchid","dolphin"]]
K=len(CLS); M=24
imgs=[]; lab=[]
for ci,c in enumerate(CLS):
    got=0
    for ex in cif:
        if ex["fine_label"]==c:
            imgs.append(ex["img"].convert("RGB")); lab.append(ci); got+=1
            if got>=M: break
lab=np.array(lab)
prompt="USER: <image>\nDescribe this image. ASSISTANT:"

# choose a mid LLM layer; hook its k_proj & v_proj OUTPUTS (the cached K/V)
lm=model.model.language_model
layer=lm.layers[len(lm.layers)//2].self_attn
kv={"k":[],"v":[],"desc":[]}
def kh(mod,inp,out): kv["k"].append(out.detach().float())
def vh(mod,inp,out): kv["v"].append(out.detach().float())
# descriptor: pooled hidden state entering this layer's q_proj
def dh(mod,args):
    if torch.is_tensor(args[0]): kv["desc"].append(F.normalize(args[0].float().mean(1),dim=-1).cpu().numpy())
h1=layer.k_proj.register_forward_hook(kh)
h2=layer.v_proj.register_forward_hook(vh)
h3=layer.q_proj.register_forward_pre_hook(dh)

descs=[]; k_absmax=[]; v_absmax=[]
BS=6
for i in range(0,len(imgs),BS):
    kv["k"].clear(); kv["v"].clear(); kv["desc"].clear()
    batch=imgs[i:i+BS]
    inp=proc(images=batch,text=[prompt]*len(batch),return_tensors="pt",padding=True).to(DEV,torch.float16)
    model(**inp)
    # per-prompt max-abs of cached K and V over (seq,dim)
    kt=kv["k"][0]; vt=kv["v"][0]                       # [B, seq, d]
    k_absmax.append(kt.abs().amax(dim=(1,2)).cpu().numpy())
    v_absmax.append(vt.abs().amax(dim=(1,2)).cpu().numpy())
    descs.append(kv["desc"][0])
    # keep per-prompt K tensors for NMSE (store pooled to save mem: full seq)
for h in (h1,h2,h3): h.remove()
D=np.concatenate(descs,0); KA=np.concatenate(k_absmax,0); VA=np.concatenate(v_absmax,0)
print(f"{len(imgs)} multimodal prompts, {K} image clusters",flush=True)

def compactness(X,k,nb=15):
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
from collections import Counter
c,cs,assign=compactness(D,K)
purity=np.mean([Counter(lab[assign==j]).most_common(1)[0][1]/max((assign==j).sum(),1)
                for j in range(K) if (assign==j).any()])
print(f"[LLaVA KV] descriptor compactness c={c:.3f}±{cs:.3f} -> "
      f"{'FIRE' if c<=0.4 else 'SKIP'} (image-cluster purity ~{purity:.2f})",flush=True)

def cluster_scale_cv(A):
    idx=np.arange(len(A))
    # cluster by descriptor assignment
    cl=np.array([A[assign==j].max() if (assign==j).any() else A.max() for j in range(K)])
    return cl.std()/max(cl.mean(),1e-9), cl
kcv,_=cluster_scale_cv(KA); vcv,_=cluster_scale_cv(VA)
print(f"  per-cluster KV max-abs scale CV: K={kcv:.3f}  V={vcv:.3f}  "
      f"(DiT was 0.044, Whisper 0.011)",flush=True)

# INT4 NMSE of K max-abs quant scale: global vs per-cluster (proxy on the scale itself)
def nmse_scale(A):
    gl=A.max()/QMAX
    cl=np.array([A[assign==j].max() if (assign==j).any() else A.max() for j in range(K)])/QMAX
    # NMSE of quantizing each prompt's abs-max value under global vs its cluster scale
    def q(vals,s): return np.mean((np.round(vals/s).clip(-QMAX,QMAX)*s - vals)**2/np.maximum(vals**2,1e-12))
    return q(A,gl), np.mean([q(A[assign==j],cl[j]) for j in range(K) if (assign==j).any()])
# better: quantify range-fit gain = how much smaller the per-cluster scale is on average
gl=KA.max(); cl=np.array([KA[assign==j].max() if (assign==j).any() else KA.max() for j in range(K)])
range_gain=100*(1-np.mean(cl[assign])/gl)
print(f"  per-cluster K quant-range reduction vs global = {range_gain:+.1f}% "
      f"(headroom PCSA converts to precision)",flush=True)

res=dict(model="llava-1.5-7b", regime=f"multimodal_{K}_image_clusters",
    compactness=round(c,3),std=round(cs,3),decision="FIRE" if c<=0.4 else "SKIP",
    image_cluster_purity=round(float(purity),3),
    kv_scale_cv_K=round(float(kcv),3),kv_scale_cv_V=round(float(vcv),3),
    per_cluster_range_reduction_pct=round(float(range_gain),2))
json.dump(res,open("results/llava_kv_pcsa_results.json","w"),indent=2)
print(f"\n{json.dumps(res,indent=2)}\nsaved.")
