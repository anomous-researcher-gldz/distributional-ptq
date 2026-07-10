"""(b) Hunt for a legitimate PCSA-fire site (compactness c<=0.4) on new families.

PCSA is per-site: the paper fires it at low-c sites (SAM mask-decoder c=0.19,
KV-cache) and self-disables at q-projections (c~0.79). We scan MANY sites per
new family -- attention q across depth, and content/class-conditioned sites --
and report the minimum c and whether any fires. Honest either way: a fire is a
PCSA-generalizes result; all-skip is a consistent-conservative result.
"""
import sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_grad_enabled(False); DEV="cuda"

def compactness(X,k=4,nb=12):
    X=X.astype(np.float64)
    if len(X)<k+1: return 1.0,0.0
    def km(A,seed):
        rng=np.random.default_rng(seed); C=A[rng.choice(len(A),k,replace=False)].copy()
        for _ in range(120):
            a=((A[:,None]-C[None])**2).sum(-1).argmin(1)
            for j in range(k):
                if (a==j).any(): C[j]=A[a==j].mean(0)
        return np.linalg.norm(A-C[a],axis=1).mean()
    dr=km(X,0); rs=[]
    for s in range(nb):
        Xp=X.copy(); rng=np.random.default_rng(s+7)
        for j in range(X.shape[1]): Xp[:,j]=rng.permutation(Xp[:,j])
        rs.append(dr/max(km(Xp,s),1e-8))
    return float(np.mean(rs)),float(np.std(rs))

def measure(model, sites, run_forward):
    """sites: {label: module}. Returns {label:(c,std,n)}."""
    store={l:[] for l in sites}
    def mk(l):
        def h(mod,args):
            x=args[0]
            if torch.is_tensor(x):
                store[l].append(F.normalize(x.float().mean(1),dim=-1).cpu().numpy())
        return h
    hs=[m.register_forward_pre_hook(mk(l)) for l,m in sites.items()]
    run_forward(model)
    for h in hs: h.remove()
    out={}
    for l in sites:
        if store[l]:
            X=np.concatenate(store[l],0)
            c,cs=compactness(X); out[l]=dict(c=round(c,3),std=round(cs,3),n=int(len(X)),
                decision="FIRE" if c<=0.4 else "SKIP")
    return out

results={}

# ===== DiT: class-conditioned; scan blocks + a class-descriptor site =====
print("=== DiT site scan ===",flush=True)
from diffusers import DiTTransformer2DModel
t=DiTTransformer2DModel.from_pretrained("facebook/DiT-XL-2-256",subfolder="transformer",
    torch_dtype=torch.float32).to(DEV).eval()
nb_blocks=len(t.transformer_blocks)
sites={f"block{b}.attn_q": t.transformer_blocks[b].attn1.to_q
       for b in [0, nb_blocks//4, nb_blocks//2, 3*nb_blocks//4, nb_blocks-1]}
def dit_fwd(m):
    g=torch.Generator(device=DEV).manual_seed(1)
    # STRONG class conditioning: 8 classes x 16 noises, same low timestep
    cl=torch.arange(8,device=DEV).repeat_interleave(16)
    lt=torch.randn(128,4,32,32,device=DEV,generator=g)
    tt=torch.full((128,),100,device=DEV)
    m(lt,timestep=tt,class_labels=cl)
results["DiT"]=measure(t,sites,dit_fwd)
for l,v in results["DiT"].items(): print(f"  {l:22s} c={v['c']:.3f} -> {v['decision']}",flush=True)
del t; torch.cuda.empty_cache()

# ===== Whisper: encoder q across depth + DECODER CROSS-ATTN (audio-conditioned) =====
print("=== Whisper site scan ===",flush=True)
import io, soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio
wm=WhisperForConditionalGeneration.from_pretrained("openai/whisper-small",torch_dtype=torch.float32).to(DEV).eval()
wp=WhisperProcessor.from_pretrained("openai/whisper-small")
ds=load_dataset("openslr/librispeech_asr","clean",split="test",streaming=True).cast_column("audio",Audio(decode=False))
auds=[]
for ex in ds:
    a=ex["audio"]; data=a["bytes"] if a.get("bytes") else open(a["path"],"rb").read()
    arr,_=sf.read(io.BytesIO(data),dtype="float32"); auds.append(arr.mean(1) if arr.ndim>1 else arr)
    if len(auds)>=64: break
enc=wm.model.encoder; dec=wm.model.decoder
sites={"enc.block0.q":enc.layers[0].self_attn.q_proj,
       "enc.block11.q":enc.layers[-1].self_attn.q_proj,
       "dec.block0.crossattn_q":dec.layers[0].encoder_attn.q_proj,
       "dec.block11.crossattn_q":dec.layers[-1].encoder_attn.q_proj}
def wh_fwd(m):
    for i in range(0,64,16):
        feats=wp([a for a in auds[i:i+16]],sampling_rate=16000,return_tensors="pt").input_features.to(DEV)
        dec_ids=torch.tensor([[m.config.decoder_start_token_id]]*feats.shape[0],device=DEV)
        m(input_features=feats,decoder_input_ids=dec_ids)
results["Whisper"]=measure(wm,sites,wh_fwd)
for l,v in results["Whisper"].items(): print(f"  {l:24s} c={v['c']:.3f} -> {v['decision']}",flush=True)
del wm; torch.cuda.empty_cache()

# ===== CLIP: vision q across depth + text q (prompt-conditioned) =====
print("=== CLIP site scan ===",flush=True)
from transformers import CLIPModel, CLIPProcessor
cm=CLIPModel.from_pretrained("openai/clip-vit-large-patch14",torch_dtype=torch.float32).to(DEV).eval()
cp=CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
from datasets import load_dataset as ld
cif=ld("uoft-cs/cifar100",split="test").select(range(400))
imgs=[e["img"] for e in cif]
vl=cm.vision_model.encoder.layers
sites={"vis.block0.q":vl[0].self_attn.q_proj,"vis.block12.q":vl[12].self_attn.q_proj,
       "vis.block23.q":vl[-1].self_attn.q_proj}
def cl_fwd(m):
    for i in range(0,400,100):
        ib=cp(images=imgs[i:i+100],return_tensors="pt").to(DEV)
        m.get_image_features(**ib)
results["CLIP"]=measure(cm,sites,cl_fwd)
for l,v in results["CLIP"].items(): print(f"  {l:20s} c={v['c']:.3f} -> {v['decision']}",flush=True)

# summary
allsites=[(fam,l,v["c"],v["decision"]) for fam,d in results.items() for l,v in d.items()]
fires=[s for s in allsites if s[3]=="FIRE"]
mn=min(allsites,key=lambda s:s[2])
print(f"\nMIN compactness across all {len(allsites)} sites: {mn[0]}/{mn[1]} c={mn[2]} ({mn[3]})")
print(f"FIRE sites: {fires if fires else 'none (all self-disable, consistent)'}")
json.dump(results,open("results/pcsa_site_hunt_results.json","w"),indent=2)
print("saved.")
