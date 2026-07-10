"""Second PCSA-fire candidate: Whisper under a FEW-SPEAKER clustered regime.

Speaker identity is acoustically strong (tight clusters) and per-speaker
recording energy genuinely varies (divergent activation magnitudes) -- the two
conditions PCSA needs. We take a few LibriSpeech speakers, and test at the
cross-attention K/V source (the encoder output, which is audio-conditioned):
  - descriptor: per-utterance pooled encoder output.
  - compactness c (expect low with few speakers).
  - per-speaker activation scale CV (expect > DiT's 0.044).
  - INT4 quant NMSE of the encoder output: global scale vs PCSA per-speaker
    scale. Lower PCSA NMSE = PCSA fires and helps on speech.
"""
import sys, io, json
import numpy as np, torch, torch.nn.functional as F
torch.set_grad_enabled(False); DEV="cuda"; QMAX=7
import soundfile as sf
from collections import Counter, defaultdict
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio

m=WhisperForConditionalGeneration.from_pretrained("openai/whisper-small",torch_dtype=torch.float32).to(DEV).eval()
wp=WhisperProcessor.from_pretrained("openai/whisper-small")
ds=load_dataset("openslr/librispeech_asr","clean",split="test",streaming=True).cast_column("audio",Audio(decode=False))

# collect utterances grouped by speaker
by_spk=defaultdict(list)
seen=0
for ex in ds:
    a=ex["audio"]; data=a["bytes"] if a.get("bytes") else open(a["path"],"rb").read()
    arr,_=sf.read(io.BytesIO(data),dtype="float32"); arr=arr.mean(1) if arr.ndim>1 else arr
    by_spk[ex["speaker_id"]].append(arr); seen+=1
    if seen>=600: break
top=[s for s,_ in Counter({k:len(v) for k,v in by_spk.items()}).most_common(4)]
K=len(top)
audio=[]; spk=[]
for si,s in enumerate(top):
    for arr in by_spk[s][:30]:
        audio.append(arr); spk.append(si)
spk=np.array(spk)
print(f"{K} speakers, {len(audio)} utterances ({Counter(spk)})",flush=True)

# per-utterance encoder output: descriptor (pooled) + per-utterance abs-max
descs=[]; absmax=[]; enc_outs=[]
for i in range(0,len(audio),16):
    feats=wp(audio[i:i+16],sampling_rate=16000,return_tensors="pt").input_features.to(DEV)
    eo=m.model.encoder(feats).last_hidden_state.float()        # [B,T,D]
    descs.append(F.normalize(eo.mean(1),dim=-1).cpu().numpy())
    absmax.append(eo.abs().amax(dim=(1,2)).cpu().numpy())
    enc_outs.append(eo.cpu())
D=np.concatenate(descs,0); A=np.concatenate(absmax,0); EO=torch.cat(enc_outs,0)

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
c,cs,assign=compactness(D,K)
purity=np.mean([Counter(spk[assign==j]).most_common(1)[0][1]/max((assign==j).sum(),1)
                for j in range(K) if (assign==j).any()])
print(f"[Whisper {K}-speaker] enc-output compactness c={c:.3f}±{cs:.3f} -> "
      f"{'FIRE' if c<=0.4 else 'SKIP'} (speaker purity ~{purity:.2f})",flush=True)

# split calib/eval, route eval by nearest calib centroid
n=len(audio); idx=np.arange(n); rng=np.random.default_rng(0); rng.shuffle(idx)
cal,ev=idx[:int(0.75*n)],idx[int(0.75*n):]
def kmfit(X,k):
    rng=np.random.default_rng(1); C=X[rng.choice(len(X),k,replace=False)].copy()
    for _ in range(140):
        a=((X[:,None]-C[None])**2).sum(-1).argmin(1)
        for j in range(k):
            if (a==j).any(): C[j]=X[a==j].mean(0)
    return C
C4=kmfit(D[cal].astype(np.float64),K)
route=lambda x:((x[:,None]-C4[None])**2).sum(-1).argmin(1)
ev_id=route(D[ev].astype(np.float64)); cal_id=route(D[cal].astype(np.float64))

g_scale=A[cal].max()/QMAX
cl_scale=np.array([A[cal][cal_id==j].max() if (cal_id==j).any() else A[cal].max() for j in range(K)])/QMAX
scale_cv=cl_scale.std()/cl_scale.mean()
print(f"  per-speaker scale CV = {scale_cv:.3f}  (DiT was 0.044)",flush=True)

def nmse(mode):
    tot=0.0
    for k_i,i in enumerate(ev):
        x=EO[i]
        s=g_scale if mode=="global" else cl_scale[ev_id[k_i]]
        s=max(s,1e-9)
        xq=torch.round(x/s).clamp(-QMAX,QMAX)*s
        tot+=((xq-x)**2).mean().item()/ (x**2).mean().item()
    return tot/len(ev)
ng=nmse("global"); npc=nmse("pcsa")
print(f"  encoder-output INT4 NMSE: global={ng:.5f}  PCSA={npc:.5f} "
      f"({100*(1-npc/ng):+.1f}% vs global)",flush=True)

res=dict(regime=f"clustered_{K}_speakers", compactness=round(c,3),std=round(cs,3),
    decision="FIRE" if c<=0.4 else "SKIP", speaker_purity=round(float(purity),3),
    scale_cv=round(float(scale_cv),3), nmse_global=round(ng,5), nmse_pcsa=round(npc,5),
    pcsa_improvement_pct=round(100*(1-npc/ng),2))
json.dump(res,open("results/whisper_speaker_pcsa_results.json","w"),indent=2)
print(f"\n{json.dumps(res,indent=2)}\nsaved.")
