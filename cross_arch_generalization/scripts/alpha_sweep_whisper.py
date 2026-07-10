"""One-block reconstruction sweep for DBAF alpha on Whisper-small (paper's rule).

Grid {alpha*,0.25,0.3,0.5,0.75,0.95,0.99}; selection = argmin reconstruction NMSE
on encoder output vs FP over calibration audio. Task metric = WER at W3 (the
bit-width where DBAF matters most) computed at each alpha to confirm alignment.
"""
import sys, copy, json, io, re, string
import numpy as np, torch, torch.nn as nn
torch.set_grad_enabled(False)
REPO="/home/ubuntu/distributional-ptq"; sys.path.insert(0,REPO); sys.path.insert(0,REPO+"/FlatQuant")
from flatquant.baselines.rtn import _quantize_per_channel_with_dbaf
DEV="cuda"; import jiwer, soundfile as sf
OUT="/home/ubuntu/distributional-ptq/cross_arch_generalization/results/alpha_sweep_whisper_results.json"
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio
MID="openai/whisper-small"
model=WhisperForConditionalGeneration.from_pretrained(MID,torch_dtype=torch.float32).to(DEV).eval()
proc=WhisperProcessor.from_pretrained(MID); orig=copy.deepcopy(model.state_dict())
ds=load_dataset("openslr/librispeech_asr","clean",split="test",streaming=True).cast_column("audio",Audio(decode=False))
NS=200; samples=[]
for ex in ds:
    a=ex["audio"]; data=a["bytes"] if a.get("bytes") is not None else open(a["path"],"rb").read()
    arr,sr=sf.read(io.BytesIO(data),dtype="float32")
    if arr.ndim>1: arr=arr.mean(1)
    samples.append((arr,sr,ex["text"]))
    if len(samples)>=NS: break
def _norm(s): return re.sub(r"\s+"," ",s.lower().translate(str.maketrans("","",string.punctuation))).strip()
refs=[_norm(s[2]) for s in samples]
# calibration feats for reconstruction (first 48 utts)
calib=proc([c[0] for c in samples[:48]],sampling_rate=16000,return_tensors="pt").input_features.to(DEV)
fp_enc=model.model.encoder(calib).last_hidden_state.float()

def transcribe(bs=16):
    hyps=[]
    for i in range(0,len(samples),bs):
        feats=proc([c[0] for c in samples[i:i+bs]],sampling_rate=16000,return_tensors="pt").input_features.to(DEV)
        ids=model.generate(feats,max_new_tokens=200,num_beams=1)
        hyps+=proc.batch_decode(ids,skip_special_tokens=True)
    return [_norm(h) for h in hyps]
def astar_est():
    vals=[]
    for name,mod in model.named_modules():
        if not isinstance(mod,nn.Linear) or mod.weight.dim()!=2 or "proj_out" in name: continue
        w=mod.weight.data.float()
        for r in range(0,w.shape[0],max(1,w.shape[0]//6)):
            row=w[r].abs(); T=3*row.std(); M=torch.quantile(row,0.999)
            p_out=(row>T).float().mean(); p_in=1-p_out
            if p_out>0 and M>T:
                a=torch.pow(T*p_out/((M-T)*p_in+1e-12),1/3.0)
                if torch.isfinite(a): vals.append(a.item())
    return float(np.median(vals)) if vals else 0.07
def quant_forced(bits,alpha):
    for name,mod in model.named_modules():
        if not isinstance(mod,nn.Linear) or mod.weight.dim()!=2 or "proj_out" in name: continue
        mod.weight.data=_quantize_per_channel_with_dbaf(mod.weight.data,bits,alpha=alpha).to(mod.weight.dtype)

astar=round(astar_est(),3); grid=[astar,0.25,0.3,0.5,0.75,0.95,0.99]
fp_wer=100.0*jiwer.wer(refs,transcribe())
print(f"FP WER={fp_wer:.2f}  alpha*={astar}",flush=True)
res={"alpha_star_est":astar,"bits":3,"grid":grid,"FP_WER":round(float(fp_wer),2),"sweep":{}}
for a in grid:
    model.load_state_dict(orig); quant_forced(3,a)
    enc=model.model.encoder(calib).last_hidden_state.float()
    nm=float(((enc-fp_enc)**2).mean()/(fp_enc**2).mean())
    wer=100.0*jiwer.wer(refs,transcribe())
    res["sweep"][str(a)]={"recon_nmse":round(nm,5),"wer":round(float(wer),2)}
    print(f"  alpha={a:<6} recon-NMSE={nm:.5f}  WER={wer:.2f}",flush=True)
model.load_state_dict(orig)
best=min(res["sweep"],key=lambda k:res["sweep"][k]["recon_nmse"])
res["selected_alpha"]=float(best); res["selected"]=res["sweep"][best]
print(f"\n[Whisper] reconstruction-selected alpha={best} -> WER={res['sweep'][best]['wer']}",flush=True)
json.dump(res,open(OUT,"w"),indent=2); print("saved ->",OUT)
