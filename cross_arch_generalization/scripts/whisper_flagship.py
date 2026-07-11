"""Flagship end-to-end on Whisper-small (speech, NOT in paper).

DBAF: training-free W{4,3} per-channel RTN vs RTN+DBAF (forced/gated), paper's
exact quantizer. Metric: WER on LibriSpeech test-clean subset (greedy decode).
PCSA site: compactness at decoder/encoder attention q-proj -> fire/skip.
"""
import sys, copy, json
import os as _os_repo
_REPO_ROOT = _os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_grad_enabled(False)
REPO = _REPO_ROOT
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/FlatQuant")
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf
DEV = "cuda"
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25   # `python whisper_flagship.py [alpha]`
import jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset

MID = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(MID, torch_dtype=torch.float32).to(DEV).eval()
proc = WhisperProcessor.from_pretrained(MID)
orig_sd = copy.deepcopy(model.state_dict())

print("streaming LibriSpeech test-clean subset...", flush=True)
import io, soundfile as sf
from datasets import Audio
ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))  # avoid torchcodec; decode via soundfile
NS = 200
samples = []
for ex in ds:
    a = ex["audio"]
    data = a["bytes"] if a.get("bytes") is not None else open(a["path"], "rb").read()
    arr, sr = sf.read(io.BytesIO(data), dtype="float32")
    if arr.ndim > 1: arr = arr.mean(1)
    samples.append((arr, sr, ex["text"]))
    if len(samples) >= NS: break
print(f"  {len(samples)} utterances", flush=True)
import re, string
def _norm(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()
refs = [_norm(s[2]) for s in samples]

def transcribe_all(model, bs=16):
    hyps = []
    for i in range(0, len(samples), bs):
        chunk = samples[i:i+bs]
        feats = proc([c[0] for c in chunk], sampling_rate=16000,
                     return_tensors="pt").input_features.to(DEV)
        ids = model.generate(feats, max_new_tokens=200, num_beams=1)
        hyps += proc.batch_decode(ids, skip_special_tokens=True)
    return hyps

def wer(model):
    hyps = [_norm(h) for h in transcribe_all(model)]
    return 100.0 * jiwer.wer(refs, hyps)

def quantize(model, bits, mode):
    n=0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or mod.weight.dim()!=2: continue
        if "proj_out" in name: continue  # tied to embeddings / vocab head
        w = mod.weight.data; n+=1
        if mode=="rtn": wq=_quantize_tensor_uniform(w,bits,per_channel=True)
        elif mode=="dbaf_forced": wq=_quantize_per_channel_with_dbaf(w,bits,alpha=ALPHA)
        elif mode=="dbaf_gated": wq=_quantize_per_channel_with_dbaf(w,bits,alpha=ALPHA,gate_frac3_max=2e-2)
        mod.weight.data=wq.to(mod.weight.dtype)
    return n

results={}
print("\n[Whisper] FP baseline...", flush=True)
results["FP"]=round(wer(model),2); print(f"  FP WER={results['FP']}%", flush=True)
for bits in [4,3]:
    for mode in ["rtn","dbaf_forced","dbaf_gated"]:
        model.load_state_dict(orig_sd)
        nl=quantize(model,bits,mode)
        w=round(wer(model),2); results[f"W{bits}_{mode}"]=w
        print(f"  W{bits}_{mode:12s} WER={w}%  ({nl} linears)", flush=True)
model.load_state_dict(orig_sd)

# ---- PCSA-site compactness: decoder self-attn q_proj, descriptor per utterance ----
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

site = model.model.encoder.layers[0].self_attn.q_proj
descs=[]
def h(mod,args):
    x=args[0]
    if torch.is_tensor(x):
        d=F.normalize(x.float().mean(1),dim=-1); descs.append(d.cpu().numpy())
hd=site.register_forward_pre_hook(h)
for i in range(0,min(120,len(samples)),16):
    chunk=samples[i:i+16]
    feats=proc([c[0] for c in chunk],sampling_rate=16000,return_tensors="pt").input_features.to(DEV)
    model.model.encoder(feats)
hd.remove()
X=np.concatenate(descs,0)
c,cs=compactness(X)
results["pcsa_site_compactness_encoder_q_proj"]=dict(c=round(c,3),std=round(cs,3),
    decision="FIRE" if c<=0.4 else "SKIP", n=int(X.shape[0]))
print(f"\n[Whisper] PCSA-site compactness (encoder q_proj): c={c:.3f}±{cs:.3f} "
      f"-> {'FIRE' if c<=0.4 else 'SKIP'}", flush=True)

_tag = "_a025" if abs(ALPHA-0.25)<1e-9 else ""
out=f"{_REPO_ROOT}/cross_arch_generalization/results/whisper_flagship{_tag}_results.json"
json.dump(results,open(out,"w"),indent=2)
print(f"\nSaved -> {out}\n{json.dumps(results,indent=2)}")
