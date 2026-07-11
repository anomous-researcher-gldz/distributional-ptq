"""Flagship end-to-end on CLIP-ViT-L/14 (multimodal, NOT in paper).

DBAF: training-free per-channel W{4,3} RTN vs RTN+DBAF (forced and gated),
reusing the paper's exact quantizer (FlatQuant/flatquant/baselines/rtn.py).
Metric: zero-shot top-1 on CIFAR-100 test subset.

PCSA site: compactness c at the vision attention q-projection -> predicts
whether PCSA should fire (c<=0.4) or self-disable (c>0.4) on this family.
"""
import sys, copy, json, pathlib
import os as _os_repo
_REPO_ROOT = _os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_grad_enabled(False)
REPO = _REPO_ROOT
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/FlatQuant")
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf
DEV = "cuda"

ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25   # `python clip_flagship.py [alpha]`
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
MID = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(MID, torch_dtype=torch.float32).to(DEV).eval()
proc = CLIPProcessor.from_pretrained(MID)
orig_sd = copy.deepcopy(model.state_dict())

print("loading CIFAR-100 test subset...", flush=True)
ds = load_dataset("uoft-cs/cifar100", split="test")
N = 2500
ds = ds.select(range(N))
classes = ds.features["fine_label"].names
imgs = [ex["img"] for ex in ds]
labels = np.array([ex["fine_label"] for ex in ds])

# precompute text features once per weight-state (text tower is quantized too)
prompts = [f"a photo of a {c}" for c in classes]

_txt = proc(text=prompts, return_tensors="pt", padding=True).to(DEV)
def zeroshot_top1(model, bs=250):
    correct = 0
    for i in range(0, N, bs):
        ib = proc(images=imgs[i:i+bs], return_tensors="pt").to(DEV)
        out = model(input_ids=_txt["input_ids"], attention_mask=_txt["attention_mask"],
                    pixel_values=ib["pixel_values"])
        pred = out.logits_per_image.argmax(1).cpu().numpy()
        correct += (pred == labels[i:i+bs]).sum()
    return 100.0 * correct / N

def quantize(model, bits, mode):
    """mode: 'rtn' | 'dbaf_forced' | 'dbaf_gated'"""
    n_fold = 0; n_lin = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear): continue
        w = mod.weight.data
        if w.dim() != 2: continue
        n_lin += 1
        if mode == "rtn":
            wq = _quantize_tensor_uniform(w, bits, per_channel=True)
        elif mode == "dbaf_forced":
            wq = _quantize_per_channel_with_dbaf(w, bits, alpha=ALPHA); n_fold += 1
        elif mode == "dbaf_gated":
            wq = _quantize_per_channel_with_dbaf(w, bits, alpha=ALPHA, gate_frac3_max=2e-2)
        mod.weight.data = wq.to(mod.weight.dtype)
    return n_lin

results = {}
print("\n[CLIP] FP baseline...", flush=True)
results["FP"] = round(float(zeroshot_top1(model)), 2)
print(f"  FP top-1 = {results['FP']}", flush=True)

for bits in [4, 3]:
    for mode in ["rtn", "dbaf_forced", "dbaf_gated"]:
        model.load_state_dict(orig_sd)
        nl = quantize(model, bits, mode)
        acc = round(float(zeroshot_top1(model)), 2)
        key = f"W{bits}_{mode}"
        results[key] = acc
        print(f"  {key:18s} top-1 = {acc}  ({nl} linears)", flush=True)
model.load_state_dict(orig_sd)

# ---- PCSA-site compactness on CLIP vision q-proj ----
def compactness(X, k=4, nb=12):
    X = X.astype(np.float64)
    def km(A, seed):
        rng = np.random.default_rng(seed); C = A[rng.choice(len(A),k,replace=False)].copy()
        for _ in range(120):
            a = ((A[:,None]-C[None])**2).sum(-1).argmin(1)
            for j in range(k):
                if (a==j).any(): C[j]=A[a==j].mean(0)
        return np.linalg.norm(A-C[a],axis=1).mean()
    dr = km(X,0)
    rs=[]
    for s in range(nb):
        Xp=X.copy(); rng=np.random.default_rng(s+7)
        for j in range(X.shape[1]): Xp[:,j]=rng.permutation(Xp[:,j])
        rs.append(dr/max(km(Xp,s),1e-8))
    return float(np.mean(rs)), float(np.std(rs))

# hook the first vision attention q_proj; descriptor = mean-pooled L2-norm input
vqp = model.vision_model.encoder.layers[0].self_attn.q_proj
descs=[]
def h(mod,args):
    x=args[0]
    if torch.is_tensor(x):
        d=F.normalize(x.float().mean(1),dim=-1); descs.append(d.cpu().numpy())
hd=vqp.register_forward_pre_hook(h)
for i in range(0, 1000, 250):
    ib=proc(images=imgs[i:i+250],return_tensors="pt").to(DEV)
    model.get_image_features(**ib)
hd.remove()
X=np.concatenate(descs,0)
c,cs=compactness(X)
results["pcsa_site_compactness_q_proj"]=dict(c=round(c,3),std=round(cs,3),
    decision="FIRE" if c<=0.4 else "SKIP", n=int(X.shape[0]))
print(f"\n[CLIP] PCSA-site compactness (vision q_proj): c={c:.3f}±{cs:.3f} "
      f"-> PCSA {'FIRE' if c<=0.4 else 'SKIP'}", flush=True)

_tag = "_a025" if abs(ALPHA-0.25)<1e-9 else ""
out=f"{_REPO_ROOT}/cross_arch_generalization/results/clip_flagship{_tag}_results.json"
json.dump(results,open(out,"w"),indent=2)
print(f"\nSaved -> {out}\n{json.dumps(results,indent=2)}")
