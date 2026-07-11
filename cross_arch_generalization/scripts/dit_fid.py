"""(2) DiT FID-to-FP: standard diffusion-quantization fidelity metric.

Generate matched samples (identical seeds, class labels, sampler) from the FP
model and from each quantized model, then compute FID between the FP image set
and each quantized set. Lower FID-to-FP = quantized generator's output
distribution is closer to full precision. Self-contained (no external ImageNet
reference needed). Complements the per-step NMSE deviation-from-FP proxy.
"""
import sys, os, shutil, json, subprocess
import os as _os_repo
_REPO_ROOT = _os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.dirname(_os_repo.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
torch.set_grad_enabled(False); DEV="cuda"
sys.path.insert(0,_REPO_ROOT); sys.path.insert(0,_REPO_ROOT + "/FlatQuant")
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from PIL import Image

N=512; STEPS=25; BS=32
# DBAF fold strength: `python dit_fid.py [alpha]` (default 0.25). alpha=0.75 gives the
# outlier-protecting operating point DiT-XL FID needs (242.8->185.7); alpha=0.25 is the
# single-pass-sweep pick, which under-selects for FID (242.8->275.1, worse).
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
OUT="/tmp/claude-1000/-home-ubuntu/629c24a1-ecff-4b3c-8a83-a9277305528e/scratchpad/dit_fid_imgs"
pipe=DiTPipeline.from_pretrained("facebook/DiT-XL-2-256",torch_dtype=torch.float32).to(DEV)
pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
import copy
orig=copy.deepcopy(pipe.transformer.state_dict())

# fixed class labels + seeds so FP and quant sets are matched
rng=np.random.default_rng(0)
labels=rng.integers(0,1000,N).tolist()

def quantize(bits,mode):
    if mode=="fp": return 0
    n=0
    for name,mod in pipe.transformer.named_modules():
        if not isinstance(mod,nn.Linear) or mod.weight.dim()!=2: continue
        w=mod.weight.data; n+=1
        if mode=="rtn": wq=_quantize_tensor_uniform(w,bits,per_channel=True)
        elif mode=="dbaf_forced": wq=_quantize_per_channel_with_dbaf(w,bits,alpha=ALPHA)
        mod.weight.data=wq.to(mod.weight.dtype)
    return n

def generate(tag):
    d=os.path.join(OUT,tag); shutil.rmtree(d,ignore_errors=True); os.makedirs(d)
    k=0
    for i in range(0,N,BS):
        cl=labels[i:i+BS]
        gen=torch.Generator(device=DEV).manual_seed(1000+i)  # matched across conditions
        imgs=pipe(class_labels=cl,num_inference_steps=STEPS,generator=gen).images
        for im in imgs:
            im.save(os.path.join(d,f"{k:05d}.png")); k+=1
    return d

# FP reference
pipe.transformer.load_state_dict(orig); quantize(4,"fp")
print("generating FP set...",flush=True); fp_dir=generate("fp")
results={}
for mode in ["rtn","dbaf_forced"]:
    pipe.transformer.load_state_dict(orig); nl=quantize(4,mode)
    print(f"generating W4_{mode} set ({nl} linears)...",flush=True)
    d=generate(f"W4_{mode}")
    out=subprocess.run([sys.executable,"-m","pytorch_fid",fp_dir,d,"--device","cuda"],
                       capture_output=True,text=True)
    line=[l for l in out.stdout.splitlines() if "FID" in l]
    fid=float(line[-1].split()[-1]) if line else float("nan")
    results[f"W4_{mode}_FID_to_FP"]=round(fid,3)
    print(f"  W4_{mode}: FID-to-FP = {fid:.3f}",flush=True)
pipe.transformer.load_state_dict(orig)

results["N"]=N; results["steps"]=STEPS; results["alpha"]=ALPHA
_tag = "_a025" if abs(ALPHA-0.25)<1e-9 else ""
json.dump(results,open(f"{_REPO_ROOT}/cross_arch_generalization/results/dit_fid{_tag}_results.json","w"),indent=2)
print(f"\n{json.dumps(results,indent=2)}\nsaved.")
