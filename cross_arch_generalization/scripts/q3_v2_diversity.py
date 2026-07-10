"""(c) Q3 strengthened: compactness vs prompt diversity at LLaMA q_proj, with a
proper low-diversity cell (n=40 near-duplicate templated prompts) so the
statistic's RESPONSIVENESS to real prompt diversity is shown, not just its
invariance. Predicts: near-duplicate prompts -> low c (tight clusters); diverse
prompts -> high c (SKIP). Answers "how sensitive is compactness to diversity".
"""
import sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_grad_enabled(False)
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
MODEL="NousResearch/Meta-Llama-3-8B"
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16).cuda().eval()

def compactness(X,k=4,nb=15):
    X=X.astype(np.float64)
    def km(A,seed):
        rng=np.random.default_rng(seed); C=A[rng.choice(len(A),k,replace=False)].copy()
        for _ in range(140):
            a=((A[:,None]-C[None])**2).sum(-1).argmin(1)
            for j in range(k):
                if (a==j).any(): C[j]=A[a==j].mean(0)
        return np.linalg.norm(A-C[a],axis=1).mean()
    dr=km(X,0); rs=[km_ratio(X,dr,km,s) for s in range(nb)]
    return float(np.mean(rs)),float(np.std(rs))
def km_ratio(X,dr,km,s):
    Xp=X.copy(); rng=np.random.default_rng(s+7)
    for j in range(X.shape[1]): Xp[:,j]=rng.permutation(Xp[:,j])
    return dr/max(km(Xp,s),1e-8)

site=model.model.layers[0].self_attn.q_proj
def descriptors(texts):
    ds=[]
    def h(mod,args):
        x=args[0]
        if torch.is_tensor(x): ds.append(F.normalize(x.float().mean(1),dim=-1).cpu().numpy())
    hd=site.register_forward_pre_hook(h)
    for t in texts:
        ids=tok(t,return_tensors="pt",truncation=True,max_length=256).input_ids.cuda()
        model(ids)
    hd.remove()
    return np.concatenate(ds,0)

# --- corpora, all n=40 ---
# LOW diversity: near-duplicate templated prompts (single-slot variation)
countries=["France","Japan","Brazil","Egypt","Canada","India","Spain","Peru",
    "Kenya","Norway","Chile","Ghana","Nepal","Cuba","Mali","Laos","Fiji","Oman",
    "Togo","Iraq"]
low=[f"The capital city of {c} is" for c in countries] + \
    [f"The largest river in {c} is" for c in countries]        # 40, structurally identical
# MEDIUM: many WikiText articles (varied topics, natural prose)
wt=[t for t in load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")["text"] if len(t)>200][:40]
# HIGH: WikiText + C4 web mix (maximally varied)
c4=[]
for r in load_dataset("allenai/c4","en",split="validation",streaming=True):
    if len(r["text"])>200: c4.append(r["text"])
    if len(c4)>=20: break
high=wt[:20]+c4

res={}
for label,texts in [("low_diversity_near_duplicate",low),
                    ("medium_many_wikitext",wt),
                    ("high_wikitext+c4",high)]:
    X=descriptors(texts); c,cs=compactness(X)
    res[label]=dict(n=int(len(X)),c=round(c,3),std=round(cs,3),
                    decision="FIRE" if c<=0.4 else "SKIP")
    print(f"  {label:32s} n={len(X)} c={c:.3f}±{cs:.3f} -> {'FIRE' if c<=0.4 else 'SKIP'}",flush=True)
json.dump(res,open("results/q3_v2_results.json","w"),indent=2)
print("saved.")
