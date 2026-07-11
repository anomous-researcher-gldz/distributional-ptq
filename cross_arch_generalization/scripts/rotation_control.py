"""Reviewer 5dKj (Q2/W-C): the EXACT same-base-quantizer rotation control.

Table 2 contrasts QuaRot (rotation) against RTN/GPTQ/AWQ (no rotation), but those
are different base quantizers, so rotation is not a clean on/off toggle. Here we
hold the base quantizer fixed (RTN, W4A4 on LLaMA-3-8B) and toggle rotation and DBAF
independently — the 2x2 the reviewer asked for:

    (1) RTN                 (no rotation, no DBAF)
    (2) RTN + rotation      (rotation only)
    (3) RTN + DBAF          (DBAF only)
    (4) RTN + rotation + DBAF

Rotation = insert an orthogonal H before quantization: y = xW^T = (xH)(WH)^T with
H H^T = I, so it is exact in full precision but spreads outliers across channels,
which is the mechanism DBAF also targets. We use a seeded random orthogonal H per
input dimension (a valid rotation; QuaRot uses a Hadamard only for speed). WikiText-2
PPL, W4A4, alpha=0.25 frozen — identical protocol to the paper's headline host.

Hypothesis (rotation cliff, same base quantizer): rotation alone rescues RTN from
collapse, DBAF alone rescues it, and rotation+DBAF is no better than rotation alone
— i.e. DBAF and rotation overlap, and DBAF's value appears exactly where rotation is
absent.
"""
import sys, gc, json
import torch, torch.nn as nn
sys.path.insert(0, "/home/ubuntu/distributional-ptq")
sys.path.insert(0, "/home/ubuntu/distributional-ptq/FlatQuant")
from flatquant.baselines.rtn import _quantize_tensor_uniform, _quantize_per_channel_with_dbaf
from flatquant.baselines.act_quant import (_quantize_per_token, _dbaf_fold_per_token,
                                           _dbaf_unfold_per_token, apply_w4a4_act_quant)
from flatquant.eval_utils import ppl_eval
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
torch.set_grad_enabled(False)

M = "NousResearch/Meta-Llama-3-8B"
WBITS = ABITS = 4
ALPHA = 0.25
WINDOWS = 16
SEQ = 2048
SEED = 0
OUT = "/home/ubuntu/distributional-ptq/cross_arch_generalization/results/rotation_control_results.json"

tok = AutoTokenizer.from_pretrained(M)
class Enc:
    def __init__(self, ids): self.input_ids = ids
wt = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ENC = Enc(tok("\n\n".join(wt["text"]), return_tensors="pt").input_ids[:, :WINDOWS * SEQ])
print(f"WikiText-2 eval: {ENC.input_ids.shape[1] // SEQ} windows of {SEQ}", flush=True)

# ---- seeded orthogonal rotations, one per input dimension ----
_ROT = {}
def rot_for(d):
    if d not in _ROT:
        g = torch.Generator(device="cpu").manual_seed(SEED + d)
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float32))
        _ROT[d] = Q.to("cuda", torch.float16)
    return _ROT[d]

# orthogonality + full-precision exactness unit check
_H = rot_for(256)
assert torch.allclose(_H.float() @ _H.float().T, torch.eye(256, device="cuda"), atol=1e-3), "H not orthogonal"
_x = torch.randn(4, 256, device="cuda", dtype=torch.float16); _W = torch.randn(64, 256, device="cuda", dtype=torch.float16)
_lhs = _x @ _W.T; _rhs = (_x @ _H) @ (_W @ _H).T
assert (_lhs - _rhs).abs().mean() < 0.5, "rotation identity broken"
print("rotation sanity OK (orthogonal + fp-exact)", flush=True)

class RotActWrapper(nn.Module):
    """Rotate activation by H, (optionally DBAF-fold), per-token quant, then Linear
    whose weight has been pre-rotated to W H and quantized."""
    def __init__(self, linear, bits, H, use_dbaf, alpha):
        super().__init__()
        self.linear = linear; self.bits = bits; self.H = H
        self.use_dbaf = use_dbaf; self.alpha = alpha
    def forward(self, x):
        xr = x.to(torch.float16) @ self.H
        if self.use_dbaf:
            folded, T, mask, sgn = _dbaf_fold_per_token(xr, self.alpha)
            q = _quantize_per_token(folded, self.bits)
            xrq = _dbaf_unfold_per_token(q, T, mask, sgn, self.alpha)
        else:
            xrq = _quantize_per_token(xr, self.bits)
        return self.linear(xrq.to(x.dtype))

def fresh():
    return AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float16).cuda().eval()

def targets(m):
    return [(n, mod) for n, mod in m.named_modules()
            if isinstance(mod, nn.Linear) and mod.weight.dim() == 2 and "lm_head" not in n]

def build_norot(m, dbaf):
    for n, mod in targets(m):
        w = mod.weight.data
        wq = (_quantize_per_channel_with_dbaf(w, WBITS, alpha=ALPHA) if dbaf
              else _quantize_tensor_uniform(w, WBITS, per_channel=True))
        mod.weight.data = wq.to(mod.weight.dtype)
    apply_w4a4_act_quant(m, ABITS, use_dbaf=dbaf, alpha=ALPHA)

def build_rot(m, dbaf):
    # rotate + quantize weights, then wrap each linear with a rotating act-quant wrapper
    reps = []
    for n, mod in targets(m):
        H = rot_for(mod.weight.shape[1])           # H over d_in
        wr = (mod.weight.data.to(torch.float16) @ H)  # W H
        wrq = (_quantize_per_channel_with_dbaf(wr, WBITS, alpha=ALPHA) if dbaf
               else _quantize_tensor_uniform(wr, WBITS, per_channel=True))
        mod.weight.data = wrq.to(mod.weight.dtype)
        reps.append((n, RotActWrapper(mod, ABITS, H, dbaf, ALPHA)))
    mods = dict(m.named_modules())
    for n, wrap in reps:
        parent = mods[n.rsplit(".", 1)[0]]; child = n.rsplit(".", 1)[1]
        setattr(parent, child, wrap)

CONFIGS = [
    ("FP",                 None),
    ("RTN",                lambda m: build_norot(m, False)),
    ("RTN+rot",            lambda m: build_rot(m, False)),
    ("RTN+DBAF",           lambda m: build_norot(m, True)),
    ("RTN+rot+DBAF",       lambda m: build_rot(m, True)),
]

res = {"model": M, "w_bits": WBITS, "a_bits": ABITS, "alpha": ALPHA,
       "windows": WINDOWS, "rotation": "seeded random orthogonal per d_in", "ppl": {}}
for tag, build in CONFIGS:
    m = fresh()
    if build is not None: build(m)
    p = float(ppl_eval(m, ENC))
    res["ppl"][tag] = round(p, 3)
    print(f"[{tag:14s}] WikiText-2 PPL = {p:.3f}", flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()

json.dump(res, open(OUT, "w"), indent=2)
print("\n=== 2x2 rotation control (RTN base, W4A4, WikiText-2 PPL) ===")
print(f"                no DBAF     +DBAF")
print(f"  no rotation   {res['ppl']['RTN']:>8.1f}   {res['ppl']['RTN+DBAF']:>8.1f}")
print(f"  + rotation    {res['ppl']['RTN+rot']:>8.1f}   {res['ppl']['RTN+rot+DBAF']:>8.1f}")
print(f"  (FP reference {res['ppl']['FP']:.2f})")
print("saved ->", OUT)
