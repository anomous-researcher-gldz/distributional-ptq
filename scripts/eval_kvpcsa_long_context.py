"""Evaluate FlatQuant-calibrated LLaMA-3-8B on WikiText-2 at multiple
context lengths to test the KV-cache PCSA hypothesis.

If KV-PCSA helps long-context retention, the PPL gap between
(baseline) and (KV-PCSA) should *widen* as seq_len grows.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")


@torch.no_grad()
def wikitext_ppl(model, tokenizer, seq_len: int, n_chunks_cap: int = 32) -> float:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    n_chunks = min(n_chunks_cap, ids.shape[1] // seq_len)
    nlls = []
    for i in range(n_chunks):
        chunk = ids[:, i * seq_len:(i + 1) * seq_len]
        out = model(chunk, labels=chunk)
        nlls.append(out.loss.float().item())
    return float(torch.tensor(nlls).mean().exp().item())


@torch.no_grad()
def c4_ppl(model, tokenizer, seq_len: int, n_chunks_cap: int = 32) -> float:
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    text_parts, total_tokens, target_tokens = [], 0, seq_len * n_chunks_cap * 2
    for sample in ds:
        text_parts.append(sample["text"])
        total_tokens += len(sample["text"]) // 4
        if total_tokens > target_tokens:
            break
    text = "\n\n".join(text_parts)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    n_chunks = min(n_chunks_cap, ids.shape[1] // seq_len)
    nlls = []
    for i in range(n_chunks):
        chunk = ids[:, i * seq_len:(i + 1) * seq_len]
        out = model(chunk, labels=chunk)
        nlls.append(out.loss.float().item())
    return float(torch.tensor(nlls).mean().exp().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-path", required=True,
                   help="Path to the calibration output dir (contains flat_matrices.pth)")
    p.add_argument("--kv-pcsa", action="store_true",
                   help="Set if this calibration used --kv-pcsa")
    p.add_argument("--seq-lens", nargs="+", type=int, default=[2048, 4096, 8192])
    p.add_argument("--label", required=True, help="Name for this run in results")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Construct args object compatible with FlatQuant's apply_flatquant_to_llama
    class FQArgs:
        pass
    a = FQArgs()
    a.w_bits = 4; a.a_bits = 4; a.k_bits = 4; a.v_bits = 4; a.q_bits = 16
    a.w_groupsize = -1; a.a_groupsize = -1
    a.w_asym = False; a.a_asym = False
    a.k_asym = True; a.v_asym = True
    a.k_groupsize = 128; a.v_groupsize = 128
    a.lwc = True; a.lac = True
    a.cali_trans = True; a.add_diag = True
    a.direct_inv = False
    a.kv_pcsa = args.kv_pcsa; a.kv_pcsa_anchors = 4
    a.disable_pcsa = False
    a.disable_dbaf = False
    a.separate_vtrans = False
    a.diag_init = "sq_style"
    a.q_asym = False

    print(f"[eval] loading LLaMA-3-8B + reloading calibrated matrices", flush=True)
    import transformers
    from transformers import AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("/data/modelzoo/meta-llama/Meta-Llama-3-8B")
    cfg = transformers.LlamaConfig.from_pretrained("/data/modelzoo/meta-llama/Meta-Llama-3-8B")
    cfg._attn_implementation_internal = "eager"
    model = transformers.LlamaForCausalLM.from_pretrained(
        "/data/modelzoo/meta-llama/Meta-Llama-3-8B",
        config=cfg, torch_dtype="auto", low_cpu_mem_usage=True,
    )
    model.seqlen = 2048

    from flatquant.model_tools.llama_utils import apply_flatquant_to_llama
    apply_flatquant_to_llama(a, model)

    # Use FlatQuant's canonical load (rep_matrix_only + load matrices + load pcsa).
    from flatquant import flat_utils
    a.exp_dir = args.matrix_path
    flat_utils.load_flat_matrices(a, model, path=args.matrix_path)
    model.to("cuda")  # loaded clip factors land on CPU; move now so reparameterize sees one device
    flat_utils.reparameterize_model(model)
    print(f"[eval] loaded calibrated matrices + reparameterized model", flush=True)

    # Apply RTN weight quantization (mirrors main.py post-reparameterize step)
    import sys as _sys
    _sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")
    import gptq_utils
    a.gptq = False
    a.w_clip = True
    a.w_groupsize = -1
    a.act_order = False
    a.percdamp = 0.01
    a.nsamples = 128
    a.gptq_mse = False
    a.int8_down_proj = False
    gptq_utils.rtn_fwrd(model, "cuda", a)
    model.to("cuda")  # rtn_fwrd leaves layers on CPU
    print(f"[eval] applied RTN W4 weight quantization", flush=True)
    layers = model.model.layers

    # Set eval mode
    for m in model.modules():
        if hasattr(m, "_eval_mode"):
            m._eval_mode = True

    model.eval()
    # Make sure all buffers/params end up on GPU (loads landed on CPU for some).
    model.to("cuda")

    results_wt2, results_c4 = {}, {}
    for seq_len in args.seq_lens:
        try:
            t0 = time.time()
            ppl = wikitext_ppl(model, tok, seq_len=seq_len)
            results_wt2[seq_len] = ppl
            print(f"[eval] wt2 seq_len={seq_len}: PPL={ppl:.3f}  (took {time.time()-t0:.1f}s)", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[eval] wt2 seq_len={seq_len}: OOM", flush=True)
            results_wt2[seq_len] = None
            torch.cuda.empty_cache()
        try:
            t0 = time.time()
            ppl = c4_ppl(model, tok, seq_len=seq_len)
            results_c4[seq_len] = ppl
            print(f"[eval] c4  seq_len={seq_len}: PPL={ppl:.3f}  (took {time.time()-t0:.1f}s)", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[eval] c4  seq_len={seq_len}: OOM", flush=True)
            results_c4[seq_len] = None
            torch.cuda.empty_cache()

    out = {"label": args.label, "kv_pcsa": args.kv_pcsa, "matrix_path": args.matrix_path,
           "wikitext2_ppl_by_seqlen": results_wt2,
           "c4_ppl_by_seqlen": results_c4}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
