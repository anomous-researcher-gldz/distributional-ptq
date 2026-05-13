"""Needle-in-a-haystack eval for FlatQuant-calibrated LLaMA-3-8B.

Self-contained: builds long synthetic contexts with random key/value pairs;
asks the model to retrieve a value; scores by substring match.

This is the minimal eval to compare:
  - baseline:    FlatQuant + DBAF + PCSA, W4A4 KV4 (no KV-PCSA)
  - kv_pcsa_v2:  FlatQuant + DBAF + PCSA + KV-PCSA v2 (per-token x anchor mult)

at multiple context lengths to test whether KV-PCSA actually helps long-context
retrieval (the only place the per-prompt routing should pay off — uniform
WikiText/C4 PPL shows it doesn't help short-context).
"""
from __future__ import annotations
import sys, argparse, json, pathlib, random, time
import torch

sys.path.insert(0, "/home/ubuntu/unifying-ptq/FlatQuant")


def build_niah_prompt(rng: random.Random, n_pairs: int, needle_pair: tuple[str, str]) -> tuple[str, str]:
    """Build a needle-in-haystack prompt.

    n_pairs random distractor key/value lines plus one target (needle) line,
    placed at a random position. Asks for the needle's value.
    """
    key, value = needle_pair
    distractor_pairs = []
    for _ in range(n_pairs - 1):
        k = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=10))
        v = "".join(rng.choices("0123456789", k=8))
        distractor_pairs.append((k, v))
    needle_idx = rng.randint(0, n_pairs - 1)
    distractor_pairs.insert(needle_idx, (key, value))
    body = "\n".join(f"The magic code for {k} is {v}." for k, v in distractor_pairs)
    prompt = (
        "Below is a list of facts.\n\n"
        f"{body}\n\n"
        f"Question: What is the magic code for {key}?\n"
        "Answer: The magic code for "
        f"{key} is "
    )
    return prompt, value


@torch.no_grad()
def run_niah(model, tok, args, calibrated_kv_pcsa: bool, out_path: str):
    rng = random.Random(0)
    results_by_context = {}
    for target_seq_len in args.context_lengths:
        # Calibrate n_pairs so prompt has ~target_seq_len tokens.
        # Each pair ≈ 16 tokens. Account for prefix/suffix.
        n_pairs = max(8, target_seq_len // 16)
        n_correct = 0
        n_total = 0
        per_sample = []
        for sample_idx in range(args.num_samples):
            key = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=10))
            value = "".join(rng.choices("0123456789", k=8))
            prompt, gold = build_niah_prompt(rng, n_pairs, (key, value))
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            prompt_len = ids.shape[1]
            if prompt_len > args.max_seq_length:
                continue
            out = model.generate(
                ids, max_new_tokens=args.tokens_to_generate,
                do_sample=False, temperature=1.0, top_p=1.0,
                pad_token_id=tok.eos_token_id,
            )
            gen_text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
            is_correct = gold in gen_text
            if is_correct:
                n_correct += 1
            n_total += 1
            per_sample.append({"prompt_len": prompt_len, "gold": gold,
                               "gen": gen_text[:120], "correct": is_correct})
        acc = n_correct / max(n_total, 1)
        print(f"[NIAH] context={target_seq_len} n_pairs={n_pairs} "
              f"n_eval={n_total} acc={acc:.3f}", flush=True)
        results_by_context[target_seq_len] = {"n_pairs": n_pairs, "n_eval": n_total,
                                              "acc": acc, "samples": per_sample}

    summary = {"matrix_path": args.matrix_path, "kv_pcsa": calibrated_kv_pcsa,
               "results": results_by_context}
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v["acc"] for k, v in results_by_context.items()}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix_path", required=True,
                   help="Calibration output dir with flat_matrices.pth")
    p.add_argument("--kv_pcsa", action="store_true",
                   help="Set if calibration used --kv-pcsa")
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--context_lengths", nargs="+", type=int, default=[2048, 4096, 8192])
    p.add_argument("--tokens_to_generate", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--model", default="/data/modelzoo/meta-llama/Meta-Llama-3-8B")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Build args object compatible with apply_flatquant_to_llama.
    class FQArgs: pass
    a = FQArgs()
    a.w_bits = 4; a.a_bits = 4; a.k_bits = 4; a.v_bits = 4; a.q_bits = 16
    a.w_groupsize = -1; a.a_groupsize = -1
    a.w_asym = False; a.a_asym = False
    a.k_asym = True; a.v_asym = True
    a.k_groupsize = 128; a.v_groupsize = 128
    a.lwc = True; a.lac = True
    a.cali_trans = True; a.add_diag = True
    a.direct_inv = False
    a.diag_init = "sq_style"
    a.kv_pcsa = args.kv_pcsa; a.kv_pcsa_anchors = 4
    a.disable_pcsa = False; a.disable_dbaf = False
    a.no_dbaf_gate = False
    a.separate_vtrans = False

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[load] tokenizer + model", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 device_map="cuda", low_cpu_mem_usage=True)
    from flatquant.model_tools.llama_utils import apply_flatquant_to_llama
    apply_flatquant_to_llama(a, model)
    matrices = torch.load(f"{args.matrix_path}/flat_matrices.pth", map_location="cuda", weights_only=False)
    params = torch.load(f"{args.matrix_path}/flat_parameters.pth", map_location="cuda", weights_only=False)
    model.load_state_dict(matrices, strict=False)
    model.load_state_dict(params, strict=False)
    pcsa_path = pathlib.Path(f"{args.matrix_path}/pcsa_state.pth")
    if pcsa_path.exists():
        pcsa = torch.load(str(pcsa_path), map_location="cuda", weights_only=False)
        model.load_state_dict(pcsa, strict=False)
        print(f"[load] PCSA state loaded ({len(pcsa)} tensors)", flush=True)
    for m in model.modules():
        if hasattr(m, "_eval_mode"):
            m._eval_mode = True
    model.eval()
    print("[load] ready", flush=True)
    run_niah(model, tok, args, calibrated_kv_pcsa=args.kv_pcsa, out_path=args.out)


if __name__ == "__main__":
    main()
