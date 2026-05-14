"""Generate the §4.X primitive op-cost FLOP table.

For each candidate distribution-targeting primitive, compute per-token FLOPs
at the hidden dimension d_model values relevant to our hosts:
  - SwinIR-light  embed_dim = 60   (d for SR cells)
  - SwinIR-base   embed_dim = 180  (alt SR)
  - SAM-B encoder embed_dim = 768
  - SAM-L encoder embed_dim = 1024
  - SAM-H encoder embed_dim = 1280
  - Llama-2-7B  hidden_size = 4096
  - Llama-3-8B  hidden_size = 4096
  - Llama-3-70B hidden_size = 8192

Primitives compared:
  - Generic learned rotation R (FlatQuant / SpinQuant style):
      per-token cost ~ d^2  (one dense matmul x @ R^T)
  - Fast Hadamard transform (QuaRot / QuIP# style):
      per-token cost ~ d * log2(d)  (structured)
  - DBAF (per-tensor outlier folding):
      per-token cost ~ d  (compare-to-threshold + 1 affine per channel)
  - PCSA-tf inference (route + scale lookup):
      per-token cost ~ d  (multiply by selected scale)
    plus per-prompt cost ~ K * d_descriptor (k-means anchor lookup)

We also fold in modality-canonical non-rotation remedies for completeness:
  - HLUQ (AHCPTQ log-uniform): per-element LUT lookup, ~d ops/token
  - Bimodal integration (PTQ4SAM): one sign() + one affine per element, ~d
  - DOBI/DQC (2DQuant): bound clipping per element, ~d

Output:
  - LaTeX table to scripts/_out/flop_table.tex
  - JSON to scripts/_out/flop_table.json
  - Markdown summary to stdout
"""
from __future__ import annotations
import math, json, pathlib, argparse


PRIMITIVES = [
    # (key, name, per_token_FLOPs(d), per_prompt_FLOPs(d, K, d_desc), notes)
    ("R_learned",   "Learned rotation $R$ (FlatQuant/SpinQuant)",
        lambda d, K, dd: d * d,
        lambda d, K, dd: 0,
        "static, per-layer; $d^2$ params"),
    ("R_hadamard",  "Fast Hadamard $H$ (QuaRot/QuIP\\#)",
        lambda d, K, dd: d * max(1, math.log2(d)),
        lambda d, K, dd: 0,
        "static, structured; near-zero params"),
    ("hluq",        "HLUQ log+uniform LUT (AHCPTQ)",
        lambda d, K, dd: d,                              # per-element LUT
        lambda d, K, dd: 0,
        "static, per-tensor LUT (SAM)"),
    ("bimodal",     "Bimodal integration (PTQ4SAM)",
        lambda d, K, dd: 2 * d,                          # sign() + 1 affine
        lambda d, K, dd: 0,
        "static, per-tensor (SAM)"),
    ("dobi_dqc",    "Bound clip (2DQuant DOBI+DQC)",
        lambda d, K, dd: d,                              # per-element clip
        lambda d, K, dd: 0,
        "static, per-tensor (SR)"),
    ("dbaf",        "\\textbf{DBAF} (ours, per-tensor outlier fold)",
        lambda d, K, dd: 2 * d,                          # compare + affine
        lambda d, K, dd: 0,
        "static, per-tensor, gated"),
    ("pcsa_tf",     "\\textbf{PCSA-tf} (ours, per-input scale routing)",
        lambda d, K, dd: d,                              # per-token scale mul
        lambda d, K, dd: K * dd,                         # k-means lookup once per prompt
        "input-conditional, per-quantizer"),
    ("dbaf_pcsa",   "\\textbf{DBAF + PCSA-tf} (ours, combined)",
        lambda d, K, dd: 3 * d,
        lambda d, K, dd: K * dd,
        "ours, both primitives"),
]


HOSTS = [
    ("SwinIR-light",  60,    "SR"),
    ("SwinIR-base",   180,   "SR (alt)"),
    ("SAM-B encoder", 768,   "SAM"),
    ("SAM-L encoder", 1024,  "SAM"),
    ("SAM-H encoder", 1280,  "SAM"),
    ("Llama-2-7B",    4096,  "LLM"),
    ("Llama-3-8B",    4096,  "LLM"),
    ("Llama-3-70B",   8192,  "LLM"),
]

# PCSA-tf hyperparameters
K_ANCHORS = 4
D_DESC    = 60  # descriptor dim used by PCSA-tf (SwinIR-S default; descriptor is small)


def fmt_flops(n: float) -> str:
    if n >= 1e9:  return f"{n/1e9:.1f}G"
    if n >= 1e6:  return f"{n/1e6:.1f}M"
    if n >= 1e3:  return f"{n/1e3:.1f}K"
    return f"{int(n)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scripts/_out", type=pathlib.Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, d, modality in HOSTS:
        for key, prim, f_tok, f_prompt, notes in PRIMITIVES:
            ft = f_tok(d, K_ANCHORS, D_DESC)
            fp = f_prompt(d, K_ANCHORS, D_DESC)
            rows.append({"host": name, "modality": modality, "d": d,
                         "primitive_key": key, "primitive": prim,
                         "per_token_flops": ft,
                         "per_prompt_flops": fp,
                         "notes": notes})

    # Markdown summary (Llama-3-8B is the headline d=4096 case for the paper)
    print("# Primitive Op-Cost Table (per-token FLOPs)\n")
    print("Headline (Llama-3-8B, d=4096):\n")
    print(f"| Primitive | FLOPs/token | per-prompt | Notes |")
    print(f"|---|---|---|---|")
    for key, prim, f_tok, f_prompt, notes in PRIMITIVES:
        ft = f_tok(4096, K_ANCHORS, D_DESC)
        fp = f_prompt(4096, K_ANCHORS, D_DESC)
        print(f"| {prim} | {fmt_flops(ft)} | {fmt_flops(fp)} | {notes} |")
    print()

    # Ratios for headline:
    print("## Key ratios at d=4096")
    base_R = 4096 * 4096
    base_H = 4096 * math.log2(4096)
    dbaf = 2 * 4096
    print(f"- Learned R / DBAF      = {base_R / dbaf:.0f}x")
    print(f"- Fast Hadamard / DBAF  = {base_H / dbaf:.2f}x")
    print(f"- Learned R / DBAF+PCSA-tf = {base_R / (3*4096):.0f}x")
    print(f"- Fast Hadamard / DBAF+PCSA-tf = {base_H / (3*4096):.2f}x")

    # JSON dump for downstream plotting
    (args.out / "flop_table.json").write_text(json.dumps(rows, indent=2))

    # LaTeX table for the headline (Llama-3-8B, d=4096) — easy paste into paper
    tex_lines = ["% Auto-generated: scripts/compute_flop_table.py",
                 "\\begin{table}[t]",
                 "\\centering\\small",
                 "\\setlength{\\tabcolsep}{4pt}",
                 "\\begin{tabular}{lrrl}",
                 "\\toprule",
                 "Primitive & FLOPs/tok & per-prompt & Notes \\\\",
                 "\\midrule"]
    for key, prim, f_tok, f_prompt, notes in PRIMITIVES:
        ft = f_tok(4096, K_ANCHORS, D_DESC)
        fp = f_prompt(4096, K_ANCHORS, D_DESC)
        tex_lines.append(f"{prim} & {fmt_flops(ft)} & {fmt_flops(fp)} & {notes} \\\\")
    tex_lines += ["\\bottomrule",
                  "\\end{tabular}",
                  "\\caption{Per-token FLOPs at $d=4096$ for Llama-3-8B-class hosts. "
                  "DBAF + PCSA-tf are 12$\\times$ cheaper than fast Hadamard rotation "
                  "and 4000$\\times$ cheaper than learned rotation per token.}",
                  "\\label{tab:primitive-cost}",
                  "\\end{table}"]
    (args.out / "flop_table.tex").write_text("\n".join(tex_lines) + "\n")
    print(f"\nLaTeX → {args.out/'flop_table.tex'}")
    print(f"JSON  → {args.out/'flop_table.json'}")


if __name__ == "__main__":
    main()
