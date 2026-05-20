"""QuaRot single-cell runner for the EMNLP 2026 host-matrix (Phase B).

Runs QuaRot on Llama-3-8B with W4A4 + Hadamard rotation and optionally
the DBAF activation-folding patch. Writes an eval.json in the same schema
used by Phase A (run_training_free_full_table.py).

Usage:
    python scripts/run_quarot_one_cell.py \\
        --augment alone \\
        --out /data/outputs/HM-quarot/llama3-8b/quarot_alone/eval.json

    python scripts/run_quarot_one_cell.py \\
        --augment dbaf \\
        --out /data/outputs/HM-quarot/llama3-8b/quarot_dbaf/eval.json

Arguments:
    --augment   One of {alone, dbaf}.  pcsa_tf / dbaf+pcsa_tf are gated out
                per 2026-05-15-pcsa-tf-gate-result.md.
    --model     HuggingFace model ID or local path (default: Llama-3-8B).
    --out       Path for the output eval.json.
    --w_bits    Weight bits (default: 4).
    --a_bits    Activation bits (default: 4).
    --nsamples  GPTQ calibration samples (default: 128).
    --bsz       Eval batch size (default: 32).
    --dry_run   Print the subprocess command and exit without running.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
_REPO = pathlib.Path(__file__).resolve().parent.parent
_QUAROT_ENTRY = _REPO / "QuaRot" / "fake_quant" / "main.py"

_DEFAULT_MODEL = "/data/modelzoo/meta-llama/Meta-Llama-3-8B"
_DEFAULT_OUT   = str(_REPO / "results" / "HM-quarot" / "llama3-8b" / "quarot_AUGMENT" / "eval.json")

# Augments whose pcsa_tf cell was gated out — refuse to run to avoid confusion.
_GATED_AUGMENTS = {"pcsa_tf", "dbaf+pcsa_tf"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="QuaRot single-cell driver (Phase B, EMNLP 2026 host-matrix)"
    )
    p.add_argument("--augment", choices=["alone", "dbaf"], required=True,
                   help="Augment variant: alone (QuaRot only) or dbaf (+DBAF folding).")
    p.add_argument("--model", default=_DEFAULT_MODEL,
                   help="Model path or HF ID (must be in QuaRot's supported_models list).")
    p.add_argument("--out", default=None,
                   help="Output eval.json path.  Defaults to results/HM-quarot/...")
    p.add_argument("--w_bits", type=int, default=4, help="Weight quantisation bits.")
    p.add_argument("--a_bits", type=int, default=4, help="Activation quantisation bits.")
    p.add_argument("--nsamples", type=int, default=128, help="GPTQ calibration samples.")
    p.add_argument("--bsz", type=int, default=32, help="Eval batch size.")
    p.add_argument("--eval_dataset", default="wikitext2",
                   choices=["wikitext2", "c4"],
                   help="Eval dataset (default: wikitext2).")
    p.add_argument("--merge_existing", action="store_true",
                   help="If --out already exists, merge new metric into it "
                        "instead of skipping. Used to backfill c4 onto an "
                        "existing wikitext2-only eval.json.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the command that would be run and exit.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# PPL parsing
# ---------------------------------------------------------------------------

def _parse_ppl(output: str, dataset: str = "wikitext2") -> float | None:
    """Extract the PPL for the given dataset from QuaRot's logging output.

    QuaRot's eval_utils.evaluator() logs e.g. `WIKITEXT2 PPL: 7.342` or `C4 PPL: 9.811`.
    """
    pat = rf"{re.escape(dataset.upper())}\s+PPL:\s*([\d.]+)"
    m = re.search(pat, output, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"PPL[^:\n]*:\s*([\d.]+)", output, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_cell(args) -> dict:
    augment = args.augment

    # Resolve output path
    if args.out is None:
        out_path = pathlib.Path(
            _DEFAULT_OUT.replace("AUGMENT", augment)
        )
    else:
        out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already done (unless we're backfilling another metric)
    metric_key = f"{args.eval_dataset}_ppl"
    if out_path.exists() and not args.merge_existing:
        existing = json.loads(out_path.read_text())
        if existing.get("metrics", {}).get(metric_key) is not None:
            print(f"[quarot_cell] skipping {augment} — {metric_key} already in {out_path}",
                  flush=True)
            return existing

    # Build QuaRot CLI
    cmd = [
        sys.executable,
        str(_QUAROT_ENTRY),
        "--model", args.model,
        "--w_bits", str(args.w_bits),
        "--a_bits", str(args.a_bits),
        "--rotate",
        "--rotate_mode", "hadamard",
        "--w_clip",
        "--cal_dataset", "wikitext2",
        "--nsamples", str(args.nsamples),
        "--eval_dataset", args.eval_dataset,
        "--bsz", str(args.bsz),
    ]

    # Environment — pass QUAROT_DBAF to the subprocess so the EMNLP hook fires.
    env = os.environ.copy()
    if augment == "dbaf":
        env["QUAROT_DBAF"] = "1"
    else:
        env.pop("QUAROT_DBAF", None)

    if args.dry_run:
        print("[quarot_cell] DRY RUN — would execute:")
        print("  ENV: QUAROT_DBAF=" + env.get("QUAROT_DBAF", "(unset)"))
        print("  CMD:", " ".join(cmd))
        return {}

    print(f"[quarot_cell] starting augment={augment}", flush=True)
    print(f"[quarot_cell] cmd: {' '.join(cmd)}", flush=True)

    # Run QuaRot once via Popen, teeing each output line to stdout and to a
    # buffer so we can parse the PPL at the end without a second run.
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        captured_lines.append(line)
    proc.wait()
    elapsed = time.time() - t0
    combined_output = "".join(captured_lines)

    if proc.returncode != 0:
        raise RuntimeError(
            f"QuaRot exited with code {proc.returncode} for augment={augment}"
        )

    ppl = _parse_ppl(combined_output, args.eval_dataset)
    if ppl is None:
        print(f"[quarot_cell] WARNING: could not parse {args.eval_dataset} PPL; storing nan",
              flush=True)
        ppl = float("nan")

    if args.merge_existing and out_path.exists():
        record = json.loads(out_path.read_text())
        record.setdefault("metrics", {})[metric_key] = ppl
        record["wallclock_seconds"] = elapsed
    else:
        record = {
            "target": "llama3-8b",
            "method": "quarot",
            "augments": augment,
            "metrics": {metric_key: ppl},
            "wallclock_seconds": elapsed,
        }
    out_path.write_text(json.dumps(record, indent=2))
    print(f"[quarot_cell] eval.json written to {out_path}", flush=True)
    print(json.dumps(record, indent=2))
    return record


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    if args.augment in _GATED_AUGMENTS:
        raise SystemExit(
            f"[quarot_cell] augment '{args.augment}' was gated out "
            f"(PCSA-tf gate FAIL per 2026-05-15-pcsa-tf-gate-result.md). "
            f"Only 'alone' and 'dbaf' are valid for the QuaRot row."
        )
    run_cell(args)


if __name__ == "__main__":
    main()
