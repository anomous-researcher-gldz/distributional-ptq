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
    p.add_argument("--dry_run", action="store_true",
                   help="Print the command that would be run and exit.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# PPL parsing
# ---------------------------------------------------------------------------

def _parse_ppl(output: str) -> float | None:
    """Extract the WikiText-2 PPL from QuaRot's logging output.

    QuaRot's eval_utils.evaluator() logs:
        WIKITEXT2 PPL: 7.342
    """
    # Primary pattern — eval_utils logging.info line
    m = re.search(r"WIKITEXT2\s+PPL:\s*([\d.]+)", output, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Fallback: plain float at end of a line containing "PPL"
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

    # Skip if already done
    if out_path.exists():
        print(f"[quarot_cell] skipping {augment} — {out_path} already exists", flush=True)
        return json.loads(out_path.read_text())

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
        "--eval_dataset", "wikitext2",
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

    ppl = _parse_ppl(combined_output)
    if ppl is None:
        print("[quarot_cell] WARNING: could not parse PPL from output; storing nan",
              flush=True)
        ppl = float("nan")

    record = {
        "target": "llama3-8b",
        "method": "quarot",
        "augments": augment,
        "metrics": {"wikitext2_ppl": ppl},
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
