#!/usr/bin/env bash
# Phase C — TesseraQ row of the EMNLP 2026 host-matrix.
#
# Cells: {alone, +DBAF}  (pcsa_tf and dbaf+pcsa_tf gated out per
# docs/superpowers/notes/2026-05-15-pcsa-tf-gate-result.md)
#
# Pipeline: AWQ pre-compute (~30 min) -> TesseraQ block recon (~3-5h)
#
# Optional env overrides:
#   OUT_ROOT     — output root  (default: /data/outputs/HM-tesseraq)
#   MODEL_PATH   — local LLM path  (default: /data/modelzoo/meta-llama/Meta-Llama-3-8B)
#   AUGMENTS     — comma-separated subset  (default: alone,dbaf)
set -uo pipefail

cd /home/ubuntu/unifying-ptq
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate tesseraq

OUT_ROOT="${OUT_ROOT:-/data/outputs/HM-tesseraq}"
MODEL_PATH="${MODEL_PATH:-/data/modelzoo/meta-llama/Meta-Llama-3-8B}"
AUGS="${AUGMENTS:-alone,dbaf}"

AWQ_CONFIG="/home/ubuntu/unifying-ptq/patches/tesseraq_emnlp_configs/awq_llama3_8b_w4a4.yml"
TESSERAQ_CONFIG="/home/ubuntu/unifying-ptq/patches/tesseraq_emnlp_configs/tesseraq_w4a4_L3_8b.yml"

# === Stage 1: AWQ pre-compute (saves scales/clips to ../cache/activations/L3_8b/awq_w4a4) ===
# This is required by TesseraQ's load_transform step. Idempotent: skip if cache dir already populated.
AWQ_CACHE="/home/ubuntu/unifying-ptq/cache/activations/L3_8b/awq_w4a4"
if [[ -d "$AWQ_CACHE" && -n "$(ls -A "$AWQ_CACHE" 2>/dev/null)" ]]; then
  echo "[skip] AWQ pre-compute already cached at $AWQ_CACHE"
else
  echo "===== AWQ pre-compute (Llama-3-8B W4A4, ~30 min) ====="
  cd /home/ubuntu/unifying-ptq/TesseraQ
  export PYTHONPATH=/home/ubuntu/unifying-ptq/TesseraQ:$PYTHONPATH
  python llmc/__main__.py --config "$AWQ_CONFIG" --task_id awq-precompute 2>&1 | tail -50
  cd /home/ubuntu/unifying-ptq
fi

# === Stage 2: TesseraQ block reconstruction × {alone, dbaf} ===
IFS=',' read -ra AUG_ARR <<<"$AUGS"
for AUG in "${AUG_ARR[@]}"; do
  OUT_DIR="${OUT_ROOT}/llama3-8b/tesseraq_${AUG}"
  if [[ -f "${OUT_DIR}/eval.json" ]]; then
    echo "[skip] tesseraq $AUG already done"
    continue
  fi
  mkdir -p "$OUT_DIR"
  echo "===== TesseraQ $AUG (Llama-3-8B W4A4) ====="

  # Toggle DBAF via env var (read by patches/tesseraq_dbaf_pcsa_patch.py)
  if [[ "$AUG" == *dbaf* ]]; then
    export TESSERAQ_DBAF=1
  else
    unset TESSERAQ_DBAF
  fi

  cd /home/ubuntu/unifying-ptq/TesseraQ
  export PYTHONPATH=/home/ubuntu/unifying-ptq/TesseraQ:$PYTHONPATH
  python llmc/__main__.py \
    --config "$TESSERAQ_CONFIG" \
    --task_id "tesseraq-${AUG}" 2>&1 | tee "${OUT_DIR}/run.log" | tail -10
  cd /home/ubuntu/unifying-ptq

  # Extract PPL into eval.json (TesseraQ logs are in TesseraQ/ working dir)
  python - "$OUT_DIR" <<'PY'
import json, os, re, sys, pathlib
out_dir = pathlib.Path(sys.argv[1])
log_text = (out_dir / "run.log").read_text()
ppl_wt2 = None; ppl_c4 = None
for line in log_text.splitlines():
    m = re.search(r"wikitext2.*?ppl[^\d]*([\d.]+)", line, re.IGNORECASE)
    if m and ppl_wt2 is None: ppl_wt2 = float(m.group(1))
    m = re.search(r"\bc4\b.*?ppl[^\d]*([\d.]+)", line, re.IGNORECASE)
    if m and ppl_c4 is None: ppl_c4 = float(m.group(1))
out = {
    "target": "llama3-8b",
    "method": "tesseraq",
    "augments": out_dir.name.split("_", 1)[-1],
    "metrics": {"wikitext2_ppl": ppl_wt2, "c4_ppl": ppl_c4},
}
(out_dir / "eval.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY
done

echo "TESSERAQ_PHASE_C_DONE_$?"
