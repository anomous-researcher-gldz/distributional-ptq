#!/usr/bin/env bash
# HM Phase E — TesseraQ + trained-PCSA row.
#
# Cells: {pcsa_alone, dbaf_pcsa}
#   pcsa_alone : TesseraQ block-recon with per-block trainable PCSA scales
#   dbaf_pcsa  : same + DBAF folding at IntegerQuantizer entry points
#
# Reuses the AWQ pre-compute cache from Phase C
# (cache/activations/L3_8b/awq_w4a4) if already populated.
#
# Optional env overrides:
#   OUT_ROOT     — output root (default: /data/outputs/HM-tesseraq-trained-pcsa)
#   MODEL_PATH   — local LLM path (default: /data/modelzoo/meta-llama/Meta-Llama-3-8B)
#   AUGMENTS     — comma-separated subset (default: pcsa_alone,dbaf_pcsa)
#   PCSA_K       — number of anchors (default: 8)
set -uo pipefail

cd /home/ubuntu/unifying-ptq
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate tesseraq
: "${PYTHONPATH:=}"
export PYTHONPATH="/home/ubuntu/unifying-ptq/FlatQuant:${PYTHONPATH}"

export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

OUT_ROOT="${OUT_ROOT:-/data/outputs/HM-tesseraq-trained-pcsa}"
MODEL_PATH="${MODEL_PATH:-/data/modelzoo/meta-llama/Meta-Llama-3-8B}"
AUGS="${AUGMENTS:-pcsa_alone,dbaf_pcsa}"
PCSA_K="${PCSA_K:-8}"

AWQ_CONFIG="/home/ubuntu/unifying-ptq/patches/tesseraq_emnlp_configs/awq_llama3_8b_w4a4.yml"
TESSERAQ_CONFIG="/home/ubuntu/unifying-ptq/patches/tesseraq_emnlp_configs/tesseraq_w4a4_L3_8b.yml"

# === Stage 1: AWQ pre-compute (reuse Phase C cache if present) ===
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

# === Stage 2: TesseraQ block-recon with trained-PCSA × {pcsa_alone, dbaf_pcsa} ===
IFS=',' read -ra AUG_ARR <<<"$AUGS"
for AUG in "${AUG_ARR[@]}"; do
  OUT_DIR="${OUT_ROOT}/llama3-8b/tesseraq_${AUG}"
  if [[ -f "${OUT_DIR}/eval.json" ]]; then
    echo "[skip] tesseraq $AUG already done"
    continue
  fi
  mkdir -p "$OUT_DIR"
  echo "===== TesseraQ $AUG (Llama-3-8B W4A4, K=${PCSA_K}) ====="

  export TESSERAQ_TRAINED_PCSA=1
  export TESSERAQ_TRAINED_PCSA_K="${PCSA_K}"
  if [[ "$AUG" == *dbaf* ]]; then
    export TESSERAQ_DBAF=1
  else
    unset TESSERAQ_DBAF
  fi

  cd /home/ubuntu/unifying-ptq/TesseraQ
  export PYTHONPATH=/home/ubuntu/unifying-ptq/TesseraQ:/home/ubuntu/unifying-ptq/FlatQuant:$PYTHONPATH
  python llmc/__main__.py \
    --config "$TESSERAQ_CONFIG" \
    --task_id "tesseraq-${AUG}" 2>&1 | tee "${OUT_DIR}/run.log" | tail -10
  cd /home/ubuntu/unifying-ptq

  python - "$OUT_DIR" <<'PY'
import json, re, sys, pathlib
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
    "method": "tesseraq_trained_pcsa",
    "augments": out_dir.name.split("_", 1)[-1],
    "metrics": {"wikitext2_ppl": ppl_wt2, "c4_ppl": ppl_c4},
}
(out_dir / "eval.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY
done

echo "TESSERAQ_PHASE_E_DONE_$?"
