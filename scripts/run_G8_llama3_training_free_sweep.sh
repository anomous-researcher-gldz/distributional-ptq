#!/usr/bin/env bash
# G8 — Training-free LLM sweep on Llama-3-8B (16 cells = 4 methods x 4 augments).
# Methods: rtn, gptq, awq, smoothquant
# Augments: alone, dbaf, pcsa_tf, dbaf+pcsa_tf
#
# Each cell is a separate `python run_training_free_full_table.py` invocation
# so we can resume on a single-cell failure (the per-cell driver auto-skips
# existing eval.json files).
set -uo pipefail
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq

OUT_ROOT=/data/outputs/G8-training-free-full
mkdir -p "$OUT_ROOT"

TARGET=llama3-8b
METHODS=(rtn gptq awq smoothquant)
AUGMENTS=(alone dbaf pcsa_tf dbaf+pcsa_tf)

cell_idx=0
total=$((${#METHODS[@]} * ${#AUGMENTS[@]}))

for METHOD in "${METHODS[@]}"; do
  for AUG in "${AUGMENTS[@]}"; do
    cell_idx=$((cell_idx + 1))
    OUT_DIR="${OUT_ROOT}/${TARGET}/${METHOD}_${AUG}"
    OUT_PATH="${OUT_DIR}/eval.json"
    echo "===== [${cell_idx}/${total}] ${TARGET} method=${METHOD} aug=${AUG} ====="
    mkdir -p "$OUT_DIR"
    python scripts/run_training_free_full_table.py \
      --target "$TARGET" \
      --method "$METHOD" \
      --augments "$AUG" \
      --out "$OUT_PATH" 2>&1 | tee "$OUT_DIR/run.log"
    echo "===== [${cell_idx}/${total}] DONE: ${METHOD}_${AUG} ====="
  done
done
echo "G8_LLAMA3_TRAINING_FREE_ALL_DONE_$?"
