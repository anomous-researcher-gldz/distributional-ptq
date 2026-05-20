#!/usr/bin/env bash
# HM Phase A W4A4 redo — RTN/GPTQ/AWQ × {alone, dbaf} at W4A4 on Llama-3-8B.
# Makes Phase A apples-to-apples with Phase B/C (which are both W4A4).
#
# SmoothQuant is already W4A4 in the original Phase A and is reused as-is.
# PCSA-tf variants are not in scope (training-free PCSA-tf was gated out
# earlier; see docs/superpowers/notes/2026-05-15-pcsa-tf-gate-result.md).
set -uo pipefail
cd /home/ubuntu/unifying-ptq
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate unifyptq
: "${PYTHONPATH:=}"

OUT_ROOT="${OUT_ROOT:-/data/outputs/HM-phase-a-w4a4}"
TARGET=llama3-8b
METHODS=(rtn gptq awq)
AUGMENTS=(alone dbaf)

cell_idx=0
total=$((${#METHODS[@]} * ${#AUGMENTS[@]}))

for METHOD in "${METHODS[@]}"; do
  for AUG in "${AUGMENTS[@]}"; do
    cell_idx=$((cell_idx + 1))
    OUT_DIR="${OUT_ROOT}/${TARGET}/${METHOD}_${AUG}"
    OUT_PATH="${OUT_DIR}/eval.json"
    if [[ -f "$OUT_PATH" ]]; then
      echo "===== [${cell_idx}/${total}] SKIP (exists): ${METHOD}_${AUG} ====="
      continue
    fi
    echo "===== [${cell_idx}/${total}] W4A4 ${TARGET} method=${METHOD} aug=${AUG} ====="
    mkdir -p "$OUT_DIR"
    python scripts/run_training_free_full_table.py \
      --target "$TARGET" \
      --method "$METHOD" \
      --augments "$AUG" \
      --act_bits 4 \
      --out "$OUT_PATH" 2>&1 | tee "${OUT_DIR}/run.log"
    echo "===== [${cell_idx}/${total}] DONE: ${METHOD}_${AUG} ====="
  done
done
echo "HM_PHASE_A_W4A4_DONE_$?"
