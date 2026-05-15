#!/usr/bin/env bash
# Phase B — QuaRot row of the EMNLP 2026 LLM host-matrix.
#
# Cells: {alone, +DBAF}  (pcsa_tf and dbaf+pcsa_tf gated out per
# docs/superpowers/notes/2026-05-15-pcsa-tf-gate-result.md)
#
# Each cell is a separate python invocation so individual failures can be
# resumed; run_quarot_one_cell.py skips cells whose eval.json already exists.
#
# Usage:
#   bash scripts/run_quarot_pcsa_dbaf_sweep.sh
#
# Optional env overrides:
#   OUT_ROOT   — output root dir    (default: /data/outputs/HM-quarot)
#   MODEL      — model path/HF ID  (default: /data/modelzoo/meta-llama/Meta-Llama-3-8B)
#   NSAMPLES   — GPTQ calib samples (default: 128)
#   BSZ        — eval batch size    (default: 32)
set -uo pipefail

cd /home/ubuntu/unifying-ptq
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate unifyptq

OUT_ROOT="${OUT_ROOT:-/data/outputs/HM-quarot}"
MODEL="${MODEL:-/data/modelzoo/meta-llama/Meta-Llama-3-8B}"
NSAMPLES="${NSAMPLES:-128}"
BSZ="${BSZ:-32}"

TARGET="llama3-8b"
# pcsa_tf and dbaf+pcsa_tf are intentionally omitted — PCSA-tf gate FAIL.
AUGMENTS=(alone dbaf)

total="${#AUGMENTS[@]}"
cell_idx=0

for AUG in "${AUGMENTS[@]}"; do
    cell_idx=$((cell_idx + 1))
    OUT_DIR="${OUT_ROOT}/${TARGET}/quarot_${AUG}"
    OUT_PATH="${OUT_DIR}/eval.json"
    echo "===== [${cell_idx}/${total}] ${TARGET} method=quarot aug=${AUG} ====="
    mkdir -p "${OUT_DIR}"
    python scripts/run_quarot_one_cell.py \
        --augment "${AUG}" \
        --model   "${MODEL}" \
        --out     "${OUT_PATH}" \
        --nsamples "${NSAMPLES}" \
        --bsz     "${BSZ}" \
        2>&1 | tee "${OUT_DIR}/run.log"
    echo "===== [${cell_idx}/${total}] DONE: quarot_${AUG} ====="
done

echo "QUAROT_PHASE_B_SWEEP_DONE_$?"
