#!/usr/bin/env bash
# Structured cross-model training-free DBAF gating comparison on SwinIR-light.
# Three arms per (scale, dataset):
#   1. no-gate   (DBAF on every layer; the default in our prior runs)
#   2. gate-default     (frac3_max=0.02 — matches `flat_linear.py:61` default)
#   3. gate-permissive  (frac3_max=0.05 — accept dense-outlier layers)
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq

SET5=/home/ubuntu/unifying-ptq/data/sr_testsets/Set5_HR
URB=/home/ubuntu/unifying-ptq/data/sr_testsets/Urban100_HR
OUT=/home/ubuntu/unifying-ptq/results/S8-compsrt/swinir-light-cross-gate
mkdir -p "$OUT"

for SCALE in 2 3 4; do
  CKPT=/home/ubuntu/unifying-ptq/ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x${SCALE}.pth
  for DS_PAIR in Set5:$SET5 Urban100:$URB; do
    DS_NAME=${DS_PAIR%:*}; DPATH=${DS_PAIR#*:}
    # Arm 1: no gate
    python scripts/run_training_free_swinir.py \
      --scale $SCALE --pretrained "$CKPT" --dataset "$DPATH" \
      --bits 4 --use-dbaf --alpha 0.95 \
      --out "$OUT/x${SCALE}_${DS_NAME}_nogate.json"
    # Arm 2: gate-default
    python scripts/run_training_free_swinir.py \
      --scale $SCALE --pretrained "$CKPT" --dataset "$DPATH" \
      --bits 4 --use-dbaf --alpha 0.95 --gate-frac3-max 0.02 \
      --out "$OUT/x${SCALE}_${DS_NAME}_gate0.02.json"
    # Arm 3: gate-permissive
    python scripts/run_training_free_swinir.py \
      --scale $SCALE --pretrained "$CKPT" --dataset "$DPATH" \
      --bits 4 --use-dbaf --alpha 0.95 --gate-frac3-max 0.05 \
      --out "$OUT/x${SCALE}_${DS_NAME}_gate0.05.json"
  done
done
echo "S8_SWINIR_CROSS_GATE_DONE_$?"
