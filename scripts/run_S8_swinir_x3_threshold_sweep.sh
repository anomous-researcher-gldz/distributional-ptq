#!/usr/bin/env bash
# Sweep T_sigma and gate_frac3_max for SwinIR-light x3 at alpha=0.95.
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq

CKPT=/home/ubuntu/unifying-ptq/ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth
SET5=/home/ubuntu/unifying-ptq/data/sr_testsets/Set5_HR
URB=/home/ubuntu/unifying-ptq/data/sr_testsets/Urban100_HR
OUT=/home/ubuntu/unifying-ptq/results/S8-compsrt/swinir-light-x3-threshold-sweep
mkdir -p "$OUT"

# (A) T_sigma sweep — vary the fold threshold (lower = more aggressive fold).
for TS in 2.0 2.5 3.0 3.5 4.0 5.0; do
  for DS_PAIR in Set5:$SET5 Urban100:$URB; do
    DS_NAME=${DS_PAIR%:*}; DPATH=${DS_PAIR#*:}
    TAG="tsigma${TS}_alpha0.95_${DS_NAME}"
    echo "=== $TAG ==="
    python scripts/run_training_free_swinir.py \
      --scale 3 --pretrained "$CKPT" --dataset "$DPATH" \
      --bits 4 --use-dbaf --alpha 0.95 --T-sigma "$TS" \
      --out "$OUT/${TAG}.json"
  done
done

# (B) Gate sweep — only fire DBAF on layers passing gate (frac3 below threshold).
for FRAC in 0.01 0.015 0.02 0.03 0.05 0.10; do
  for DS_PAIR in Set5:$SET5 Urban100:$URB; do
    DS_NAME=${DS_PAIR%:*}; DPATH=${DS_PAIR#*:}
    TAG="gate${FRAC}_alpha0.95_${DS_NAME}"
    echo "=== $TAG ==="
    python scripts/run_training_free_swinir.py \
      --scale 3 --pretrained "$CKPT" --dataset "$DPATH" \
      --bits 4 --use-dbaf --alpha 0.95 --gate-frac3-max "$FRAC" \
      --out "$OUT/${TAG}.json"
  done
done

echo "S8_X3_THRESHOLD_SWEEP_DONE_$?"
