#!/usr/bin/env bash
# Search α for SwinIR-light x3 (currently the only scale that regresses with DBAF).
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq

CKPT=/home/ubuntu/unifying-ptq/ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth
SET5=/home/ubuntu/unifying-ptq/data/sr_testsets/Set5_HR
URB=/home/ubuntu/unifying-ptq/data/sr_testsets/Urban100_HR
OUT=/home/ubuntu/unifying-ptq/results/S8-compsrt/swinir-light-x3-alpha-sweep
mkdir -p "$OUT"

for ALPHA in 0.70 0.80 0.85 0.90 0.95 0.97 0.99 1.00; do
  for DS in Set5 Urban100; do
    if [ "$DS" = "Set5" ]; then DPATH=$SET5; else DPATH=$URB; fi
    TAG="alpha${ALPHA}_${DS}"
    echo "=== $TAG ==="
    python scripts/run_training_free_swinir.py \
      --scale 3 --pretrained "$CKPT" --dataset "$DPATH" \
      --bits 4 --use-dbaf --alpha "$ALPHA" \
      --out "$OUT/${TAG}.json"
  done
done
echo "S8_X3_ALPHA_SWEEP_DONE_$?"
