#!/usr/bin/env bash
# F2 extension: W4A4 dual-gate sweep on SwinIR x2 + x4.
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq

SET5=/home/ubuntu/unifying-ptq/data/sr_testsets/Set5_HR
URB=/home/ubuntu/unifying-ptq/data/sr_testsets/Urban100_HR
OUT=/home/ubuntu/unifying-ptq/results/F2-swinir-dual-gate
mkdir -p "$OUT"

for SCALE in 2 4; do
  CKPT=/home/ubuntu/unifying-ptq/ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x${SCALE}.pth
  for DS_PAIR in Set5:$SET5 Urban100:$URB; do
    DS_NAME=${DS_PAIR%:*}; DPATH=${DS_PAIR#*:}
    # Arm A: W4A4 no-DBAF
    python scripts/run_training_free_swinir.py \
      --scale $SCALE --pretrained "$CKPT" --dataset "$DPATH" \
      --bits 4 --act-bits 4 \
      --out "$OUT/x${SCALE}_A_w4a4_nodbaf_${DS_NAME}.json"
    # Arm B: DBAF Wforce + Aforce
    python scripts/run_training_free_swinir.py \
      --scale $SCALE --pretrained "$CKPT" --dataset "$DPATH" \
      --bits 4 --act-bits 4 --use-dbaf --act-gate-frac3-max -1 \
      --out "$OUT/x${SCALE}_B_w4a4_dbaf_Wforce_Aforce_${DS_NAME}.json"
    # Arm E: DBAF Wgate(0.02) + Agate(0.02) — codebase default
    python scripts/run_training_free_swinir.py \
      --scale $SCALE --pretrained "$CKPT" --dataset "$DPATH" \
      --bits 4 --act-bits 4 --use-dbaf --gate-frac3-max 0.02 --act-gate-frac3-max 0.02 \
      --out "$OUT/x${SCALE}_E_w4a4_dbaf_Wgate_Agate_${DS_NAME}.json"
  done
done
echo "F2_SWINIR_X2X4_DUAL_GATE_DONE_$?"
