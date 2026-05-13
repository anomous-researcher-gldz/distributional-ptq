#!/usr/bin/env bash
# Training-free SwinIR-light RTN+DBAF at scales x2, x3, x4 on Set5/Urban100
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq
export PYTHONPATH=/home/ubuntu/unifying-ptq/FlatQuant:${PYTHONPATH:-}

for scale in 2 3 4; do
  ckpt=ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x${scale}.pth
  for dataset_name in Set5 Urban100; do
    HR=data/sr_testsets/${dataset_name}_HR
    LR=data/sr_testsets/${dataset_name}_LR_x${scale}
    [ -d "$LR" ] || LR=
    for flag in "" "--use-dbaf"; do
      suffix=baseline; [ -n "$flag" ] && suffix=with-dbaf
      out=results/S8-compsrt/swinir-light-x${scale}/${dataset_name}/${suffix}/eval.json
      mkdir -p "$(dirname "$out")"
      echo "===== swinir-S x${scale} ${dataset_name} ${suffix} =====  $(date)"
      python scripts/run_training_free_swinir.py \
        --scale $scale --pretrained "$ckpt" \
        --dataset "$HR" \
        ${LR:+--lr-subdir "$LR"} \
        --bits 4 --alpha 0.95 $flag --out "$out" 2>&1 | tail -10
    done
  done
done
echo "S8_SWINIR_LIGHT_DONE_$?"
