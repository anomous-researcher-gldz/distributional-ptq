#!/usr/bin/env bash
# C1: FlatQuant + DBAF (no-gate) + PCSA on LLaMA-3-8B, W4A4 KV16 to match the
# published 6.96 setting. Compares end-to-end with the gated training number.
set -e
cd /home/ubuntu/unifying-ptq/FlatQuant
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq
export HF_HOME=/data/huggingface_cache
export PYTHONPATH=/home/ubuntu/unifying-ptq/FlatQuant:${PYTHONPATH:-}

OUT=/data/outputs/C1-llama-no-gate
mkdir -p "$OUT/logs"

python main.py \
  --model /data/modelzoo/meta-llama/Meta-Llama-3-8B \
  --w_bits 4 --a_bits 4 \
  --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
  --lwc --lac --cali_trans --add_diag \
  --no-dbaf-gate \
  --output_dir "$OUT" --save_matrix \
  --exp_name "fq-dbaf-no-gate-pcsa" \
  2>&1 | tee "$OUT/logs/run.log"
echo "C1_LLAMA_NO_GATE_CALIB_DONE_$?"
