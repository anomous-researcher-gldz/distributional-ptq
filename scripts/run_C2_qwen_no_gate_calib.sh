#!/usr/bin/env bash
# C2: FlatQuant + DBAF (no-gate) + PCSA on Qwen-2.5-7B, W4A4 KV16.
# Matches C1 (LLaMA-3-8B) so the two LLM rows in Table 2 are comparable.
set -e
cd /home/ubuntu/unifying-ptq/FlatQuant
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq
export HF_HOME=/data/huggingface_cache
export PYTHONPATH=/home/ubuntu/unifying-ptq/FlatQuant:${PYTHONPATH:-}

OUT=/data/outputs/C2-qwen-no-gate
mkdir -p "$OUT/logs"

python main.py \
  --model /data/modelzoo/Qwen/Qwen2.5-7B \
  --w_bits 4 --a_bits 4 \
  --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
  --lwc --lac --cali_trans --add_diag \
  --no-dbaf-gate \
  --output_dir "$OUT" --save_matrix \
  --exp_name "fq-dbaf-no-gate-pcsa-qwen" \
  2>&1 | tee "$OUT/logs/run.log"
echo "C2_QWEN_NO_GATE_CALIB_DONE_$?"
