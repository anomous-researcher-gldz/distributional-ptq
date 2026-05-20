#!/usr/bin/env bash
# D2 long-context KV-PCSA eval — baseline vs KV-PCSA-v2 across 2k/4k/8k context.
# Reuses scripts/eval_kvpcsa_long_context.py (now fixed for eager attention + canonical FQ load).
set -uo pipefail
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
mkdir -p /data/outputs/S5-long-context-eval

echo "===== KV-PCSA-v2 @ 2k/4k/8k ====="
python scripts/eval_kvpcsa_long_context.py \
  --matrix-path /data/outputs/S5-kv-pcsa-v2-calib/Meta-Llama-3-8B/w4a4/fq-dbaf-pcsa-kvpcsa-v2 \
  --kv-pcsa \
  --label kv_pcsa_v2 \
  --seq-lens 2048 4096 8192 \
  --out /data/outputs/S5-long-context-eval/kv_pcsa_v2_v3.json 2>&1 | tail -25

echo "D2_KV_PCSA_DONE_$?"
