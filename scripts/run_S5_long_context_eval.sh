#!/usr/bin/env bash
# Eval calibrated FlatQuant checkpoints on WikiText-2 at multiple seq_lens.
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq
export HF_HOME=/data/huggingface_cache
export PYTHONPATH=/home/ubuntu/unifying-ptq/FlatQuant:${PYTHONPATH:-}

OUT=/data/outputs/S5-long-context-eval
mkdir -p "$OUT"

# Arm 1: baseline (no KV-PCSA)
python scripts/eval_kvpcsa_long_context.py \
  --matrix-path /data/outputs/S5-baseline-calib/Meta-Llama-3-8B/w4a4/fq-dbaf-pcsa-baseline \
  --label baseline_kv4_asym_g128 \
  --out "$OUT/baseline.json" 2>&1 | tee "$OUT/baseline.log"

# Arm 2: KV-PCSA (the buggy per-anchor-scalar version, for comparison)
python scripts/eval_kvpcsa_long_context.py \
  --matrix-path /data/outputs/S5-kv-pcsa-calib/Meta-Llama-3-8B/w4a4/fq-dbaf-pcsa-kvpcsa \
  --kv-pcsa \
  --label kv_pcsa_v1_per_anchor_scalar \
  --out "$OUT/kv_pcsa_v1.json" 2>&1 | tee "$OUT/kv_pcsa_v1.log"

echo "LONG_CONTEXT_EVAL_DONE_$?"
