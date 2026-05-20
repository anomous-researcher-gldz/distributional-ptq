#!/usr/bin/env bash
# Run α-sensitivity on GPTQ/AWQ/SmoothQuant LLaMA-3-8B W4A4, α∈{0.5, 0.75}.
# (α=0.25 already done in HM-alpha025, α=0.95 not needed yet.)
# Waits for SAM-L queue to finish before starting.
set -uo pipefail

# Wait for SAM-L PCSA-only cell to finish (both cells finish marker).
SAML_LOG=/data/outputs/C3-saml-yolox-pcsa-only/logs/run.log
echo "[queue] waiting for SAM-L PCSA-only cell to finish ..."
while true; do
    if grep -q 'C3_SAML_PCSA_ONLY_DONE' /tmp/saml-queue.log 2>/dev/null; then
        echo "[queue] SAM-L PCSA-only marker found, starting α-sweep"
        break
    fi
    sleep 60
done

cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq

for METHOD in gptq awq smoothquant; do
  for ALPHA in 0.5 0.75; do
    OUT_ROOT=/data/outputs/HM-alpha-sweep-${METHOD//,/-}-a${ALPHA//./p}
    echo "[queue] === ${METHOD} α=${ALPHA} ==="
    python scripts/run_phaseA_alpha025.py \
        --target llama3-8b \
        --methods "${METHOD}" \
        --alpha "${ALPHA}" \
        --out_root "${OUT_ROOT}" 2>&1 | tail -20
    echo "[queue] === ${METHOD} α=${ALPHA} DONE ==="
  done
done

echo "ALPHA_SWEEP_3HOSTS_DONE"
