#!/usr/bin/env bash
# Wait for C2 (FlatQuant+DBAF+PCSA Qwen) to finish, then run SAM-L singletons:
#   1) AHCPTQ + DBAF only (no PCSA): default config, no --save-pcsa
#   2) AHCPTQ + PCSA only (no DBAF): --save-pcsa --no-dbaf
# Both fill tab:ahcptq-sam SAM-L singleton cells.
set -uo pipefail

C2_LOG=/data/outputs/C2-qwen-no-gate/logs/run.log

echo "[queue] waiting for C2 to finish ..."
while true; do
    if grep -q 'C2_QWEN_NO_GATE_CALIB_DONE_' "$C2_LOG" 2>/dev/null; then
        echo "[queue] C2 finish marker found in run.log"
        break
    fi
    sleep 60
done

cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ahcptq-old
export PYTHONPATH=/home/ubuntu/unifying-ptq:${PYTHONPATH:-}

# ---------- Cell 1: AHCPTQ + DBAF only on SAM-L ----------
OUT1=/data/outputs/C3-saml-yolox-dbaf-only
mkdir -p "$OUT1/logs"
echo "[queue] === Cell 1: AHCPTQ+DBAF only on SAM-L ==="
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/yolox/yolo_l-sam-vit-l.py \
  --q_config ./exp/config44.yaml \
  --quant-encoder \
  --work-dir "$OUT1" \
  --save_sam_path "$OUT1/sam.pt" \
  --eval segm \
  --no-dbaf-gate 2>&1 | tee "$OUT1/logs/run.log"
echo "C3_SAML_DBAF_ONLY_DONE_$?"

# ---------- Cell 2: AHCPTQ + PCSA only on SAM-L ----------
OUT2=/data/outputs/C3-saml-yolox-pcsa-only
mkdir -p "$OUT2/logs"
echo "[queue] === Cell 2: AHCPTQ+PCSA only on SAM-L ==="
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/yolox/yolo_l-sam-vit-l.py \
  --q_config ./exp/config44.yaml \
  --quant-encoder \
  --work-dir "$OUT2" \
  --save-pcsa "$OUT2/pcsa.pt" \
  --save_sam_path "$OUT2/sam.pt" \
  --eval segm \
  --no-dbaf 2>&1 | tee "$OUT2/logs/run.log"
echo "C3_SAML_PCSA_ONLY_DONE_$?"

echo "[queue] both SAM-L singletons done"
