#!/usr/bin/env bash
# C3a: AHCPTQ + DBAF (no-gate) + PCSA on SAM-B + YOLOX W4A4.
# Mirrors run_s2_samh_yolox.sh recipe but on SAM-B + with --no-dbaf-gate.
# Compares end-to-end vs the gated prior 18.2 mAP result on SAM-B.
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ahcptq-old
export PYTHONPATH=/home/ubuntu/unifying-ptq:${PYTHONPATH:-}
OUT=results/C3a-samb-yolox-no-gate
mkdir -p $OUT/logs
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/yolox/yolo_l-sam-vit-b.py \
  --q_config ./exp/config44.yaml \
  --quant-encoder \
  --work-dir $OUT \
  --save-pcsa $OUT/pcsa.pt \
  --save_sam_path $OUT/sam.pt \
  --eval segm \
  --no-dbaf-gate 2>&1 | tee $OUT/logs/run.log
echo "C3A_SAMB_NO_GATE_DONE_$?"
