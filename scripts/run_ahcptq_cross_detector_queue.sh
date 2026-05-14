#!/usr/bin/env bash
# Task 15: queue SAM-H+H-DETR -> SAM-B+H-DETR -> SAM-L+H-DETR after SAM-H+YOLOX wraps.
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ahcptq-old
export PYTHONPATH=/home/ubuntu/unifying-ptq:${PYTHONPATH:-}

OUT_ROOT=results/G9-ahcptq-cross-detector
mkdir -p $OUT_ROOT

# Wait for prior AHCPTQ training to finish
echo "[queue] waiting for s2-samh-yolox session to finish..."
while tmux has-session -t s2-samh-yolox 2>/dev/null && \
      ! tmux capture-pane -t s2-samh-yolox -p 2>/dev/null | grep -q 'SAMH_YOLOX_DONE'; do
  sleep 90
done
echo "[queue] prior SAM-H+YOLOX done"

run_cell() {
  local MODEL=$1 DETECTOR=$2 CONFIG=$3 QCONFIG=$4
  local CELL_OUT=$OUT_ROOT/$MODEL/$DETECTOR-w4a4
  mkdir -p $CELL_OUT/logs
  echo "===== $MODEL + $DETECTOR ====="
  python ahcptq/solver/test_quant.py \
    --config $CONFIG \
    --q_config $QCONFIG \
    --quant-encoder --eval segm \
    --work-dir $CELL_OUT \
    --save-pcsa $CELL_OUT/pcsa.pt \
    --save_sam_path $CELL_OUT/sam.pt \
    2>&1 | tee $CELL_OUT/logs/run.log
}

run_cell sam-h h-detr ./projects/configs/hdetr/r50-hdetr_sam-vit-h.py ./exp/config44_samh.yaml
run_cell sam-b h-detr ./projects/configs/hdetr/r50-hdetr_sam-vit-b.py ./exp/config44_hdetr.yaml
run_cell sam-l h-detr ./projects/configs/hdetr/r50-hdetr_sam-vit-l.py ./exp/config44_hdetr.yaml

echo "G9_CROSS_DETECTOR_DONE_$?"
