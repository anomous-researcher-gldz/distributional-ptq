#!/usr/bin/env bash
# Training-free SAM-B: RTN baseline + RTN+DBAF on COCO val2017
# Uses vendored segment_anything (mmdet-free) + torchvision Faster-RCNN detector
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq
export PYTHONPATH=/home/ubuntu/unifying-ptq:/home/ubuntu/unifying-ptq/FlatQuant:${PYTHONPATH:-}

MAX=${1:-500}  # 500 images for speed; 5000 = full val2017
for flag in "" "--use-dbaf"; do
  suffix=baseline; [ -n "$flag" ] && suffix=with-dbaf
  out=results/S4-dbaf-weak/sam-b-rtn/${suffix}/eval.json
  log=results/S4-dbaf-weak/sam-b-rtn/${suffix}/run.log
  mkdir -p "$(dirname "$out")"
  echo "===== $suffix (n=$MAX) ===== $(date)"
  python scripts/run_training_free_sam.py --bits 4 --max-images $MAX --out "$out" $flag 2>&1 | tee "$log"
done
echo "S4_9_SAM_TRAINING_FREE_DONE_$?"
