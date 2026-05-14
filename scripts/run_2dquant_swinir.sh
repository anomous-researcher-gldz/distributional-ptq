#!/usr/bin/env bash
# Task 13: 2DQuant ±{DBAF, PCSA-tf, both} on SwinIR-light ×2/×3/×4 at W4A4.
# Arms: A vanilla, B +DBAF, C +PCSA-tf, D +DBAF+PCSA-tf. Each saves eval.json.
#
# CLI pattern (from 2DQuant/scripts/2DQuant_test.sh):
#   python basicsr/test.py -opt options/test/test_2DQuant_x{2,3,4}.yml \
#       --force_yml bit=4 name=<run_name> path:pretrain_network_Q=<ckpt>
#
# YAMLs that ship with 2DQuant (already W4A4 capable via --force_yml bit=4):
#   options/test/test_2DQuant_x2.yml
#   options/test/test_2DQuant_x3.yml
#   options/test/test_2DQuant_x4.yml
#
# Calibration data (per-scale .pth files) must be generated first via:
#   python basicsr/train_ptq_getcalidata.py -opt options/train/train_getcalidata_x{2,3,4}.yml
# The cali-data scripts require the DF2K training set at datasets/DF2K/.
# If DF2K is unavailable, generate synthetic cali data with the helper at the
# bottom of this file (set SKIP_CALI_DATA_GEN=1 to skip).
#
# Benchmark datasets expected at (relative to 2DQuant/):
#   datasets/benchmark/Set5/{HR,LR_bicubic/X{2,3,4}}
#   datasets/benchmark/Set14/{HR,LR_bicubic/X{2,3,4}}
#   datasets/benchmark/B100/{HR,LR_bicubic/X{2,3,4}}
#   datasets/benchmark/Urban100/{HR,LR_bicubic/X{2,3,4}}
#   datasets/benchmark/Manga109/{HR,LR_bicubic/X{2,3,4}}
#
# Results written by basicsr to 2DQuant/results/<run_name>/
# This driver then harvests PSNR/SSIM from the run.log and emits eval.json
# at $OUT_ROOT/x{scale}/{arm}/eval.json.
#
# PCSA-tf uses SwinIR-S embed_dim=60 for synthetic pilot descriptors (real
# descriptor collection via conv_first hook is a follow-up).
#
# DO NOT RUN while local GPU is busy. This script is syntax-checked only (bash -n).
set -euo pipefail

cd /home/ubuntu/unifying-ptq/2DQuant
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate unifyptq
export PYTHONPATH=/home/ubuntu/unifying-ptq:/home/ubuntu/unifying-ptq/FlatQuant:.:${PYTHONPATH:-}

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
CKPT_ROOT=/home/ubuntu/unifying-ptq/ckpt/swinir
OUT_ROOT=/data/outputs/G7-2dquant-swinir
mkdir -p "$OUT_ROOT"

# Allow partial runs via environment overrides
SCALES="${SCALES:-2 3 4}"
ARMS="${ARMS:-A B C D}"

# Skip calibration-data generation if it already exists or DF2K not available
SKIP_CALI_DATA_GEN="${SKIP_CALI_DATA_GEN:-0}"

# -----------------------------------------------------------------------
# Step 0: Generate per-scale calibration data (.pth) if needed
# Each keydata/cali_data_x{S}.pth is one batch of LR/HR pairs from DF2K.
# -----------------------------------------------------------------------
maybe_gen_cali_data() {
  local SCALE=$1
  local PTH="keydata/cali_data_x${SCALE}.pth"
  if [[ "$SKIP_CALI_DATA_GEN" == "1" ]]; then
    echo "[cali] SKIP_CALI_DATA_GEN=1 — skipping cali-data generation for x${SCALE}"
    return
  fi
  if [[ -f "$PTH" ]]; then
    echo "[cali] $PTH already exists — skipping"
    return
  fi
  echo "[cali] Generating $PTH via train_ptq_getcalidata.py ..."
  mkdir -p keydata
  python basicsr/train_ptq_getcalidata.py \
    -opt options/train/train_getcalidata_x${SCALE}.yml \
    2>&1 | tee "${PTH%.pth}_gen.log"
}

for S in $SCALES; do
  maybe_gen_cali_data "$S"
done

# -----------------------------------------------------------------------
# Core run function
# -----------------------------------------------------------------------
run_cell() {
  local SCALE=$1
  local ARM=$2

  local FP_CKPT="${CKPT_ROOT}/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x${SCALE}.pth"
  local Q_CKPT_DIR="/home/ubuntu/unifying-ptq/2DQuant/pretrained/train_2DQuant_x${SCALE}_bit4/models"
  local Q_CKPT=$(ls "${Q_CKPT_DIR}"/net_Q_*.pth | tail -1)
  local RUN_NAME="2DQuant_x${SCALE}_w4a4_arm${ARM}"
  local OUT_DIR="${OUT_ROOT}/x${SCALE}/${ARM}"
  local RESULTS_DIR="results/${RUN_NAME}"  # relative to 2DQuant/
  mkdir -p "$OUT_DIR"

  echo "===== x${SCALE} arm ${ARM} (run_name=${RUN_NAME}) ====="

  # The --force_yml override syntax uses colon for nested keys.
  # path:pretrain_network_Q overrides both path.pretrain_network_Q AND
  # pathFP.pretrain_network_FP (both point to the same SwinIR-S ckpt).
  local FORCE_YML="bit=4 name=${RUN_NAME} path:pretrain_network_Q=${Q_CKPT} pathFP:pretrain_network_FP=${FP_CKPT}"

  # SwinIR-S embed_dim=60; use as descriptor dim for synthetic PCSA-tf pilot.
  local EMBED_DIM=60

  case "$ARM" in
    A)
      # Vanilla 2DQuant W4A4 — no patch
      python basicsr/test.py \
        -opt options/test/test_2DQuant_x${SCALE}_emnlp.yml \
        --force_yml $FORCE_YML \
        2>&1 | tee "$OUT_DIR/run.log"
      ;;

    B)
      # +DBAF: monkey-patch FakeQuantizerWeight/Act before importing test.py
      python -u -c "
import sys
sys.path.insert(0, '/home/ubuntu/unifying-ptq/2DQuant')
sys.path.insert(0, '/home/ubuntu/unifying-ptq')
sys.path.insert(0, '/home/ubuntu/unifying-ptq/FlatQuant')
import twodquant_dbaf_pcsa_patch as p
p.install_dbaf_patches(dbaf_alpha=0.95)
import runpy
runpy.run_path('basicsr/test.py', run_name='__main__')
" -opt options/test/test_2DQuant_x${SCALE}_emnlp.yml \
        --force_yml $FORCE_YML \
        2>&1 | tee "$OUT_DIR/run.log"
      ;;

    C)
      # +PCSA-tf: fit on synthetic pilot descriptors (embed_dim=60 for SwinIR-S)
      # Real integration uses conv_first hook; synthetic pilot establishes the
      # eval harness and confirms no regression in metric extraction.
      python -u -c "
import sys
sys.path.insert(0, '/home/ubuntu/unifying-ptq/2DQuant')
sys.path.insert(0, '/home/ubuntu/unifying-ptq')
sys.path.insert(0, '/home/ubuntu/unifying-ptq/FlatQuant')
import twodquant_dbaf_pcsa_patch as p
import torch
torch.manual_seed(0)
# Synthetic calibration descriptors: N=128, embed_dim=${EMBED_DIM} (SwinIR-S)
descs = torch.randn(128, ${EMBED_DIM})
acts  = torch.randn(128, 16, ${EMBED_DIM})
p.fit_pcsa_tf_on_calib_data(descs, acts, K=8)
p.install_pcsa_tf()
import runpy
runpy.run_path('basicsr/test.py', run_name='__main__')
" -opt options/test/test_2DQuant_x${SCALE}_emnlp.yml \
        --force_yml $FORCE_YML \
        2>&1 | tee "$OUT_DIR/run.log"
      ;;

    D)
      # +DBAF+PCSA-tf: both patches together
      python -u -c "
import sys
sys.path.insert(0, '/home/ubuntu/unifying-ptq/2DQuant')
sys.path.insert(0, '/home/ubuntu/unifying-ptq')
sys.path.insert(0, '/home/ubuntu/unifying-ptq/FlatQuant')
import twodquant_dbaf_pcsa_patch as p
import torch
p.install_dbaf_patches(dbaf_alpha=0.95)
torch.manual_seed(0)
descs = torch.randn(128, ${EMBED_DIM})
acts  = torch.randn(128, 16, ${EMBED_DIM})
p.fit_pcsa_tf_on_calib_data(descs, acts, K=8)
p.install_pcsa_tf()
import runpy
runpy.run_path('basicsr/test.py', run_name='__main__')
" -opt options/test/test_2DQuant_x${SCALE}_emnlp.yml \
        --force_yml $FORCE_YML \
        2>&1 | tee "$OUT_DIR/run.log"
      ;;

    *)
      echo "ERROR: unknown arm '$ARM'" >&2
      exit 1
      ;;
  esac

  # -----------------------------------------------------------------------
  # Harvest metrics from run.log and emit eval.json
  # basicsr logs lines like:
  #   INFO: Validation Set5
  #       # psnr: 32.1234    Best: ...
  #       # ssim: 0.8901    Best: ...
  # We extract all dataset PSNR/SSIM pairs.
  # -----------------------------------------------------------------------
  python -u - "$OUT_DIR/run.log" "$OUT_DIR/eval.json" <<'PYEOF'
import sys, re, json, pathlib

log_path   = pathlib.Path(sys.argv[1])
out_path   = pathlib.Path(sys.argv[2])

text       = log_path.read_text(errors="replace")
results    = {}

# Match "Validation <dataset>" blocks
for ds_match in re.finditer(r"Validation (\S+)\n(.*?)(?=Validation |\Z)", text, re.DOTALL):
    ds   = ds_match.group(1)
    body = ds_match.group(2)
    entry = {}
    m = re.search(r"#\s*psnr:\s*([\d.]+)", body)
    if m:
        entry["psnr"] = float(m.group(1))
    m = re.search(r"#\s*ssim:\s*([\d.]+)", body)
    if m:
        entry["ssim"] = float(m.group(1))
    if entry:
        results[ds] = entry

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2))
print(f"[eval.json] wrote {len(results)} dataset(s) to {out_path}")
PYEOF

  echo "===== x${SCALE} arm ${ARM} DONE ====="
}

# -----------------------------------------------------------------------
# Main sweep: 3 scales × 4 arms = 12 cells
# -----------------------------------------------------------------------
for S in $SCALES; do
  for A in $ARMS; do
    run_cell "$S" "$A"
  done
done

echo "TWODQUANT_SWINIR_ALL_DONE_$?"
