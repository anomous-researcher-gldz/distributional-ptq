#!/usr/bin/env bash
# Task 11: OmniQuant ±{DBAF, PCSA-tf, both} on LLaMA-3-8B W4A4
# Arms: A (vanilla), B (+DBAF), C (+PCSA-tf), D (+DBAF+PCSA-tf)
#
# CLI flag notes (discovered from main.py + scripts/Llama-2/Llama-2-7b/w4a4.sh):
#   --model         : path to HuggingFace model dir
#   --wbits         : weight bits (int, default 4)
#   --abits         : activation bits (int, default 16)
#   --epochs        : calibration epochs (default 10); requires --lwc or --let
#   --nsamples      : number of calibration samples (default 128)
#   --calib_dataset : one of wikitext2/ptb/c4/mix/pile (default wikitext2)
#   --output_dir    : logging output directory
#   --save_dir      : (optional) save fake-quantized model weights
#   --lwc           : enable Learnable Weight Clipping (action flag)
#   --let           : enable Learnable Equivalent Transformation — requires act_scales/act_shifts
#   --eval_ppl      : (action flag) run wikitext2+c4 PPL eval after calibration
#   --tasks         : lm_eval tasks string (default ""; omit to skip harness eval)
#   --net           : auto-derived from model path last component if not set
#   --alpha         : SmoothQuant-style alpha (default 0.5; w4a4 paper uses 0.75)
#   --let_lr / --lwc_lr: optimizer LRs (defaults 5e-3 / 1e-2)
#
# NOTE: --let requires pre-computed act_scales/<net>.pt and act_shifts/<net>.pt.
# For LLaMA-3-8B these don't ship with OmniQuant; use --lwc only for W4A4 until
# we generate them via generate_act_scale_shift.py.
# Arm D adds both DBAF + PCSA-tf on top of --lwc.
set -e
cd /home/ubuntu/unifying-ptq/OmniQuant
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
export PYTHONPATH=/home/ubuntu/unifying-ptq:/home/ubuntu/unifying-ptq/FlatQuant:.:${PYTHONPATH:-}
export HF_HOME=/data/huggingface_cache

OUT=/data/outputs/G6-omniquant-llama3-8b
mkdir -p "$OUT"
MODEL=/data/modelzoo/meta-llama/Meta-Llama-3-8B

# Real OmniQuant CLI flags; --lwc required (--let skipped: no act_scales for Llama-3).
# --alpha 0.75 follows the w4a4 reference script for Llama-2-7b.
COMMON_ARGS="--model $MODEL \
  --wbits 4 --abits 4 \
  --calib_dataset wikitext2 \
  --nsamples 128 --epochs 20 \
  --lwc \
  --alpha 0.75 \
  --lwc_lr 1e-2"

ARM=${1:-all}

run_arm_A() {
  echo "[arm_A] vanilla OmniQuant W4A4 on LLaMA-3-8B" | tee "$OUT/arm_A_vanilla.log"
  python main.py $COMMON_ARGS \
    --output_dir "$OUT/arm_A_vanilla" \
    2>&1 | tee -a "$OUT/arm_A_vanilla.log"
}

run_arm_B() {
  echo "[arm_B] +DBAF on LLaMA-3-8B" | tee "$OUT/arm_B_dbaf.log"
  # Install DBAF patch before importing main via runpy so the monkey-patch fires
  # before any UniformAffineQuantizer is instantiated.
  python -u -c "
import sys
sys.path.insert(0, '/home/ubuntu/unifying-ptq/OmniQuant')
sys.path.insert(0, '/home/ubuntu/unifying-ptq')
import omniquant_dbaf_pcsa_patch as p
p.install_dbaf_patches(dbaf_alpha=0.95)
import runpy
runpy.run_path('main.py', run_name='__main__')
" $COMMON_ARGS \
    --output_dir "$OUT/arm_B_dbaf" \
    2>&1 | tee -a "$OUT/arm_B_dbaf.log"
}

run_arm_C() {
  echo "[arm_C] +PCSA-tf on LLaMA-3-8B" | tee "$OUT/arm_C_pcsa_tf.log"
  # PCSA-tf: we feed synthetic descriptors / acts at driver level for the routing
  # fit. The arms are a pilot; full integration (real calibration descriptors from
  # the omniquant.py per-layer loop) is a follow-up task.
  python -u -c "
import sys
sys.path.insert(0, '/home/ubuntu/unifying-ptq/OmniQuant')
sys.path.insert(0, '/home/ubuntu/unifying-ptq')
import omniquant_dbaf_pcsa_patch as p
import torch
torch.manual_seed(0)
# Synthetic calib descriptors: nsamples=128, hidden_dim=4096 (Llama-3-8B)
descs = torch.randn(128, 4096)
acts  = torch.randn(128, 128, 4096)  # [nsamples, short_seq, hidden]
p.fit_pcsa_tf_on_calib_data(descs, acts, K=8)
p.install_pcsa_tf()
import runpy
runpy.run_path('main.py', run_name='__main__')
" $COMMON_ARGS \
    --output_dir "$OUT/arm_C_pcsa_tf" \
    2>&1 | tee -a "$OUT/arm_C_pcsa_tf.log"
}

run_arm_D() {
  echo "[arm_D] +DBAF+PCSA-tf on LLaMA-3-8B" | tee "$OUT/arm_D_both.log"
  python -u -c "
import sys
sys.path.insert(0, '/home/ubuntu/unifying-ptq/OmniQuant')
sys.path.insert(0, '/home/ubuntu/unifying-ptq')
import omniquant_dbaf_pcsa_patch as p
import torch
p.install_dbaf_patches(dbaf_alpha=0.95)
torch.manual_seed(0)
descs = torch.randn(128, 4096)
acts  = torch.randn(128, 128, 4096)
p.fit_pcsa_tf_on_calib_data(descs, acts, K=8)
p.install_pcsa_tf()
import runpy
runpy.run_path('main.py', run_name='__main__')
" $COMMON_ARGS \
    --output_dir "$OUT/arm_D_both" \
    2>&1 | tee -a "$OUT/arm_D_both.log"
}

case "$ARM" in
  A|all) run_arm_A ;;
esac
case "$ARM" in
  B|all) run_arm_B ;;
esac
case "$ARM" in
  C|all) run_arm_C ;;
esac
case "$ARM" in
  D|all) run_arm_D ;;
esac
echo "OMNIQUANT_LLAMA3_${ARM}_DONE_$?"
