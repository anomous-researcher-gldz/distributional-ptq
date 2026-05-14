#!/usr/bin/env bash
# Re-run OmniQuant Arms A and B after main.py was patched for PyTorch 2.6.
#
# Background:
#   Arms A and B were started BEFORE the torch.load weights_only=False patch was
#   applied to OmniQuant/main.py.  Because each arm is a fresh `python main.py`
#   invocation, the in-flight processes (already in memory) do not see the patch.
#   Arms C and D (started after the patch) are fine.
#
# Strategy:
#   1. Wait for the current g6-omniquant tmux session to finish (arms C and D).
#   2. Re-run arms A and B from scratch using the now-patched main.py.
#      Calibration will be fast (cached dataloader on disk) and eval will succeed.
#
# Usage:
#   # Option 1 — wait automatically then rerun both:
#   bash scripts/run_omniquant_rerun_AB.sh
#
#   # Option 2 — rerun a single arm (useful if only one crashed):
#   bash scripts/run_omniquant_rerun_AB.sh A
#   bash scripts/run_omniquant_rerun_AB.sh B
#
# Output goes to the same directories as the original run so results are
# consistent with arms C and D:
#   /data/outputs/G6-omniquant-llama3-8b/arm_A_vanilla/
#   /data/outputs/G6-omniquant-llama3-8b/arm_B_dbaf/

set -uo pipefail
# NOTE: deliberately no `-e` — we want Arm B to still run even if Arm A fails.

ARM="${1:-AB}"   # A | B | AB

# ---------------------------------------------------------------------------
# Step 1: Wait for current g6-omniquant session to finish (arms C and D).
# ---------------------------------------------------------------------------
wait_for_session_to_finish() {
  local session="g6-omniquant"
  echo "[rerun_AB] Checking if tmux session '$session' is still running..."
  while tmux has-session -t "$session" 2>/dev/null; do
    echo "[rerun_AB] Session '$session' still active — sleeping 5 min. ($(date))"
    sleep 300
  done
  echo "[rerun_AB] Session '$session' finished (or never existed). Proceeding."
}

if [[ "$ARM" == "AB" ]]; then
  wait_for_session_to_finish
fi

# ---------------------------------------------------------------------------
# Step 2: Environment setup (mirrors run_omniquant_llama3_8b.sh exactly)
# ---------------------------------------------------------------------------
cd /home/ubuntu/unifying-ptq/OmniQuant
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate unifyptq
export PYTHONPATH=/home/ubuntu/unifying-ptq:/home/ubuntu/unifying-ptq/FlatQuant:.:${PYTHONPATH:-}
export HF_HOME=/data/huggingface_cache

OUT=/data/outputs/G6-omniquant-llama3-8b
mkdir -p "$OUT"
MODEL=/data/modelzoo/meta-llama/Meta-Llama-3-8B

COMMON_ARGS="--model $MODEL \
  --wbits 4 --abits 4 \
  --calib_dataset wikitext2 \
  --nsamples 128 --epochs 20 \
  --lwc \
  --alpha 0.75 \
  --lwc_lr 1e-2 \
  --eval_ppl"

# ---------------------------------------------------------------------------
# Step 3: Re-run the selected arm(s)
# ---------------------------------------------------------------------------

run_arm_A() {
  echo "[rerun_AB] === Arm A (vanilla) re-run at $(date) ===" | tee -a "$OUT/arm_A_vanilla.log"
  python main.py $COMMON_ARGS \
    --output_dir "$OUT/arm_A_vanilla" \
    --save_dir "$OUT/arm_A_vanilla/model" \
    2>&1 | tee -a "$OUT/arm_A_vanilla.log"
  echo "[rerun_AB] Arm A finished with exit code $?"
}

run_arm_B() {
  echo "[rerun_AB] === Arm B (+DBAF) re-run at $(date) ===" | tee -a "$OUT/arm_B_dbaf.log"
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
    --save_dir "$OUT/arm_B_dbaf/model" \
    2>&1 | tee -a "$OUT/arm_B_dbaf.log"
  echo "[rerun_AB] Arm B finished with exit code $?"
}

case "$ARM" in
  A)  run_arm_A ;;
  B)  run_arm_B ;;
  AB) run_arm_A; run_arm_B ;;
  *)  echo "Usage: $0 [A|B|AB]" >&2; exit 1 ;;
esac

echo "[rerun_AB] Done — check $OUT/arm_{A,B}_*/eval.log for PPL results."
