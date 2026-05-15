#!/usr/bin/env bash
# Create the `tesseraq` conda env (separate from `unifyptq` because TesseraQ
# pins different transformers/torch versions). Idempotent.
set -uo pipefail

cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh

if conda env list | grep -qE '^\s*tesseraq\s'; then
  echo "[setup] env 'tesseraq' already exists — skipping create"
else
  conda create -n tesseraq python=3.10 -y
fi

conda activate tesseraq
pip install --upgrade pip
pip install -r TesseraQ/requirements.txt

# Sanity import check — fail loudly if a critical dep is missing.
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
echo "TESSERAQ_ENV_SETUP_DONE"
