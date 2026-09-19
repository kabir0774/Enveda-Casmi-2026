#!/usr/bin/env bash
# Environment setup for the GPU box. Run once after cloning.
#
#   bash scripts/setup_env.sh
#
# Creates a venv, installs torch for the machine's CUDA, then the rest.
set -euo pipefail

PY=${PY:-python3}
VENV=${VENV:-.venv}

$PY -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip wheel

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "== GPU detected =="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  # cu124 works on driver 550+. Check `nvidia-smi` and pick the matching index
  # URL from https://pytorch.org/get-started/locally/ if this fails.
  pip install torch --index-url https://download.pytorch.org/whl/cu124
else
  echo "== no GPU found, installing CPU torch =="
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements.txt
pip install -e .

python - <<'PY'
import torch, rdkit, polars
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
print("rdkit", rdkit.__version__, "polars", polars.__version__)
PY
echo "done. activate with: source $VENV/bin/activate"
