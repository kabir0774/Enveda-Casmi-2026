#!/usr/bin/env bash
# Download the competition data to ./data.
#
# Prerequisite, done by YOU on the machine, once:
#   1. kaggle.com -> your profile -> Settings -> API -> Create New Token
#   2. save the downloaded kaggle.json to ~/.kaggle/kaggle.json
#   3. chmod 600 ~/.kaggle/kaggle.json
#
# The token is a credential. Do not paste it into a chat, a notebook, or this
# repo -- .gitignore blocks kaggle.json, but the safest place is ~/.kaggle only.
set -euo pipefail

COMP=enveda-CASMI26-molecule-id-mass-spectra
DEST=${1:-data}

if ! command -v kaggle >/dev/null 2>&1; then
  pip install kaggle
fi
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "ERROR: ~/.kaggle/kaggle.json not found. See the header of this script." >&2
  exit 1
fi

mkdir -p "$DEST"
echo "downloading ~3.04 GB to $DEST ..."
kaggle competitions download -c "$COMP" -p "$DEST"
unzip -o "$DEST/$COMP.zip" -d "$DEST"
rm -f "$DEST/$COMP.zip"
ls -lh "$DEST"

python - <<PY
import polars as pl
lf = pl.scan_parquet("$DEST/train.parquet")
print("train columns:", lf.collect_schema().names())
print("train rows:", lf.select(pl.len()).collect().item())
PY
