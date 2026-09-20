# Submitting to Kaggle

The scoring notebook runs with **internet disabled** and a **9-hour cap**, so
everything it needs must be uploaded first. This is the whole procedure.

## 1. Upload two Kaggle Datasets

From the machine that has the trained checkpoint:

```bash
pip install kaggle          # credentials already at ~/.kaggle/kaggle.json

# (a) the code
mkdir -p /tmp/ds_code && cp -r casmi26 /tmp/ds_code/
cd /tmp/ds_code
kaggle datasets init -p .
# edit dataset-metadata.json: set "title" and "id" to "<username>/casmi26-code"
kaggle datasets create -p . -r zip

# (b) the model weights
mkdir -p /tmp/ds_model && cp runs/fp_all_full/best.pt /tmp/ds_model/
cd /tmp/ds_model
kaggle datasets init -p .
# edit dataset-metadata.json: "<username>/casmi26-weights"
kaggle datasets create -p .
```

To update either later:

```bash
kaggle datasets version -p . -m "note about what changed"
```

Both are private by default. Keep them private — competition rule 3.6 forbids
sharing competition code outside your team except through the Kaggle forum.

## 2. Create the notebook

New Notebook on the competition page, then:

- **Add Data**: the competition dataset, `casmi26-code`, `casmi26-weights`
- **Settings → Accelerator**: GPU (only needed if using the model)
- **Settings → Internet**: OFF. Turn it off from the start so a dependency
  that only works online fails now rather than at re-run.

## 3. Notebook contents

```python
import sys, subprocess
sys.path.insert(0, "/kaggle/input/casmi26-code")

# rdkit and lightgbm are not in the base image. If the pinned rdkit is
# unavailable offline, upload wheels as a third dataset and install with
#   pip install --no-index --find-links=/kaggle/input/casmi26-wheels rdkit
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "rdkit==2026.3.3", "polars", "lightgbm"], check=False)

subprocess.run([
    sys.executable, "/kaggle/input/casmi26-code/scripts/predict.py",
    "--test-parquet", "/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra/test.parquet",
    "--train-parquet", "/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra/train.parquet",
    "--checkpoint", "/kaggle/input/casmi26-weights/best.pt",
    "--out", "/kaggle/working/submission.csv",
], check=True)
```

`scripts/` must be inside the code dataset for that path to resolve — copy the
whole repo, not just the package.

## 4. Smoke test before the real run

```
--limit 20
```

Twenty molecules proves the plumbing in minutes. Only then remove it.

## 5. Check the output

```python
import polars as pl
sub = pl.read_csv("/kaggle/working/submission.csv")
assert sub.height == pl.read_parquet(TEST).select(pl.col("molecule_id").n_unique()).item()
assert sub["molecule_id"].n_unique() == sub.height
assert sub["smiles"].null_count() == 0
assert sub["smiles"].str.count_matches(";").max() <= 24
```

The grader rejects a submission for a missing column, an empty file, a null, a
repeated `molecule_id`, or more than 25 guesses in any row. `predict.py`
validates all of these before writing, but check again after the re-run.

## 6. Submit

Save Version → Save & Run All. When it finishes, Submit from the notebook's
Output tab.

Five submissions per day. Two final submissions count for the leaderboard, so
near the deadline pick two that differ in approach, not two variants of one.

## Timing

Measured locally, extrapolated to the full data:

| stage | estimate |
| --- | --- |
| build library (1.88M spectra) | ~8 min |
| search 1,213 test spectra | ~20 min |
| entropy rescore | ~2 min |
| database fingerprints (275k) | ~3 min |
| **total** | **~35 min of 540** |

If a run is heading for the cap, `--library-structures` is the dial: it caps
how many structures enter the spectral library.

## What predict.py guarantees

- Every `molecule_id` in the file appears exactly once, including molecules
  whose spectra are all unusable — the molecule list is read from the file, not
  from the parsed spectra.
- Exactly 25 guesses per row, always. A wrong guess costs only the slot.
- A molecule that raises falls back to the most common training structures
  rather than taking the run down.
- Validated against every documented rejection condition before writing.

Tested against a deliberately pathological test file: empty peak lists,
single-peak spectra, a 20,000-peak spectrum, NaN intensities, a zero precursor
mass, and an unrecognised adduct. All 65 molecules came back with 25 guesses.
