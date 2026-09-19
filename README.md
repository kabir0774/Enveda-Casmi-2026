# casmi26

Working code for the **Enveda CASMI 2026 - Molecule ID From Mass Spectra** Kaggle
competition. Train on your own GPU, run inference-only in the Kaggle notebook.

```
casmi26/
  metric.py      MRR@25, RDKit tautomer + InChIKey14 matching, submission validator
  preprocess.py  peak cleaning, deisotoping, windowed top-N, fragment + neutral-loss binning
  library.py     sparse cosine library search over both channels
  fusion.py      per-molecule evidence fusion, confidence features
  splits.py      class-stratified CV by InChIKey14, per-class reporting
  gate.py        one-sided gate (failed), two-sided gate, blending policies, oracle bound
  data.py        train/test parquet loading, filtering, subsampling
  model.py       PeakFormer -- spectrum transformer predicting a Morgan fingerprint
  torch_data.py  Dataset, collation, per-bit pos_weight
  synth.py       synthetic spectra for testing the pipeline
scripts/
  setup_env.sh          venv + torch for the machine's CUDA + deps
  download_data.sh      kaggle CLI download (you supply your own API token)
  inspect_data.py       what is actually in train.parquet
  gate_experiment.py    end-to-end policy comparison
  train_fingerprint.py  GPU training for the fingerprint head
```

## Quickstart on the GPU box

```bash
git clone git@github.com:<you>/casmi26.git && cd casmi26
bash scripts/setup_env.sh          # venv, CUDA torch, deps
source .venv/bin/activate
bash scripts/download_data.sh      # needs ~/.kaggle/kaggle.json, see script header
python scripts/inspect_data.py --train-parquet data/train.parquet

# prove the training loop works before spending GPU hours
python scripts/train_fingerprint.py --synthetic --structures 900 --epochs 14 \
    --batch-size 64 --d-model 128 --layers 3 --out runs/smoke

# then the real thing, small first
python scripts/train_fingerprint.py --train-parquet data/train.parquet \
    --structures 40000 --max-spectra-per-structure 6 --epochs 20 \
    --batch-size 256 --d-model 256 --layers 6 --out runs/fp_v1
```

`--structures` controls the subsample. Start at 20-40k structures, confirm the
retrieval metric moves, then scale. Going straight to all 275k structures on run
one wastes days finding out a hyperparameter was wrong.

## Run the policy experiment (no data needed)

```bash
python scripts/gate_experiment.py --fp-accuracy 0.65   # one detailed run
python scripts/gate_experiment.py --sweep              # across retrieval strengths
```

Runs in about 10 seconds on 2 CPU cores.

## What training reports

Validation prints bit-level F1 **and** a retrieval proxy (top-1/5/20 against a
database built from the held-out structures). Watch the retrieval numbers. Bit
F1 alone is misleading: a model can score well by getting common bits right and
still never rank the correct molecule first, which is the only thing MRR@25
pays for.

Smoke run on synthetic data, 765 train structures, 14 epochs, CPU:

```
ep  0  val 0.3919  bitF1 0.145  top1 0.007  top20 0.148
ep  7  val 0.2284  bitF1 0.226  top1 0.044  top20 0.637
ep 13  val 0.2160  bitF1 0.251  top1 0.059  top20 0.696
```

That is the loop working, not a result. Synthetic spectra are not chemistry.

## Publishing this repo

**Make the GitHub repo private.** Competition rule 3.6 says code developed on
the competition data may not be privately shared outside your team, and any
public sharing must go on the Kaggle forum so every competitor gets it. A public
GitHub repo is neither. Private repo, team members as collaborators.

`data/`, `*.parquet`, `runs/`, `*.pt` and `kaggle.json` are gitignored. The
competition data is CC BY-NC 4.0 and must not be redistributed -- do not commit
it, and do not push model weights trained on it to a public host.

## What the experiment measured

Synthetic spectra, 2,976 structures, class mix 40/40/20, 447 molecules scored per
row. Absolute numbers mean nothing about the real competition. The *relative*
behaviour of the policies is what this tests.

`fp_acc` is the per-bit accuracy of a stand-in fingerprint head, i.e. how good
your retrieval model is. It is the one knob that changes the regime.

```
 fp_acc  lib-only  retr-only    naive  dual-pin  dual-soft   oracle
   0.52    0.2966     0.0240   0.2384    0.2957     0.2957   0.3000
   0.56    0.2966     0.0550   0.2591    0.2979     0.2979   0.3084
   0.60    0.2966     0.1260   0.3126    0.2951     0.2932   0.3339
   0.65    0.2966     0.2706   0.3724    0.3408     0.3522   0.4079
   0.72    0.2966     0.5352   0.4802    0.5454     0.5541   0.5888
   0.80    0.2966     0.7336   0.5238    0.7330     0.7247   0.7377
SE ~ 0.019
```

### Findings

**1. Retrieval quality dominates everything.** Moving the fingerprint head from
0.52 to 0.80 accuracy takes the score from 0.30 to 0.73. No fusion policy
changes the result by more than ~0.07. If you only have time for one thing,
build the retrieval model.

**2. A one-sided confidence gate is actively harmful.** Asking only "is the
library top-1 correct?" produces a low probability everywhere when the library
is usually wrong, so the policy hands the top slots to retrieval regardless of
whether retrieval is any good. It scored 0.08 against a 0.30 baseline in the
weak-retrieval regime. The decision has to be relative: score both sources and
compare. `ConfidenceGate` is kept in the code as the documented failure;
`DualGate` is the fix.

**3. Fixed 50/50 blending is a gamble on the regime.** It wins by ~0.02 (about
1 SE, not significant) in the narrow band where both sources are comparable,
and loses badly outside it: -0.06 when retrieval is weak, -0.21 when retrieval
is strong. A learned weight never collapses that way. That is the real value of
gating -- not a headline gain, but insurance against being in the wrong regime
without knowing it.

**4. Hard source selection is not better than soft blending.** Pinning rank 1 to
the more confident source performs the same as blending with learned weights.
A candidate both sources rank highly is stronger evidence than either source
being confident alone, and pinning discards that agreement.

**5. Gating is close to saturated.** A perfect oracle that always picks the
right source beats the learned gate by at most 0.06. The gap to a perfect score
is almost entirely candidate quality, not ranking policy.

## Porting to the real data

Everything is written against plain dicts (`mzs`, `intensities`, `precursor_mz`,
`smiles`, `key`), so loading `train.parquet` with polars and feeding rows
through is the only glue needed. Three things to get right:

- **Split by `inchikey14`, never by spectrum.** The same structure appears in
  several source libraries; a random split leaves the answer in the library.
- **Report per novelty class.** A change that gains on class 2 and loses on
  class 1 nets out differently depending on the hidden mix. `splits.py` does it.
- **RDKit version.** The grader pins 2026.03.3. This was developed against
  2026.03.6. Pin the grader's version in your Kaggle Dataset before trusting
  local scores to the third decimal.

Inference must run with internet disabled, so the candidate database and every
model weight has to be packaged as a Kaggle Dataset in advance.
