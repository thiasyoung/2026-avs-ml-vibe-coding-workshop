# IR functional-group workshop

## Objective

Identify **which functional groups are present in a molecule from its IR spectrum**.
This is a multi-label task: one spectrum can have several of the 12 labels, which are
derived from molecular structures using documented SMARTS rules.

The hands-on focus is **data examination and carpentry**, not model tuning. Form a
hypothesis about the data, prompt a coding assistant to implement a justified change,
then measure its effect using the same static neural network and fixed evaluation data.

## Notebook sequence

1. [00 — Get data](00_IR_Get_Data_NISTChemPy_v2.ipynb): acquire native NIST spectra,
   their units, SMILES, and labels. This is pre-work, not the live workshop exercise.
2. [01 — Examine and prepare](01_IR_Vibe_Coding_Data_Carpentry.ipynb): start with a
   deliberately simple numeric baseline, then investigate more careful carpentry.
   [Examples 01a–01d](examples/README.md) demonstrate signal conversion, coverage,
   padding/resampling, duplicate and label checks, quality filtering, and balancing.
3. [02 — Static training](02_IR_Optimized_Training.ipynb): select a prepared dataset
   with `CARPENTRY_DIR`; train and evaluate the unchanged functional-group classifier.
4. [03 — Compare results](03_IR_Compare_Carpentry.ipynb): compare compatible runs,
   aggregate F1 scores, per-group results, and stopping behavior.

## Before running

Follow the [workshop setup guide](../SETUP_GUIDE.md), select its `.venv` kernel in VS Code,
and run notebooks top-to-bottom from this module folder.

Obtain the native checkpoint from the instructor or enable live acquisition in 00:

```text
outputs/nist_acquisition/ir_raw_checkpoint.parquet
```

This large file is not included in a normal repository clone. Notebook 00 reuses it
with live collection off by default; it does not automatically download a release asset.
The included normalized rehearsal datasets cannot replace the native arrays for these exercises.

## Keep the comparison fair

Every preparation must finish with **500 finite spectral features and the same 12 binary
labels**. Keep `data/frozen_evaluation/` unchanged and exclude held-out molecular identities
before training-data carpentry. Notebook 02 fits scaling on training only and selects
checkpoints and thresholds using validation data, not test scores.

Keep architecture and training settings fixed while changing data. Review actual gains,
ties, and regressions: sophisticated cleansing is not guaranteed to help, and each worked
example bundles several choices. Prepared data and model results go under `outputs/`;
worked-example data go under `examples/outputs/`. See the [examples guide](examples/README.md)
for the full preparation/training/comparison command and detailed policies.
