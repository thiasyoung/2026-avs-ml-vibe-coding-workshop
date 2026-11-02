# IR carpentry examples

Use these references while developing the main
[01 data-carpentry workbook](../01_IR_Vibe_Coding_Data_Carpentry.ipynb).
They illustrate different hypotheses, not a required sequence or guaranteed improvements.

## Choose a reference

- [01 — Minimal baseline](01_IR_Vibe_Coding_Data_Carpentry.ipynb): a complete
  preparation/save/reload pattern; writes `../outputs/carpentry/`.
- **01a–01d** below: independent alternatives starting from the same native data,
  not from one another's outputs.
- [03 — Compare reference results](03_IR_Compare_Carpentry.ipynb): reads compatible
  saved classifier runs for the baseline and four strategies. It does not discover
  arbitrary participant experiments.

The baseline uses recorded numerical coordinates/signals without distinguishing
absorbance from transmittance. It rejects malformed, nonfinite or incomplete
records, linearly interpolates to the reference grid and min-max normalizes each
signal. The alternatives investigate assumptions that this simple approach ignores.

| Example | What it changes |
| --- | --- |
| [01a — Omit transmittance](01a_IR_Omit_Transmittance.ipynb) | Keep absorbance only; convert coordinates explicitly, require full coverage, interpolate, and deduplicate canonical molecules while quarantining contradictory labels. |
| [01b — Convert transmittance](01b_IR_Convert_Transmittance.ipynb) | Convert transmittance to absorbance, require full coverage, interpolate, and check labels against documented SMARTS. |
| [01c — Padding and quality](01c_IR_Padding_and_Quality.ipynb) | Convert signals, retain spectra with at least 80% coverage, pad boundaries, and filter above the training-only 99.5th-percentile adjacent-point jump. |
| [01d — Resampling and balancing](01d_IR_Resampling_and_Balancing.ipynb) | Convert/pad, bin-average to 250 points then upsample to 500, and cap rare-category oversampling at three presentations per original observation. |

Every alternative includes an examination prompt, audit, implementation and
uniform-data check. Review missing/nonbinary labels, NaN/Inf, unit categories,
invalid/duplicate molecules, coverage and statistical tails before deciding on a repair.
Unusual chemistry is not automatically an error; a SMARTS check may find no corrections.

Padding assumes no absorption outside measured support; it is not a measurement.
Resampling to 250 points loses detail, and upsampling adds no measured resolution.
All retained native spectra in these examples are denser than 500 points.
A jump filter can reject real peaks or padding boundaries. Oversampling repeats
existing observations, changes co-occurring label frequencies and combines with
the classifier's class weights. Inspect outcomes rather than assuming these choices help.

## Select a dataset in notebook 02

Set `CARPENTRY_DIR` in the main
[02 classifier](../02_IR_Optimized_Training.ipynb) to one module-relative directory:

- Baseline: `outputs/carpentry`.
- 01a: `examples/outputs/01a_omit_transmittance`.
- 01b: `examples/outputs/01b_convert_transmittance`.
- 01c: `examples/outputs/01c_padding_and_quality`.
- 01d: `examples/outputs/01d_resampling_and_balancing`.
- Your workbook: `outputs/workshop_carpentry`.

Each directory contains `IR_train.parquet`, `IR_validation.parquet`,
`IR_test.parquet`, `data_manifest.json` and `cleaning_audit.csv`.
Every retained training spectrum has 500 finite normalized `IR` values, the ordered
500-point `wavenumber` grid and 12 binary labels. Coverage/padding/resampling columns
are audit metadata, not extra model features.

Keep `../data/frozen_evaluation/` unchanged. Validation and test each contain
975 spectra; their files, hashes, labels, grid and ordering stay byte-identical
across attempts. Exclude all native observations of held-out molecules before
training carpentry. Do not recreate the benchmark or use test scores to choose
filters, balancing or thresholds.

## Run the optional reference comparison

Use the [workshop environment](../../SETUP_GUIDE.md) and download the native IR
checkpoint first. Read the saved outputs without rerunning if you only need a reference.
A complete reproduction trains five models and may take substantially longer
than the live carpentry exercise.

Batch scripts and the strategy registry are in this folder; shared preparation,
examination and validation functions are in `../lib/`.

### Prepare, train and compare all five references

Code Summary: Reuses the native checkpoint, prepares the reference datasets, trains the unchanged classifier and populates reference 03.

From the repository root:

```bash
uv run --locked python 02_ir_functional_groups/examples/run_carpentry_comparison.py --prepare
```

The batch runs IR 00 with live collection disabled, the completed 01 **in this folder**,
then 01a–01d. It does not populate the main prompt-only workbook. The native checkpoint
is required; normalized rehearsal data cannot substitute for it.

**This command updates saved reference notebooks and outputs.** It populates main
IR 02 with the baseline run and resets its selector to `outputs/carpentry`.
Choose `outputs/workshop_carpentry` again to resume your own exercise.
Executed per-option training copies go under `../outputs/carpentry_comparison/`.

### Prepare and check the four alternatives without training

Code Summary: Executes 01a–01d and smoke-tests their output files with the actual static trainer loader.

```bash
uv run --locked python 02_ir_functional_groups/examples/run_examples.py
```

### Verify a completed comparison

Code Summary: Checks selected weights, reloaded predictions, metrics, stopping decisions, input schemas and populated notebooks.

```bash
uv run --locked python 02_ir_functional_groups/examples/verify_carpentry_comparison.py
```

## Interpret the results

All fits use only `IR` features with the same seed, architecture and settings:
AdamW, initial/maximum LR `1e-5`, cosine annealing with warm restarts, and the robust
20-epoch validation-slope stopping rule. Each restores its validation-selected
checkpoint and saves real history, predictions, metrics and plots.
`strategies.py` lists the active alternatives; other output folders are ignored.

Reference 03 accepts only current matching data/model hashes and settings, and
marks stale or missing runs explicitly. It reports selected epochs and total
epochs because different stopping times can affect the comparison.
Validation F1 uses validation-tuned thresholds. These are bundled strategies and
single seeded fits, not causal estimates or uncertainty intervals. Report gains,
ties and regressions; do not repeatedly consult test scores to choose a winner.
