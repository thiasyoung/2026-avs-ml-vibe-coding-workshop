# Four worked IR data-carpentry examples

Run IR 00 (checkpoint reload by default), the main IR 01 baseline, then these four
independent examples. They start from the same native records, not from each other's
outputs. Every example contains an examination prompt, executed audit, a repair
hypothesis and implementation, and a final saved/reloaded uniform-data acceptance check.

The main **01 baseline is deliberately naïve**: use recorded numeric coordinates and
signals as-is, reject malformed/nonfinite/incomplete records, linearly resize to the
reference grid and min-max normalize each signal. It does not discuss or distinguish
absorbance from transmittance. It is not secretly corrupted to force a bad score.
The examples explicitly investigate the measurement assumptions that baseline ignores.

| Example | Signal/bounds/resampling policy | Additional cleansing hypothesis | `CARPENTRY_DIR` in IR 02 |
| --- | --- | --- | --- |
| [01a: omit transmittance](01a_IR_Omit_Transmittance.ipynb) | Absorbance only; explicit coordinate conversion; trim/full coverage; linear up/downsampling | Canonical-molecule deduplication; quarantine contradictory duplicate labels | `examples/outputs/01a_omit_transmittance` |
| [01b: convert transmittance](01b_IR_Convert_Transmittance.ipynb) | Explicit transmittance-to-absorbance conversion; trim/full coverage; linear up/downsampling | Check/correct labels against the documented SMARTS | `examples/outputs/01b_convert_transmittance` |
| [01c: padding and quality](01c_IR_Padding_and_Quality.ipynb) | Conversion; trim/interpolate; explicitly pad boundary regions when at least 80% is measured | Filter above the training-only 99.5th-percentile adjacent-point jump | `examples/outputs/01c_padding_and_quality` |
| [01d: resampling and balancing](01d_IR_Resampling_and_Balancing.ipynb) | Conversion/padding; bin-average to 250 measured-region points, then upsample to the common 500-point interface | Capped rare-category oversampling; at most three presentations per original observation | `examples/outputs/01d_resampling_and_balancing` |

All four examine missing/nonbinary categories, NaN/Inf inside arrays, unit categories,
invalid/duplicate molecular identities, native lengths, coverage and statistical tails.
Unusual chemistry is not automatically an error. Label checks may legitimately make
zero corrections. Sharp real peaks or padding edges may trigger the quality filter.
All retained native spectra are denser than 500 points. Example 01d deliberately
tests a lower-resolution 250-point representation and then upsamples it to the common
interface, making both operations visible. It loses detail, may hurt accuracy, and
creates no new measured resolution. Padding assumes no absorption in
unmeasured regions and is not an observation. Oversampling repeats existing examples,
alters co-occurring class frequencies and combines with the static classifier's class
weights. None of these choices is guaranteed to help prediction.

## Uniform training interface and fixed evaluation

Every example ends with the same prepared training schema, including 500 finite
normalized `IR` values, the same ordered 500-point `wavenumber` grid and 12 binary
target columns. `padded_points`, `measured_fraction`, `source_points` and
`resampling_action` are audit metadata, never ML features. The outputs include:

- `IR_train.parquet`, `IR_validation.parquet`, `IR_test.parquet`;
- `data_manifest.json`, `cleaning_audit.csv`, examination tables and plots.

Keep `../data/frozen_evaluation/` in Git. Both reference files contain 975 spectra;
the original validation/test Parquets and hashes are preserved byte-for-byte. All
native observations of held-out molecular identities are excluded before any training
carpentry. No balancing, filtering, resampling or threshold decision uses test scores.
IR 02 rejects changed reference copies, labels or grids. Do not delete/recreate the
benchmark while trying different policies. Changing its grid requires a separate
benchmark version, not a different test transformation for each strategy.

## Execute and compare

From the repository root, prepare everything, train five complete models with the
unchanged IR 02 settings, and execute/populate IR 03:

```bash
uv run --locked python 02_ir_functional_groups/run_carpentry_comparison.py --prepare
```

IR 00 uses the existing native checkpoint with live collection disabled by default;
no redownload is needed. The native checkpoint is required: already normalized
rehearsal arrays are not silently substituted. To prepare and loader-smoke-test just
the four examples without model fitting, run:

```bash
uv run --locked python 02_ir_functional_groups/examples/run_examples.py
```

IR 02 reads only `IR` features. All fits use the same seed, architecture, AdamW with
initial/maximum LR `1e-5`, cosine annealing with warm restarts and the existing robust
20-epoch validation-slope stopping rule. Each fit restores its validation-selected
checkpoint and saves history, predictions, metrics, weights and plots. The batch
populates IR 02 with its real baseline run and saves executed per-option copies under
`outputs/carpentry_comparison/`, executed with the IR module as working directory.
IR 03 selects current matching data/model hashes, displays a summary table and figure,
and marks stale or missing results explicitly. `strategies.py` is the single active
list used by preparation, training and comparison; old output directories are ignored.

Report the actual scores, including regressions and ties. These are bundled strategies
and single seeded fits, not causal estimates or uncertainty intervals. Validation F1
uses validation-tuned thresholds. Different stopping times can confound the comparison;
IR 03 reports both selected epochs and total epochs run. Do not choose a winner by
repeatedly consulting test scores.

After a complete batch, independently verify the actual selected weights, reloaded
predictions, metric calculations, early-stop/selection decisions, matching settings,
input schemas and populated notebooks:

```bash
uv run --locked python 02_ir_functional_groups/verify_carpentry_comparison.py
```

The six superseded notebooks were moved to `archive/before_consolidation/` for local
recovery; that ignored archive is not part of the active workshop or batch. Old model
outputs were retained for provenance, not reused as results of the new examples.
