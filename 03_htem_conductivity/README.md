# HTEM spatial process-emulator workshop

## Objective

Predict **thickness and conductivity at each substrate location** from deposition
settings, source-material features, physical X/Y coordinates, and a specified local
temperature profile. Reconstruct the local predictions into maps rather than assigning
one average prediction to a whole substrate.

The hands-on focus is **model development through prompting**: establish a baseline,
propose an algorithm or architecture change, run an experiment, and analyze the outcome.
Data preparation and chemical features are supplied and shared across models. Measured
film properties and spectra are not prediction inputs.

## Notebook sequence

- [00 — Get data](00_HTEM_Get_Data_NREL.ipynb): collect compact HTEM process and
   scalar records without retaining full spectra.
- [01 — Prepare data](01_HTEM_Process_Data_Carpentry.ipynb): complete cleansing,
   feature encoding, training-only preprocessing, and grouped data splitting.
- [02 — Vibe-coding workbook](02_HTEM_Vibe_Coding_Workbook.ipynb): edit one prompt,
   then ask your coding assistant to populate and execute baseline, experiment,
   analysis, evaluation and saved-model inference steps using the supplied data.

Optional [completed references in examples/](examples/README.md) include the original
02 linear Ridge workflow, alternatives 02a–02d, and the original 03 architecture comparison.
They are reference points, not additional main workshop steps.

Reusable download, chemistry, preparation, graph and evaluation helpers live in
`lib/`. Notebook setup cells load them automatically from the repository, module,
or examples folder. Data and output paths remain at this module's root.

## Using the workbook

Edit the starting approach and the blanks for your hypothesis, plan, vibe-coding
strategy, effort priorities and compute/search budget. Baseline, training and analysis
methods are editable suggestions. Only the **FIXED input/output contracts** stay
unchanged; copy the whole block into your coding assistant. Notebook 01 supplies the
shared features and grouped splits; the exercise changes models, not preparation.

Workbook results go under `outputs/htem_workbook/<unique_experiment_id>/`, with separate
baseline and experiment artifacts. Supplied reference runs remain under
`outputs/htem_architectures/`; the reference 03 compares those registered runs, not
arbitrary participant experiments. The generated workbook handles its own comparison.

## Before running

Follow the [workshop setup guide](../SETUP_GUIDE.md) and select the workshop `.venv`
kernel in VS Code. Run each notebook top-to-bottom. Compact acquisition snapshots and
frozen feature matrices are included; external feature-extraction packages are not
needed for normal workshop use.

For the optional reference comparison, run 01 once, then the completed 02 and all four
alternatives in `examples/`, and finally `examples/03`.
Keep the prepared data unchanged between models. Training can use the distributed
frozen matrices if generated preparation is absent; stale generated data are rejected.
The exact-kernel example requires several GiB of RAM and more time than the simple baseline.
Generated models, predictions, and figures go under `outputs/htem_architectures/`.

## Keep the comparison fair

Keep entire depositions and repeated recipes together when splitting; many locations
on one substrate are not independent experiments. All models share the same features
and held-out data. Use validation results for model selection, not test scores.

Point models predict each requested location independently; the grid GNN additionally
uses neighboring locations. Missing property labels are masked, not invented. Inspect
spatial maps and failures as well as aggregate scores: the supplied models still miss
some gradients, and conductivity errors remain large.

See [technical notes and run history](TECHNICAL_NOTES.md) for feature provenance,
reproduction commands, memory requirements, and detailed limitations.
