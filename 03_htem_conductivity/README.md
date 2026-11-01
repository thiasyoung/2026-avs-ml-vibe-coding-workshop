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
- [02 — Build an emulator](02_HTEM_Process_Emulator_Vibe_Coding.ipynb): start with
   linear Ridge regression, then use prompts to develop and evaluate experiments.
- [03 — Compare outcomes](03_HTEM_Compare_Architectures.ipynb): compare saved results,
   including test-set parity plots for both properties. This notebook does not train models.

Worked alternatives to `02`: [02a — Kernel Ridge](examples/02a_HTEM_Kernel_Ridge.ipynb),
[02b — Feed-forward network](examples/02b_HTEM_Feed_Forward_Network.ipynb),
[02c — Grid GNN](examples/02c_HTEM_Grid_GNN.ipynb), and
[02d — Random Forest](examples/02d_HTEM_Random_Forest.ipynb).

## Before running

Follow the [workshop setup guide](../SETUP_GUIDE.md) and select the workshop `.venv`
kernel in VS Code. Run each notebook top-to-bottom. Compact acquisition snapshots and
frozen feature matrices are included; external feature-extraction packages are not
needed for normal workshop use.

For a full comparison, run 01 once, then 02 and all four alternatives, and finally 03.
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
