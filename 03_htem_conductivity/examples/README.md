# HTEM completed model references

The main module is **00 acquisition → 01 completed data preparation → 02 prompt-only
vibe-coding workbook**. Use these executed notebooks as optional implementation and
analysis references while developing your own model experiment.

| Reference | Purpose |
| --- | --- |
| [02: linear Ridge](02_HTEM_Process_Emulator_Vibe_Coding.ipynb) | Completed baseline, evaluation, spatial maps and saved-model inference. |
| [02a: kernel Ridge](02a_HTEM_Kernel_Ridge.ipynb) | Exact RBF kernel model with validation-selected hyperparameters. |
| [02b: feed-forward network](02b_HTEM_Feed_Forward_Network.ipynb) | Neural regressor with masked targets and validation monitoring. |
| [02c: grid GNN](02c_HTEM_Grid_GNN.ipynb) | Geometry-only neighboring locations, one prediction per grid node. |
| [02d: Random Forest](02d_HTEM_Random_Forest.ipynb) | Nonlinear tree ensemble fit on each property's observed labels. |
| [03: architecture comparison](03_HTEM_Compare_Architectures.ipynb) | Checks and compares the five saved reference runs without training. |

All references locate the HTEM module from the repository, module or examples folder;
data and generated outputs stay at the module root, not in this folder.

For a full reference comparison, run the main 01 once, then all five model references,
then 03. Each uses the same checksummed prepared matrices, features, label masks and
whole-deposition/recipe splits. Missing or stale results are errors, not silently
substituted scores. Exact kernel Ridge can require several GiB of memory.

Shared functions live in `../lib/` and are loaded by each notebook's setup cell.
You can read saved figures without fitting a model. To rerun 03, first generate
all five compatible model runs; saved weights and run folders are not all supplied
with a repository download. See [technical notes](../TECHNICAL_NOTES.md) for
execution commands, checksums and troubleshooting.

Reference runs write `../outputs/htem_architectures/<registered_experiment_id>/` and
replace that reference model's artifacts when rerun. Your workbook results instead
belong under `../outputs/htem_workbook/<unique_experiment_id>/`. The reference comparison
does not automatically discover custom workbook runs; compare them in your workbook.

Keep [notebook 01](../01_HTEM_Process_Data_Carpentry.ipynb)'s preparation fixed. Tune on
validation, not test scores; preserve per-location XY predictions and reconstruct maps
without averaging depositions or filling holes. Report regressions and missed gradients
honestly. See [technical notes](../TECHNICAL_NOTES.md) for reference execution commands,
feature provenance and limitations.
