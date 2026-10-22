# Shared frozen HTEM feature matrices

Instructor-selected representation: **curated composition descriptors + frozen
ALIGNN reference-crystal embeddings + reference DOS**, plus the existing recipe,
source/power pairing, physical XY and local-temperature inputs.

- 361 encoded input columns; 36,762 labeled positions from 859 depositions.
- Whole-deposition/identical-recipe training/validation/test splits are unchanged.
- `X` is the model input; `y` contains two standardized log10 targets, with
  `y_mask` distinguishing observed labels. Missing targets are never imputed.
- IDs, physical XY and legacy-evaluation flags remain metadata beside the arrays;
  IDs and label-availability masks are not model features.
- Imputation, scaling, categorical vocabularies and target scaling use training
  data only. Reference-crystal PCA was fitted on unique active training compositions.
- All five architectures load these exact matrices when generated preparation is
  absent. Notebook 01 reproduces preparation in `outputs/htem_prepared/` and shows
  executable code rebuilding the selected features and checking matrix equality.

`manifest.json` records ordered features, transformations, selection provenance,
label counts, limitations and artifact checksums. The clean table contains targets
and auditing metadata, but only the explicitly listed feature columns are encoded.
Full planned substrate grids include unlabeled locations for graph inference.

The instructor study selected features on the existing grouped validation data,
not test results. Its modest gain is not independently statistically established.
Reference crystals/DOS do not identify actual source-target or film phases.

Keep this folder and its companion reference/specification files in Git. Do not
regenerate features separately inside an architecture notebook. A deliberate new
feature specification requires rerunning preparation, all five models and comparison.
