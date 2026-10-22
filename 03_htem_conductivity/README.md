# HTEM spatial process-emulator workshop

Predict **local thickness and conductivity versus substrate position** from a deposition recipe and a requested substrate grid. A substrate is not reduced to one average row. Each location has a `(sample_library_id, position)` key and its own XY inputs and outputs.

Prediction flow: **one row per location (recipe + x_mm + y_mm + local temperature) → two local
predictions → save point identities/coordinates/predictions → arrange by XY for visualization only**.
Ridge, Kernel Ridge, the MLP, and Random Forest predict queried locations independently: predicting one point alone,
in a batch, or in a different row order must give the same result within numerical tolerance. Plotting
uses an exact XY pivot, never a substrate-average prediction, position-number reshape, interpolation,
or smoothing. Missing grid cells stay blank. The `position` identifier is metadata, not a substitute
for physical X/Y coordinates. Predicting all rows in a batch does not make them a joint substrate output.

The previously requested **GNN is a neighbor-aware comparison**, not an independent point model:
it still outputs thickness/conductivity per location, but uses other nodes' process/coordinate features
and grid connectivity. It needs the full planned grid; removing nodes may change its predictions.
There is no graph-level output pooling. Keep this distinction explicit when comparing architectures.

## Current expanded-data workflow (2026-10-21)

Preparation v5 keeps positions with **either** valid thickness or conductivity, rather than requiring both.
Missing/invalid labels remain NaN; labels are never imputed. Linear Ridge, exact RBF kernel Ridge, and Random Forest fit
each property on all of its observed training rows. MLP/GNN training uses masked loss with equal mean
MSE per observed property, preventing the more numerous thickness labels from dominating validation.
The stored arrays include a per-target `y_mask`; graph node masks separately retain the full planned grid.

The acquisition now retains the scalar API's `absolute_temp_c`, encoded as **local_temperature_C**.
This can vary over a substrate and is not equivalent to library initial temperature. Pass a specified or
calibrated local-temperature column with the planned inference grid. Missing profile values use train-only
imputation/indicators; no silent uniform-temperature assumption or reconstruction from measured outputs.
This field is not a full temperature trajectory. Compact old scalar caches automatically refresh once
to obtain it; no spectra are retained.

All original library split assignments and the original 9,482 paired observations are frozen in
`data/frozen_evaluation/paired_split.csv`. New groups fill material-class split deficits around those anchors.
This retains 60/20/20 as an approximate goal, not permission to move old holdouts. New rows on old libraries
inherit their split. Notebook 03 reports expanded-test metrics with **different observed counts per property**,
plus a separate before/after table on exactly the **original 2,108 paired test points**. Old/new target scaling
differs because training data grew, so standardized scores are not direct before/after measures.

The expanded pool contains **36,762 positions, 859 depositions, 821 connected recipe/library groups**:

| Split | Positions with ≥1 label | Depositions | Independent groups | Thickness labels | Conductivity labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 22,272 | 520 | 494 | 21,360 | 6,377 |
| Validation | 7,127 | 166 | 162 | 6,776 | 2,260 |
| Test | 7,363 | 173 | 165 | 7,182 | 2,289 |

Previously training had 155 groups, 5,465 thickness labels and 5,465 conductivity labels. Not every new
point is an independent experiment, and new holdout groups are not training examples. The public inventory
remains 1,891 libraries; most of this expansion uses records already present in the compact raw download.
PLD/pulse/energy records still need separately validated method-specific preparation; they are not silently
combined with powered-target recipes. Three upstream libraries remain unavailable (132 declared points).

Exact KRR still uses the full RBF kernel and nine validation-selected alpha/gamma choices, in float64.
For the larger cohort, `ExactRBFRegressor` solves `(K + alpha I)w = y` using
[SciPy CG](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html), checking both
solver success and relative residual ≤1.1e-8. This is numerical convergence of the **full** system, not a
Nyström approximation. Kernels are reused across alphas and predictions are batched to bound memory.
The largest current kernel is about 3.4 GiB; allow adequate RAM and more time than a small workshop demo.
Saved models include all observed training rows. Small tests compare CG predictions with the direct solver.

Pre-expansion outputs are recoverable locally under `archive/paired_labels_2026_10_21/`; compact reference
metrics ship in `data/frozen_evaluation/paired_test_metrics.csv`. Sections below describing dated paired-label
runs are historical, not current dataset sizes or the current partial-label numerical solver.

## Notebook sequence

1. `00_HTEM_Get_Data_NREL.ipynb`: broad public acquisition, compact scalar/process records only.
2. `01_HTEM_Process_Data_Carpentry.ipynb`: completed spatial data preparation and leakage-safe splits.
3. `02_HTEM_Process_Emulator_Vibe_Coding.ipynb`: the default **linear Ridge regression** (`Ridge(alpha=10)`, not unregularized OLS or kernel ridge), with model-experiment prompts, evaluation, parity/maps, and saved-model inference.
4. `examples/02a_HTEM_Kernel_Ridge.ipynb`, `02b_HTEM_Feed_Forward_Network.ipynb`, `02c_HTEM_Grid_GNN.ipynb`, `02d_HTEM_Random_Forest.ipynb`: independent nonlinear/architecture alternatives on exactly the same prepared problem.
5. `03_HTEM_Compare_Architectures.ipynb`: a short table/error chart and a test-parity panel for each algorithm and property. Thickness is in µm, conductivity in log10(S/cm); shared property limits and all points expose poor predictions rather than clipping them. It does not train or change data; the workshop default stays linear regardless of the validation ranking.

Run top-to-bottom with the workshop UV environment/VS Code kernel. From the repository root:

```powershell
uv run --locked python -m nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 03_htem_conductivity/00_HTEM_Get_Data_NREL.ipynb
uv run --locked python -m nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 03_htem_conductivity/01_HTEM_Process_Data_Carpentry.ipynb
uv run --locked python -m nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 03_htem_conductivity/02_HTEM_Process_Emulator_Vibe_Coding.ipynb
uv run --locked python -m nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=2400 03_htem_conductivity/examples/02a_HTEM_Kernel_Ridge.ipynb
uv run --locked python -m nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 03_htem_conductivity/examples/02b_HTEM_Feed_Forward_Network.ipynb
uv run --locked python -m nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 03_htem_conductivity/examples/02c_HTEM_Grid_GNN.ipynb
uv run --locked python -m nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 03_htem_conductivity/examples/02d_HTEM_Random_Forest.ipynb
uv run --locked python -m nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 03_htem_conductivity/03_HTEM_Compare_Architectures.ipynb
```

For an architecture comparison, run **01 once**, then 02/02a/02b/02c/02d and 03 without regenerating preparation. Each algorithm notebook works independently from the module, repository, or `examples` folder. After a deliberate preparation change, rerun all five model notebooks before comparing again.

## Algorithm examples and fair comparison

| Notebook | Algorithm / supplied architecture | Selection / optimization |
| --- | --- | --- |
| 02 (default) | Linear Ridge, alpha 10 | Fixed simple linear baseline; no epochs |
| examples/02a | Exact RBF kernel ridge, one regressor per observed property | Nine alpha/gamma candidates; minimum validation score only |
| examples/02b | ReLU MLP, hidden widths 64–32 | AdamW, cosine decay, validation-minimum checkpoint |
| examples/02c | Grid GNN, width 64, two message layers | AdamW, cosine decay, masked node loss, validation-minimum checkpoint |
| examples/02d | Random Forest per observed property, 300 trees | Fixed seed; depth 20, minimum leaf 3, feature fraction 0.7; grouped validation reporting |

Kernel ridge uses a [`KernelRidge`](https://scikit-learn.org/1.6/modules/kernel_ridge.html) subclass with the full RBF kernel and the checked CG solver described above, not a linear kernel, Nyström approximation, or subsampling. The supplied grid is alpha `[0.1, 1, 10]` and gamma `[0.1, 1, 10] / encoded_feature_count`. The same training-encoded feature matrix and target scaling are reused; there is no new preprocessing. The solver promotes those unchanged values to float64 to avoid dense-kernel numerical roundoff, including batch-size-dependent predictions. The selected candidate remains trained on training data only, not training plus validation. Its dense training kernel scales quadratically in sample count, and fit/search time includes **all nine trials**, unlike the single supplied MLP/GNN fits. These are illustrative, unequal search budgets—not exhaustive optimization.

The shared `htem_experiments.py` handles physical-unit evaluation, per-point identities, fixed full-grid maps, model saving/reloading, and strict comparison. Every run records preparation/array/preprocessing/evaluation-code fingerprints and output checksums. Notebook 03 rejects missing, stale, changed, or mismatched runs, including different actual targets, point ordering, or map examples. It shows train/validation/test standardized MSE, per-target test errors, validation score, and fit/search time. Rank by validation, not test; a single seed/holdout is not evidence of statistically significant superiority.

All five algorithms predict thickness and conductivity at the same labeled locations, and map the same held-out deposition's full grid. The graph may use additional **unlabeled geometry nodes**, never additional labels or measured inputs. Each notebook ends with its own reloadable `predict_substrate(recipe, grid, spacing_mm=None)` function; 02 uses linear Ridge, 02a KRR, 02b its MLP, 02c its GNN, and 02d its forests. The reference reload check is done before editing the recipe, so a deliberate new recipe is not incorrectly compared against the unchanged prediction.

Random Forest uses scikit-learn's [RandomForestRegressor](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html), with separate forests for partial thickness/conductivity labels and four bounded worker threads. It retains the shared training-fitted StandardScaler and standardized log targets; trees do not need scaled inputs, but using identical preparation keeps the comparison controlled. There is no new feature engineering, imputed label, epoch curve, test-based search, or per-substrate output normalization. Ordinary row-bootstrap/OOB performance would not be an independent deposition holdout, so OOB scoring is disabled. Trees may learn recipe/coordinate interactions, but their predictions are piecewise and improved gradients are not guaranteed. The before/after table keeps its Random Forest row with missing previous scores because no archived pre-expansion forest exists.

## Broad, compact acquisition

Notebook 00 defaults to `ACQUISITION_MODE='live_api'`, `LIBRARY_IDS=None`, four bounded recipe-download workers, and resumable compact caches. No course-cohort limit or `has_ele`/`has_opt`/complete-label prefilter is applied. The public library inventory is refreshed each run; responses in compact per-library caches are reused unless `REFRESH=True`. Use explicit `snapshot` mode for fully offline operation. Cached or snapshot data are not described as freshly downloaded.

The working base is `https://htem-api.nlr.gov/api`. Collection uses library details and **`/sample_library/prop?ids=...` scalar batches**, not tens of thousands of full-spectrum sample requests. Rare unsupported libraries use individual sample responses, stripped in memory before any cache is written. No spectrum, XRF composition, contact identity, or full raw HTTP response is retained. Placeholder `[null]` sample-ID lists are reported as zero valid points, not invented observations. Unavailable points remain coverage gaps, not fabricated rows.

Retained recipes include chamber/tool label, duration, initial temperature, pressure, substrate, paired target/power and gas/flow arrays, plus source pulse/repetition/energy/cycle/distance metadata where available. The point table retains substrate XY, position, thickness, conductivity, and optional direct bandgap even when labels/settings are missing. The separate **library catalog retains every public recipe**, including depositions without usable observations.

The distributed snapshot is `data/htem_spatial_snapshot.parquet`, with `htem_spatial_libraries.parquet` and a checksummed provenance/coverage sidecar. Source errors, missing-grid fallbacks, counts, timestamp, and mode are explicit. The former course snapshot and spectral input are superseded, not silently combined with the live crawl.

## Correct substrate positions—not instrument coordinates

The scalar endpoint supplies aligned `position`, **`x_mm`**, **`y_mm`**, and property arrays for each library. These substrate-map coordinates are materially different from the sample endpoint's instrument-stage **`xyz_mm`**, whose Z includes focus/stage height. `xyz_mm` is never a feature, coordinate substitute, or thickness proxy.

Internal `id` is the explicit **library:position** composite key, not `/api/sample`'s numeric ID. Scalar arrays align by their position list; unordered `sample_ids` are never zipped to them. Grids have different sizes. There is no hard-coded 44-point reshape, rotation guessed from measured gradients, coordinate normalization using labels, or deposition averaging. Missing/duplicate coordinates are audited, not repaired by assumption. Prospective predictions require a planned grid **in the same HTEM substrate frame**; other coordinate frames need a separately validated transform.

Notebook 02 preserves IDs/XY beside predictions and displays one held-out library's actual/predicted maps with matched color scales (conductivity logarithmic). Missing cells stay blank; maps do not interpolate or average duplicates. `predict_substrate(recipe, grid)` returns one prediction per requested point, preserving identity/order. Multiple depositions cannot be pooled into one map.

## Completed carpentry

### Instructor-selected frozen chemical features

The temporary instructor notebook
[`scratch/HTEM_Feature_Selection_Study.ipynb`](../scratch/HTEM_Feature_Selection_Study.ipynb)
compares nine feature sets using fixed linear Ridge and Random Forest selectors.
The study reads **only the archived training/validation data**, never test artifacts
or scores. Whole-deposition/recipe splits, label masks, source/power slots and target
scaling stay fixed. Each candidate fits input imputation/scaling on training only.

Candidate additions are curated Magpie-style elemental/valence/stoichiometry
descriptors, published 16-D MEGNet element vectors averaged by stoichiometry, real
frozen ALIGNN reference-crystal embeddings, and the existing reference DOS summaries.
The MEGNet representation is **not a full Roost/AtomSets model**. ALIGNN receives
real JARVIS lattice/species/coordinates, pools atoms before its property head, averages
all composition-matched reference structures, and uses 16 PCA components fitted on
unique **active training compositions only**. It does not identify an actual HTEM
source or film phase. Elemental melting-temperature statistics are not a predicted
compound melting point. Missing reference coverage remains explicit.

Selection minimizes the equal-selector mean relative validation RMSE (each selector
normalized by its existing-DOS baseline; targets equally weighted after training-only
log scaling). See `data/htem_feature_ranking.csv`, `htem_feature_validation_trials.csv`
and `htem_feature_selection.json`. A paired whole-validation-group bootstrap interval
is descriptive after selection, not independent confirmation. The winning representation
is best under this finite experiment, not guaranteed optimal for every architecture.
Previously viewed test results are not a new untouched confirmation set.

The selected candidate is **curated descriptors + frozen ALIGNN reference-crystal
features + reference DOS**, alongside existing chemistry, recipe, physical XY and
local-temperature inputs: **361 encoded columns**. MEGNet composition vectors were
tested but are **not included**. The two fixed selectors improved from 0.75349 to
0.72267 (Ridge) and 0.66583 to 0.65805 (200-tree selection forest). The normalized
equal-selector gain is **2.63%**, with an exploratory paired-group 95% bootstrap
interval of **−2.86% to +8.29%**: it includes no improvement. This is the best of
the nine tested feature sets under that protocol, not established superiority.
The shared 16-component crystal PCA retains 86.3% of training-reference variance.

Notebook 01 section **2a** now contains the selected worked workflow: calculate
elemental descriptors, project cached real crystal vectors with training-only PCA,
fit the training-only encoder and assert exact matrix reproduction. It also defines
the optional official ALIGNN graph/embedding function for instructor use; this was
smoke-tested against a real cached JARVIS structure. Normal workshop execution
uses compact JSON vectors, not DGL/ALIGNN installations or model downloads.

**Workshop deployment:** `data/htem_frozen_source_features.json` supplies offline
source features; `data/frozen_features/` distributes the exact selected training,
validation and test matrices, transformations, identities, full grids and checksums.
Notebook 01 reproduces preparation in `outputs/htem_prepared/`. Every architecture
loads those shared matrices unchanged; if outputs are absent, it uses the distributed
frozen copy. Existing but stale outputs fail explicitly rather than silently falling
back. No architecture selects features, fits its own PCA, or downloads an encoder.
Keep the frozen source/specification/reference and prepared files in Git.

The archive `archive/pre_feature_study_2026_10_21/` preserves the earlier full run.
The instructor's `scratch/freeze_htem_feature_matrix.py` asserts exact equality of
all prior labels, masks, identities, XY, split assignments and selected study matrices
before publishing the frozen preparation. After a deliberate feature change, rerun
01 and **all five** architecture notebooks plus 03. Never mix old/new fingerprints.

#### Executed frozen-feature comparison: 2026-10-21

Notebook 01 and all five architecture notebooks were executed on the selected
361-column preparation; notebook 03 assembles their results. All 36,762 positions,
859 libraries, target values/masks, coordinates, original paired holdouts and full
expanded split assignments were preserved exactly. No acquisition rerun was needed.
The table below replaces the older runs elsewhere in this document as the current
comparison. Selection used validation only; test metrics are subsequent evaluation,
not criteria for picking features or retuning architectures.

| Algorithm | Previous DOS validation score | Selected-feature validation score ↓ | Test thickness MAE (µm) ↓ | Test conductivity log-MAE (decades) ↓ |
| --- | ---: | ---: | ---: | ---: |
| Linear Ridge (02 default) | 0.75349 | 0.72267 | 0.22009 | 1.48492 |
| Exact RBF kernel Ridge (02a) | 0.65678 | 0.66040 | 0.15818 | 1.31983 |
| Feed-forward MLP (02b) | 0.67482 | 0.66278 | 0.15967 | 1.34504 |
| Grid GNN (02c) | 0.65917 | 0.67374 | 0.17006 | 1.36114 |
| Random Forest (02d) | 0.66593 | 0.65761 | 0.18159 | 1.36258 |

The current validation leader is Random Forest, but the default remains linear
Ridge for teaching. The deployed forest has 300 trees, while the feature-selection
selector used 200; do not confuse those two scores. KRR and GNN validation worsened:
the feature study did **not** find an across-the-board improvement. MLP restored
epoch 4 after 44 epochs; GNN restored epoch 68 after 108, using the existing
validation-patience rule. KRR selected alpha 1 and gamma 1/361 in about 458 seconds.
Architecture hyperparameters were not changed to rescue these results.

Conductivity errors still exceed a decade. Linear Ridge has severe inverse-log
extrapolation (test conductivity raw-unit R² approximately −1.11×10⁹), despite a
better validation score. Do not present finite predictions or these modest feature
gains as a reliable process model, calibrated uncertainty, or a solution to missing
spatial process information. The test labels were not clipped or removed.

Verification: **122 repository tests passed**, alongside environment imports,
25-notebook structural validation, the automation contract and whitespace checks.
The study, 01, all five model notebooks and 03 have populated outputs with no failed
cells. `scratch/verify_htem_frozen_run.py` verifies exact distributed/prepared matrix
equality, common run fingerprints, all five saved-model/full-grid reloads, point-model
single/batch/shuffled invariance, and the fresh-checkout frozen-data fallback. The
optional notebook 01 atomic encoder separately reproduced a cached real ALIGNN
vector. Raw reference archives, external weights, extractor environments and trained
model outputs remain ignored; the compact fixed matrices and notebook outputs ship.

#### Reproducing external extraction (instructor only, Windows)

The generated compact JSON references ship with the repository. **Do not install
these extraction packages into the participant environment.** To reproduce extraction:

```powershell
uv venv --system-site-packages --python .venv/Scripts/python.exe 03_htem_conductivity/outputs/htem_feature_study/extractor_env
uv pip install --python 03_htem_conductivity/outputs/htem_feature_study/extractor_env/Scripts/python.exe --no-deps dgl==1.1.2 alignn==2026.8.11 jarvis-tools==2026.6.12 ase==3.26.0 spglib==2.6.0 pydantic-settings==2.12.0 toolz==1.0.0 xmltodict==1.0.2
uv pip install --python 03_htem_conductivity/outputs/htem_feature_study/extractor_env/Scripts/python.exe pydantic==2.12.5 pydantic-settings==2.12.0 lmdb==1.7.5
03_htem_conductivity/outputs/htem_feature_study/extractor_env/Scripts/python.exe scratch/build_htem_feature_candidates.py
```

The builder reuses the pinned JARVIS DOS archive in ignored outputs (run
`htem_source_dos.py` first if missing), downloads immutable-revision Magpie/MEGNet
tables and the pinned 48 MB official ALIGNN formation-energy archive, and records
checksums, versions, graph parameters and coverage. Its legacy training checkpoint
contains optimizer objects, so it is deserialized **only in the instructor extractor**
after verifying the hard-coded upstream archive SHA256; never accept user checkpoints.
Participant preparation only reads plain JSON vectors. Full upstream redistribution
notices are in `data/htem_feature_reference_licenses.json`; attribution is in
`data/CHEMISTRY_REFERENCE_LICENSE.md`. No API key or film information is used.

### Reference DOS candidate and earlier enrichment

The current enrichment adds **reference electronic DOS shapes** from independent
[JARVIS calculations](https://doi.org/10.6084/m9.figshare.14745327.v3), with
[Kaundinya, Choudhary & Kalidindi (2022)](https://doi.org/10.1007/s11837-022-05199-y)
and CC BY 4.0 attribution in `data/CHEMISTRY_REFERENCE_LICENSE.md`. The roughly 95 KB compact
`data/htem_source_dos_reference.json` covers 75 catalog compositions, including **all 66 source
formulas in the current prepared cohort**. It contains 571 matching reference calculations selected
from 55,659 external records using the source catalog only, never HTEM film outcomes or split labels.

For each source, 15 added features describe spectral centroid/width, occupied-window fraction,
near-Fermi (+/-0.5 eV) fraction, seven coarse energy-window fractions, known/count indicators, and
near-Fermi range/mean bin standard deviation across references. Three recipe-level additions are
power-weighted centroid, power-weighted near-Fermi fraction, and DOS power coverage. Power-weighted
values remain missing unless every active source has a reference. Existing train-only imputation,
missing indicators and StandardScaler apply. In that earlier DOS-only enrichment, inputs grew
**136 → 223 encoded columns**; all original
encoded values, observations, labels, target transforms and full split assignments are unchanged.

Each `edos_up` spectrum is unit-area normalized on the sampled support **-5 to 9.95 eV relative to
Fermi** (300 samples, 0.05 eV spacing, nominal upper limit 10 excluded). No extrapolation/padding or
arbitrary `linspace` stretching is used. Feature values are averaged across all valid exact-composition
matches; no preferred polymorph is selected. Reference IDs, checksum and processing rules are saved.
These are **spin-up shape summaries**, not absolute/spin-summed DOS, carrier density, a physical
mixture DOS or deposited-film electronic structure. Reference variation is not calibrated uncertainty.

The HTEM recipes do not provide verified source crystal phases/lattices/atomic coordinates.
The newer instructor study uses real **external reference** structures for ALIGNN features, not
guessed actual target structures. Reference-phase matching and its limitations remain explicit.
The substrate-grid GNN is a separate spatial graph, not an atomic-structure model. DOS descriptors are
constant within a fixed recipe; they add chemistry information, **not source placement or a guaranteed
spatial gradient**. No measured HTEM film spectrum, structure or optical/electrical output enters them.

Reproduce the independent reference acquisition/extraction from the repository root:

```powershell
uv run --locked python 03_htem_conductivity/htem_source_dos.py
```

This streams a pinned **155 MB** public archive into ignored
`outputs/htem_source_reference_cache/`, then writes compact derived summaries. No new packages,
credentials or downloads are needed when running preparation/training with the shipped reference.
After regenerating it, rerun 01, all five architectures, and 03; checksum guards reject stale models.

Source-material one-hot columns are **replaced**, not just supplemented. Each paired source slot supplies atom-fraction-weighted atomic number, atomic mass (u), Pauling electronegativity (mean/range/standard deviation), atomic radius (Å), first ionization energy (eV), elemental O/N/S/Se/Te fractions, number of elements, and composition entropy. Per-source power fractions and power-weighted source properties supply additional process/chemistry interactions. Power weighting is an engineering heuristic: powers are not atomic fluxes, deposited stoichiometry, or a physical mixture band gap.

Source band-gap features are the median and range of independent experimental references matched by normalized source composition. The reference is [Zhuo, Mansouri Tehrani & Brgoch (2018)](https://doi.org/10.1021/acs.jpclett.8b00124), distributed as [Expt Gap Data v1](https://doi.org/10.6084/m9.figshare.7289687.v1) through matminer. Formula matching does **not** establish target phase, doping, temperature, or defects; these are **reference descriptors**, not actual HTEM source/film measurements. Duplicate reference records are retained as distributed, not treated as independent uncertainty estimates. Missing references remain NaN with known/unknown/coverage indicators and training-only imputation. No gap is estimated from elemental gaps or filled from HTEM optical measurements, including at inference.

Atomic properties are a pinned extraction of [pymatgen v2025.1.9](https://github.com/materialsproject/pymatgen/tree/v2025.1.9). The compact offline `data/htem_chemistry_reference.json` records download checksums, units, aggregation, and source citations; [the reference license notice](data/CHEMISTRY_REFERENCE_LICENSE.md) preserves attribution. No new dependencies, accounts, or API keys are needed. `scratch/build_htem_chemistry_reference.py` prints a reproducible reference JSON using the pinned external dataset and complete recipe catalog, never film labels or splits.

Formula names are retained solely for source/power identity, auditing, and material-balanced splits; chamber/substrate/gas remain train-fitted categorical variables. Unknown/unsupported formulas get explicit unknown status, missing descriptors, and an inference warning, not guessed chemistry. `outputs/htem_prepared/source_chemistry_coverage.csv` reports per-formula coverage, gap range, reference count, and citation. Preparation records schema version 5 and chemistry-reference/featurizer hashes; old/stale preparation and model bundles are rejected. Models use identical descriptors at fitting and recipe/full-grid inference. Original recipe identity uses only process settings/source names, so chemistry changes do not move libraries between splits.

Reference inputs are recipe settings, substrate XY, and a specified/calibrated local-temperature profile—not measured composition, spectra, optical/electrical summaries, property targets, quality, IDs, dates, operators, or stage/focus coordinates. Targets are **log10 thickness (µm)** and **log10 conductivity (S/cm)**, independently standardized using observed training labels. Optional bandgap (eV) remains in the raw data but is not required or used as input for this two-output map task.

Training uses a powered-target cohort with positive duration, known valid initial temperature, consistent active-target powers, unambiguous XY, and **at least one** valid output: thickness 0.001–10 µm or positive finite conductivity. An invalid/missing property remains NaN without discarding the other label. Local temperature is retained independently at every position. Labels are never imputed. Statistical extremes are reviewed but not automatically removed. These are eligibility rules, not guessed unit repairs. Extra deposition-mode fields remain in the raw inventory rather than being forced into the sputtering-focused reference features.

Chemical formula case and source/power pairing are preserved; gas/substrate case is normalized without speculatively merging trade names. Optional settings use training-only median imputation and indicators. **XY is never imputed.** All positions of one deposition stay in one split; identical recipes are grouped even on different grids. Preprocessing fits training only. Validation selects kernel parameters/ranks alternatives; neural weights restore minimum validation MSE with patience and cosine decay. The default stays linear Ridge. Test data are for final reporting, not tuning.

### Material-stratified splits

Strata are the **set of active, positive-power target materials**, e.g. `CoO + ZnO`, not measured film composition. Order is ignored only for the split category; source-slot/material/power pairing is unchanged in model inputs. Every material class with at least three independent recipe/library groups appears in train, validation, and test. Integer group quotas aim at 60/20/20 **within each class**, not equal numbers across different classes. No records are oversampled, duplicated, discarded, or assigned using measured target values or model results.

Rare classes cannot be made representative by splitting individual substrate locations: **one independent group stays in training; two groups get training plus one validation/test group**. Rare holdouts are distributed between validation/test according to group-count deficits. Overall fractions differ from 60/20/20 because classes, indivisible group sizes, and unequal numbers of grid points constrain the split. Within each class's fixed group quotas, a seeded search balances point/library counts without looking at property values. This is reproducible stratification, not an optimal allocation guarantee or equal-class training weighting.

Notebook 01 displays the coverage table and saves `outputs/htem_prepared/material_split_balance.csv` with per-class independent-group, library, and point counts. The prepared manifest records the rare-class policy and coverage. Singleton/two-group classes do **not** support a complete three-way per-class evaluation. The current expansion freezes the previous material-stratified assignments and allocates only new groups around them. The loader rejects older prepared manifests; expanded-test scores must not be compared directly with previous smaller-test scores.

## Graph connectivity on each deposition's grid

`examples/02c_HTEM_Grid_GNN.ipynb` uses the node-regression GNN in `htem_graph.py`, using the existing PyTorch dependency (no PyTorch Geometric installation). The current eligible depositions have 11-by-4 grids, but the implementation derives topology from coordinates and supports other grid dimensions; it never reshapes by position number.

- Each **deposition is a separate graph**, with one node at each known planned substrate location. Node inputs are the same train-encoded recipe settings, absolute XY, and local-temperature profile used by the other models.
- Edges join adjacent horizontal/vertical coordinate levels in both directions. No diagonals, cross-deposition edges, graph-level property pooling, or measured-label smoothing are used. Edge inputs are signed dx, signed dy, and distance, divided by a fixed 10 mm reference length.
- All planned locations remain nodes even without usable thickness/conductivity labels. Only eligible labels contribute to MSE/metrics; the loss mask is **not a feature**. This prevents label-dependent neighborhoods and keeps full-grid inference consistent with evaluation.
- Two residual message-passing layers (width 64) learn directional neighbor messages and aggregate their mean. A 64-to-32-to-2 head predicts local thickness/conductivity in the existing standardized log-target space. AdamW, cosine decay, gradient clipping, validation patience, and restoration of minimum-validation-loss weights follow the neural training template. This is a custom edge-aware network inspired by [GraphSAGE's neighbor aggregation](https://arxiv.org/abs/1706.02216), not a paper reproduction. The implementation uses [PyTorch indexed accumulation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html).
- Batched graphs form a disjoint union; edge offsets cannot connect different libraries. The exact existing material-stratified splits, preprocessing, labeled rows, and point-wise loss weighting are reused. Neighbor connectivity is an inductive bias for spatial patterns, not evidence of a causal interaction between sampling sites or a guarantee of better accuracy.

The connectivity figure uses a training deposition. Each neural notebook shows its own training/validation loss and learning-rate curves; Ridge/KRR do not invent neural epoch curves. The GNN has its own saved model, held-out maps, per-point predictions, and test metrics regardless of its rank in 03. Do not select architectures using test results.

The 02c notebook's `predict_substrate(recipe, grid, spacing_mm=None)` always uses its own saved graph model; default 02 instead uses its own linear model. Supply the **full planned grid**, not only locations with labels: changing the node set changes messages. The grid is a query template available before deposition, not a measured-property input. Axis spacing is inferred from the smallest observed coordinate-level interval; gaps greater than 1.05 times it are not bridged. If entire levels are absent and nominal spacing is ambiguous, explicitly supply `spacing_mm=(dx, dy)`. Irregular/jittered coordinates require an explicitly validated grid definition; there is no silent nearest-neighbor fallback or coordinate snapping.

The graph experiment does not change acquisition or preparation artifacts, so existing material splits stay fixed. Rerun 02c after changing the graph helper, and all five algorithm notebooks after changing shared preparation/evaluation. As with other local artifacts, load only trusted joblib files.

## Artifacts

- `outputs/htem_acquisition/`: compact points, **all-recipe catalog**, provenance/coverage, compact-only resumable caches.
- `outputs/htem_prepared/`: row audit, clean spatial table, full unfiltered grid catalog, split manifest with identity/XY/material class, material balance report, fixed arrays, train-fitted preprocessing, checksums.
- `outputs/htem_architectures/{linear_ridge,kernel_ridge,mlp,grid_gnn,random_forest}/`: independent `experiment.json` completion/provenance, `model.joblib`, metrics, per-point predictions, parity/map figures and CSVs. KRR adds `validation_search.csv`; neural models add `loss_history.csv`/training figures; the GNN adds graph summary/connectivity. Model bundles include the shared train-fitted preprocessing.
- `outputs/htem_architectures/comparison/`: `architecture_summary.csv`, comparison error figure, `test_set_parity.png`, and `original_test_before_after.csv` produced by 03.
- `outputs/htem_emulator/`: historical results from the superseded combined notebook, not consumed by 03. That populated notebook is recoverable locally as `archive/02_HTEM_Combined_Algorithms_2026_10_20.ipynb`.

Outputs remain generated/ignored; executed notebooks preserve visible figures/tables. Compact distribution snapshots are under `data/`. Superseded full-spectrum course inputs and old full-response caches are recoverable locally under ignored `archive/`, not the active dataset.

## Limits and verification

This is an observational spatial surrogate, not a validated causal simulator. Missing history, initial-vs-actual temperature trajectories, unrecorded geometry/annealing, sparse chemistry, coordinate-frame transfer, and cohort selection limit prediction. Adding XY need not improve held-out scores. Never feed measured outputs into inputs to make scores look better.

### Executed DOS feature enrichment: 2026-10-21

Notebook 01, all five model notebooks, and notebook 03 were executed with populated outputs and
no failed cells. Preparation fingerprint is `b25c2a5f55f7`; **36,762 positions, 859 depositions,
821 independent groups**, the entire split manifest, every label/mask/identity/XY array and all
136 original encoded feature values were asserted exactly unchanged. Only reference DOS features
and their missing indicators were added (223 total inputs in that earlier run). No HTEM acquisition was repeated.
Pre-DOS preparation, models, populated notebooks and source are recoverable locally under
`archive/pre_source_dos_2026_10_21/`. The table compares the same expanded held-out labels:

| Algorithm | Validation score: before → DOS | Test thickness MAE (µm): before → DOS | Test conductivity log-MAE: before → DOS |
| --- | ---: | ---: | ---: |
| Linear Ridge | 0.73606 → 0.75349 | 0.23339 → 0.33865 | 1.54161 → 1.33508 |
| Exact RBF kernel Ridge | 0.66275 → 0.65678 | 0.16937 → 0.15996 | 1.38070 → 1.29993 |
| Feed-forward MLP | 0.68848 → 0.67482 | 0.18339 → 0.17481 | 1.38001 → 1.35050 |
| Grid GNN | 0.68601 → 0.65917 | 0.17117 → 0.16785 | 1.39404 → 1.43156 |
| Random Forest | 0.65886 → 0.66593 | 0.17570 → 0.17712 | 1.33019 → 1.33859 |

**Honest outcome:** extra chemistry did not help every model or property. KRR/MLP improved both
reported test MAEs; the GNN's conductivity log-error worsened despite better validation and thickness.
Ridge's thickness extrapolation worsened substantially (test thickness R² about -7.75), and forests
worsened slightly on these metrics. KRR now has the lowest validation score; default 02 remains linear.
Conductivity errors are still roughly 1.3–1.4 decades, and DOS features do not supply missing source
placement or solve the spatial-gradient problem. These are single-seed results, not significance tests.

Training/selection rules stayed fixed and never used test feedback. KRR retains nine candidates and
gamma scaling by input count, so its numerical gamma grid changed with 136 → 223 inputs; this is not
a claim to hold every numeric kernel parameter constant. The MLP restored epoch 1/stopped at 41;
the grid GNN restored epoch 24/stopped at 64. Minimum-validation checkpoints, not final weights, are used.

All **110 repository tests** passed, including nine new DOS tests. Environment imports, 24-notebook
structural validation, automation contract, fingerprint/checksum guards, and whitespace checks passed.
All five saved-model full-grid reloads and point-model single/batch/shuffled-query checks passed.
Generated models, reference archive and results remain ignored; compact references and populated
notebooks carry the workshop materials without shipping the large caches/weights.

### Random Forest addition before DOS features: 2026-10-21

`examples/02d_HTEM_Random_Forest.ipynb` and notebook 03 have populated outputs. All five model
notebooks were rerun against the unchanged prepared-data fingerprint `899df0a489f1` and refreshed
evaluation-code fingerprints. Random Forest fitted two forests in 27.7 seconds, using the fixed
configuration above and only each property's observed training labels. No test-driven tuning occurred.

Its validation score is **0.65886**, versus KRR's 0.66275: the best score in this single comparison,
not evidence of statistical superiority. Expanded-test thickness MAE is **0.17571 µm** (KRR: 0.16937),
and conductivity log-MAE is **1.33019 decades** (KRR: 1.38070). Conductivity errors remain large.
The displayed deposition 6671 still exposes a missed thickness gradient: actual 0.573–1.903 µm,
Random Forest 0.152–0.178 µm. Adding trees has not solved the weak spatial response.

Notebook 03 now compares all five algorithms, with test parity for both properties and no fabricated
pre-expansion Random Forest score. Linear Ridge remains the default. All six model/comparison notebooks
executed without failed cells; all saved results share the same point identities, targets, splits,
and preprocessing. Random Forest reload and all 44 single-point/batch/shuffled inference checks passed.
Environment imports, **101 repository tests**, **24-notebook structural validation**, the automation
contract, and whitespace checks passed. Generated models/results remain local and ignored; notebook
figures and tables are preserved.

### Executed expanded-data comparison before adding Random Forest: 2026-10-21

All seven active HTEM notebooks (00, 01, four algorithms, 03) were executed with populated outputs and
no failed cells. Acquisition refreshed compact scalar caches for local temperature without retaining
spectra; the same three source failures remain explicitly reported. Training grew from 155 to **494
independent recipe/library groups**, using 136 encoded features. No original library assignments or
paired source labels/positions changed. All four saved-model reload/inference checks passed.

The comparison below uses exactly the **original 2,108 paired test points**, not the expanded test set:

| Algorithm | Thickness MAE, previous → expanded (µm) | Conductivity log-MAE, previous → expanded (decades) |
| --- | ---: | ---: |
| Linear Ridge | 0.18463 → 0.22787 | 1.77876 → 1.56402 |
| Exact RBF kernel Ridge | 0.17667 → 0.17497 | 1.35985 → 1.36132 |
| Feed-forward MLP | 0.17569 → 0.19630 | 1.48470 → 1.34709 |
| Grid GNN | 0.17497 → 0.17652 | 1.46953 → 1.35895 |

**Honest outcome:** more data and local temperature did not improve every property/model. MLP/GNN
conductivity errors fell, but thickness was worse or essentially unchanged; KRR changed little.
Conductivity errors remain large. This combined change is not an ablation separating temperature
from dataset size, and no architecture was tuned on test results.

On the expanded test set (7,182 observed thickness labels, 2,289 conductivity labels), KRR has thickness
MAE 0.16937 µm and conductivity log-MAE 1.38070 decades. KRR has the best **current validation** score,
selecting alpha 1 and gamma 0.00735294 from nine candidates; full-kernel search took 428 seconds locally.
MLP restored epoch 4/stopped at 44; GNN restored epoch 35/stopped at 75. Neither is claimed fully optimized.
The workshop default remains linear, which still has poor conductivity extrapolation (expanded-test
raw-unit R² about −650); finite positive predictions are not sufficient evidence of useful accuracy.

Verification: **98 repository tests passed**, as did environment imports, 23-notebook structural validation,
the automation contract, comparison fingerprints, original-holdout preservation, model reloads, and
whitespace checks. Generated output/model files remain local and ignored; visible notebook outputs
and compact snapshots are available in the workshop folder.

### Tracing nearly horizontal thickness bands

These are distinct substrate positions, not averaged deposition targets. The same held-out depositions
appear across algorithms by design; no deposition appears in multiple train/validation/test splits.
Examples from the **current test set**, with all ranges in µm:

| Library | Active sources | Actual thickness | Linear prediction | KRR prediction | MLP prediction | GNN prediction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8317 | Ga2O3 + ZnO | 0.384–1.904 | 0.247–0.251 | 0.210–0.266 | 0.184–0.251 | 0.135–0.192 |
| 6812 | Cu2S + Sb2S3 | 0.337–2.534 | 1.436–1.462 | 0.937–1.373 | 1.351–1.552 | 0.860–1.135 |
| 6671 | Cu2O + ZnO | 0.573–1.903 | 0.573–0.633 | 0.512–0.687 | 0.588–0.749 | 0.336–0.534 |
| 8862 | Cu2Se + Ga2Se3 + In2Se3 | 1.972–3.594 | 2.801–2.852 | 1.439–1.687 | 2.353–2.680 | 2.440–2.832 |

Each row represents 44 positions in one deposition. Values were checked against the cached scalar API
arrays by position, not merely by array order. Recipe inputs are constant across a substrate; only XY
and, for nonuniform profiles, local temperature vary. The linear model has shared global XY coefficients,
not recipe-by-position interactions. Opposite observed gradients (e.g. thickness decreases with X in
library 6663 but increases in 8325) cannot be represented by that baseline; its fitted XY effects are small.
Nonlinear models can represent interactions, but the present fits still underrepresent spatial variation.
Graph connectivity alone does not supply source placement or calibrated spatial deposition rates.
Those are plausible missing process descriptors, not established causes for every bad observation.

Not every wide range is a smooth gradient. In **training library 6895** (Cu/Sn), position 28 has thickness
3.277 µm, while the other 43 positions are 0.3062–0.5323 µm; its immediate left/right neighbors are
0.3188/0.3850 µm. This is a source-data anomaly worth investigating, not proof of a decimal/unit error.
In contrast, **validation library 6672** has a coherent 0.05487–1.303 µm gradient; linear predictions are
only 0.3207–0.3266 µm. These are different problems: suspicious labels versus inadequate spatial response.
No labels or holdout membership were changed during this diagnosis. Do not repair or remove test points
because a model misses them, or infer planned source geometry from held-out measured maps.

### Executed acquisition: 2026-10-19 (local date)

All **1,891** public library recipes were retained. The live crawl recovered **82,644** point records from **1,879** libraries; the point Parquet is **950,264 bytes**, and the all-recipe catalog is **48,948 bytes**. Nine libraries contain only null sample-ID placeholders. Three libraries (10941, 12922, 12930) returned errors even for their individual sample fallbacks: **132 declared points remain unavailable**, explicitly recorded in provenance, so full sample coverage is **false**. Another library's 44 recovered scalar points have no verified substrate XY; they remain in the raw table but are excluded from spatial training. No coordinates or labels were invented.

### Executed material-stratified preparation/training: 2026-10-20

Preparation retained **9,482** complete spatial thickness/conductivity records from **275** libraries, with **262** connected recipe/library groups. Material-stratified splits: **5,465 train / 1,909 validation / 2,108 test** points, with **120** training-encoded chemistry/process inputs (previous one-hot version: 123). Group counts are **155 / 53 / 54**, and library counts **164 / 54 / 57**. These are not 9,482 independent depositions.

There are **62** active-target-material combinations. **All 27** classes with at least three independent groups occur in every split. The **16** singleton classes stay in training; the **19** two-group classes have training plus one holdout. No feasible class is missing from any split, and no substrate or repeated recipe was broken up to achieve this. The full per-class counts are displayed in notebook 01 and saved in the material balance CSV.

### Previous one-hot independent algorithm comparison: 2026-10-20

These are the **pre-chemistry** results, not the current descriptor runs. The populated notebooks, preparation, and outputs are retained locally under `archive/one_hot_architectures_2026_10_20/`. All four model notebooks and notebook 03 were executed on the same original preparation (`a1eb363d157f…`). Notebook 02 remains linear Ridge, not an automatically selected algorithm.

| Algorithm | Validation mean standardized RMSE ↓ | Test thickness MAE (µm) ↓ | Test conductivity log-MAE (decades) ↓ |
| --- | ---: | ---: | ---: |
| Linear Ridge (02 default) | 0.64184 | 0.18625 | 1.47656 |
| RBF kernel ridge (02a) | 0.66930 | 0.17112 | 1.52318 |
| Feed-forward MLP (02b) | 0.65613 | 0.17948 | 1.47326 |
| Grid GNN (02c) | 0.64465 | 0.18698 | 1.53586 |

KRR selected **alpha 0.1, gamma 0.00081300813** from nine validation-only candidates. It did **not** beat linear Ridge on validation; its test thickness MAE improved by about 8%, while conductivity log-MAE worsened. This is a property trade-off, not a universal improvement or a reason to tune on test. Conductivity error remains roughly 1.5 decades: these are limited observational emulators, not deployment-ready predictors. The same held-out map (library 6671) is selected by point count, not favorable error, and may expose especially poor predictions.

The MLP restored epoch **5**, stopping at **45**; the GNN restored epoch **39**, stopping at **79**, both by validation patience. The 275 deposition graphs contain **12,100 nodes**, **20,075 undirected edges**, and the same **9,482 labeled locations**; 2,618 nodes are unlabeled geometry-only context. Every current grid has 44 nodes and 73 undirected edges, but neither count is hard-coded. All four models retain their own bundles, metrics, maps, and recipe inference functions regardless of ranking.

All five model/comparison notebooks have populated outputs with no failed cells. Reloaded recipe/grid inference reproduces saved full-grid predictions; full-grid predictions agree with labeled-point evaluation. The real-data smoke test caught float32 batch-dependent roundoff in exact KRR, resolved by using float64 for its solver without altering the prepared values. All **82** repository tests passed, including 11 new comparison tests for validation-only exact KRR, batch invariance, unchanged data/point identity, independent/reloaded models, and refusal to compare stale or altered results. The existing graph, material-stratification, acquisition, input-leakage, train-only preprocessing, and IR tests remain intact. Notebook structure (23 notebooks), environment imports, automation contract, and whitespace checks also passed.

### Executed source-chemistry comparison: 2026-10-20

Notebook 01, all four model notebooks, and notebook 03 were rerun with populated outputs and no failed cells. All **48/48** clean-cohort source labels parse; **27/48** have independent band-gap references. Source-name one-hot columns are absent. The same 120 encoded features are used by every model. Before training, the old/new full split manifests and every split's **labels, sample/library IDs, positions, and XY arrays were asserted exactly equal**; no HTEM data were downloaded again and no holdouts were moved. All four reload/inference checks and the strict comparison fingerprints passed. The prior preparation/models/notebooks remain recoverable in the local archive named above.

| Algorithm | Validation mean standardized RMSE ↓ | Test thickness MAE (µm) ↓ | Test conductivity log-MAE (decades) ↓ |
| --- | ---: | ---: | ---: |
| Linear Ridge (02 default) | 0.69260 | 0.18463 | 1.77876 |
| RBF kernel ridge (02a) | 0.65332 | 0.17667 | 1.35985 |
| Feed-forward MLP (02b) | 0.67654 | 0.17569 | 1.48470 |
| Grid GNN (02c) | 0.64156 | 0.17497 | 1.46953 |

**Honest outcome:** descriptors improved KRR validation (0.66930 → 0.65332) and GNN validation (0.64465 → 0.64156), but worsened the fixed linear and MLP validation scores. KRR's conductivity log-MAE improved (1.52318 → 1.35985), at slightly worse thickness MAE. The linear model has severe conductivity extrapolation: its test raw-unit R² is approximately −6.28×10^17, despite finite positive predictions. Do not interpret this default teaching baseline as reliable inference. No predictions were clipped, bad examples removed, or test-driven hyperparameters adjusted to hide this failure. The best current validation model is the GNN, but 02 deliberately remains linear. The previous best overall validation score (linear, 0.64184) and current best (GNN, 0.64156) are effectively similar on one split/seed, not evidence of a significant overall gain.

KRR now selects alpha 1 and gamma 0.0083333 from the same nine-choice search rule; gamma scales with encoded feature count. MLP restores epoch 3/stops at 43; GNN restores epoch 28/stops at 68. Training settings otherwise remain unchanged. All **91** repository tests pass, including nine chemistry-specific tests for formula parsing/stoichiometry, atom-weighted properties, reference provenance, missing gaps, source/power pairing, original recipe identity, absence of source one-hot columns, and stale-reference rejection. No new dependencies were added. Notebook/automation validation and whitespace checks also pass.

Sources: [HTEM paper](https://doi.org/10.1038/sdata.2018.53), [current HTEM site/API documentation](https://htem.nlr.gov/), [official historical examples](https://github.com/NatLabRockies/htem-api-examples). Fast checks:

```powershell
uv run --locked python -m unittest discover -s tests -p "test_htem*.py" -v
uv run --locked python scripts/ci/validate_notebooks.py
```
