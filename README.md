# AVS workshop code

This repository is a concise workshop adaptation of the course material. The original course files are not required and are not included in the GitHub distribution. It preserves the three planned examples:

1. molecular structure to boiling point (regression; Chemprop GNN and a frozen foundation-model comparison);
2. IR spectrum to functional-group presence (multi-label classification; data carpentry focus); and
3. HTEM deposition recipes and substrate XY to local thickness/conductivity maps (spatial process emulation; algorithm/architecture experimentation).

Start with `00_Workshop_Roadmap.ipynb`, then work within one numbered example folder at a time. Each sequence begins with executable acquisition code. The original course files are not modified; compact prepared inputs are retained only as an offline/rehearsal fallback.

## Notebook order

### 01_boiling_point

1. `00_BP_Get_Data_NISTChemPy_v2.ipynb` — NIST acquisition of a new table containing SMILES and boiling point.
2. `02_BP_1_Chemprop_GNN.ipynb` — GNN regression demo.
3. `03_BP_2_Foundation_Model.ipynb` — frozen foundation-model comparison.
4. `04_BP_3_One_Shot_Vibe_Coding.ipynb` — prompt-driven baseline.

### 02_ir_functional_groups

1. `00_IR_Get_Data_NISTChemPy_v2.ipynb` — pre-work acquisition only. It saves native `x_raw`/`y_raw`, signal/coordinate units, SMILES, and labels in `outputs/nist_acquisition/ir_raw_checkpoint.parquet`. Existing checkpoints are reused; live collection is off by default. It no longer filters signal types, normalizes, crops, pads or regrids.
2. `01_IR_Vibe_Coding_Data_Carpentry.ipynb` — a deliberately naïve numeric baseline: use recorded arrays as-is, trim to complete reference coverage, linearly resize and normalize. It does not distinguish signal types, perform advanced cleansing or deliberately corrupt observations. Its actual score is not guaranteed to be worse.
   Four worked alternatives [01a–01d in examples/](02_ir_functional_groups/examples/README.md) explicitly omit or convert transmittance, convert coordinate units, trim/pad, downsample/upsample, and normalize. They also sample canonical deduplication, categorical-label consistency, statistical quality filtering and training-only balancing. Each ends with the same finite 500-point feature structure and 12 binary targets. Validation/test are frozen under `data/frozen_evaluation/`; keep those reference files in Git. Held-out molecular observations are excluded before carpentry, including for the baseline. The original heldouts are preserved byte-for-byte. The native checkpoint is required: normalized rehearsal data are not silently substituted.
3. `02_IR_Optimized_Training.ipynb` — a static multi-label classifier using AdamW at initial/maximum LR `1e-5`, cosine annealing with warm restarts, and the boiling-point module's robust 20-epoch slope stopping rule (including removal of the five highest validation losses). Change only `CARPENTRY_DIR` to compare carpentry choices; validation-selected weights, loss/LR plots and periodic checkpoints are saved for each run.
4. `03_IR_Compare_Carpentry.ipynb` — a short results table and comparison figure for the baseline and four examples, using current compatible IR 02 runs. Reports actual gains/regressions and epochs/stopping reasons, not a guaranteed benefit or causal attribution to individual bundled steps. For the full IR preparation/training/comparison sequence run `uv run --locked python 02_ir_functional_groups/run_carpentry_comparison.py --prepare`. IR 02 rejects modified heldouts or a changed reference grid. Superseded notebooks are recoverable in the ignored `examples/archive/before_consolidation/`; old artifacts are not selected by the new active strategy list.

### 03_htem_conductivity

1. `00_HTEM_Get_Data_NREL.ipynb` — defaults to live acquisition of **all** public libraries, with compact resumable recipe/scalar caches. The scalar batch endpoint provides aligned substrate XY and labels without full spectra. All recipes and incomplete-label records are retained; failures and sample-coverage gaps are reported. Explicit `ACQUISITION_MODE='snapshot'` uses the distributed compact live-acquisition snapshot offline.
2. `01_HTEM_Process_Data_Carpentry.ipynb` — completed instructor preparation: preserve `(library, position)` and substrate XY, audit missing/invalid data, keep source/power and gas/flow pairing, stratify whole library/identical-recipe groups by active target-material combination with explicit rare-class exceptions, then train-fit encoding/imputation/scaling. Chemical features are selected beforehand by an instructor-only grouped training/validation comparison of curated descriptors, pretrained MEGNet composition vectors, real ALIGNN reference-crystal embeddings and reference DOS. The selected specification and exact frozen matrices are distributed under `03_htem_conductivity/data/`; all five architectures share them unchanged. Reference crystals/DOS do not establish actual source/film phases. Measured HTEM film gaps/spectra are never inputs. Unknown XY is excluded, not fabricated from stage coordinates.
3. `02_HTEM_Process_Emulator_Vibe_Coding.ipynb` — the default **linear Ridge regression** process emulator, with prompt-guided experiments, per-point evaluation, parity/substrate maps, and saved-model recipe inference. It remains the default even if an alternative scores better.
4. `examples/02a_HTEM_Kernel_Ridge.ipynb`, `02b_HTEM_Feed_Forward_Network.ipynb`, `02c_HTEM_Grid_GNN.ipynb`, `02d_HTEM_Random_Forest.ipynb` — independently run exact RBF kernel ridge, a 64–32 MLP, the grid-connectivity GNN, and Random Forest on **the same prepared features, transforms, labels, and material-balanced splits**. Neural models retain cosine decay and validation-minimum checkpoints. Forests fit each observed property separately using 300 trees, bounded depth, and a fixed seed. Each model writes separate outputs.
5. `03_HTEM_Compare_Architectures.ipynb` — a short comparison table/error plot; checks data/preprocessing/result fingerprints and identical labeled locations before comparing all five models. It does not train or alter preparation. Run 01 once, then all five model notebooks and 03.

Inputs are recipe settings plus requested substrate `x_mm`/`y_mm` in the scalar API's substrate frame—never spectra, measured composition/properties, quality, IDs, or instrument `xyz_mm` stage/focus coordinates. Thickness (µm) and conductivity (S/cm) are learned in log space. Optional bandgap remains in the raw download but does not restrict the two-output mapping cohort. Grids are not assumed to be 4×11, missing cells stay blank, and maps never pool multiple depositions. See [the HTEM module guide](03_htem_conductivity/README.md) for coverage, limitations, and outputs. Superseded spectral inputs/notebooks are recoverable locally under ignored `archive/`, not in the active sequence.

## Setup

HTEM preparation now uses **either observed output**, with per-target label masks and the sample-level
temperature profile. The eligible pool is 859 depositions / 821 recipe-library groups (494 training
groups). The original split assignments and paired test observations remain frozen. Notebook 03
separately reports expanded-test results and before/after errors on the original 2,108 paired test points.
See the HTEM module guide for memory requirements of the larger exact-kernel example.

This is the workshop-content repository. Participants use UV and VS Code, then choose either Codex in VS Code or the separate containerized local-LLM agent workflow. See [SETUP_GUIDE.md](SETUP_GUIDE.md).

After installing UV, opening this folder in VS Code, and opening its integrated terminal, verify the locked environment:

```bash
uv run --locked python test_environment.py
```

The project pins Python 3.11 in `.python-version`, declares all dependencies in `pyproject.toml`, and commits the tested `uv.lock`. The virtual environment is created locally as `.venv` and is ignored by Git. Open the notebooks in VS Code and select the Python interpreter inside `.venv` as the notebook kernel.

Run `00_Test_Installation.ipynb` or the command-line verifier before the workshop. The first run of `01_boiling_point/03_BP_2_Foundation_Model.ipynb` downloads the public MoLFormer checkpoint (about 188 MB); rehearse it once before the workshop or ensure the Hugging Face cache travels with the workshop computers.

## Notes for presenters

- `01_boiling_point/02_BP_1_Chemprop_GNN.ipynb` is a 1,500-molecule, 20-epoch demonstration. Increase either setting only after a rehearsal on the actual workshop hardware.
- The boiling-point examples use random molecular splits for speed and conceptual clarity. Do not describe their test metrics as estimates of performance on new chemical scaffolds; use a scaffold split for that claim.
- Run each acquisition notebook before the workshop and retain its output. The IR scraper is intentionally a pre-work step; the workshop starts at the carpentry notebook with its downloaded data. The static IR trainer logs TensorBoard events, saves checkpoints every 25 epochs, restores the lowest eligible validation-loss weights, and writes test metrics and a summary plot. Its 2,000-epoch cap is a resource limit; run documentation records whether the robust stopping rule actually triggered.

## Maintainer automation

Trusted repository collaborators with write access can include `@codex` in an Issue or Issue comment to use the hosted Codex workshop runner. It creates or reuses `dev/<their-GitHub-username>`, can set up the UV environment, access the documented public data/model sources, execute requested notebooks, and commits the resulting code, populated notebooks, and useful output artifacts to that branch. It does not push to `main` or merge branches. See [docs/GITHUB_CODEX_AGENT.md](docs/GITHUB_CODEX_AGENT.md) for operation and troubleshooting.
