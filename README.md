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

1. `00_IR_Get_Data_NISTChemPy_v2.ipynb` — pre-work NIST acquisition of native IR spectra, SMILES, and functional-group labels.
2. `01_IR_Vibe_Coding_Data_Carpentry.ipynb` — simple baseline for prompt-driven data examination and carpentry.
3. `02_IR_Optimized_Training.ipynb` — static neural-network classifier for comparing prepared datasets on fixed evaluation splits.
4. `03_IR_Compare_Carpentry.ipynb` — comparison of the baseline and worked carpentry results.

See [worked examples 01a–01d](02_ir_functional_groups/examples/README.md) for alternative cleansing strategies and run instructions. The native IR checkpoint is required; normalized rehearsal data are not a substitute.

### 03_htem_conductivity

1. `00_HTEM_Get_Data_NREL.ipynb` — HTEM acquisition of compact process conditions, substrate positions, and scalar properties without full spectra.
2. `01_HTEM_Process_Data_Carpentry.ipynb` — completed preparation of shared chemistry/process features and grouped train/validation/test splits.
3. `02_HTEM_Process_Emulator_Vibe_Coding.ipynb` — prompt-guided model experiments, starting with linear Ridge regression.
4. `examples/02a–02d` — kernel Ridge, feed-forward network, grid GNN, and Random Forest alternatives using the same prepared data.
5. `03_HTEM_Compare_Architectures.ipynb` — comparison of model errors, with test-set parity plots.

Predict thickness and conductivity at each substrate location from process settings and XY coordinates—not measured film properties or spectra. See the [HTEM module guide](03_htem_conductivity/README.md) for examples, spatial maps, and run instructions.

## Setup

This is the workshop-content repository. Participants use UV and VS Code, then choose either Codex in VS Code or the separate containerized local-LLM agent workflow. See [SETUP_GUIDE.md](SETUP_GUIDE.md).

After installing UV, opening this folder in VS Code, and opening its integrated terminal, verify the locked environment:

```bash
uv run --locked python test_environment.py
```

The project pins Python 3.11 in `.python-version`, declares all dependencies in `pyproject.toml`, and commits the tested `uv.lock`. The virtual environment is created locally as `.venv` and is ignored by Git. Open the notebooks in VS Code and select the Python interpreter inside `.venv` as the notebook kernel.

Run `00_Test_Installation.ipynb` or the command-line verifier before the workshop. The first run of `01_boiling_point/03_BP_2_Foundation_Model.ipynb` downloads the public MoLFormer checkpoint (about 188 MB); rehearse it once before the workshop or ensure the Hugging Face cache travels with the workshop computers.

## Notes for presenters

Each code cell now has short **What to notice** bullet points immediately above it, including
the worked alternatives and the instructor feature-selection study. Blank practice cells
are labeled as exercises, not completed work. Existing code and saved outputs are retained.

For an explanations-and-results view in **VS Code**, open the Command Palette and run
**Notebook: Collapse All Cell Inputs** (search `Collapse All Cell Inputs`). Run
**Expand All Cell Inputs** to edit code again, or expand just the cell you want. Outputs
remain visible. VS Code may restore a previous editor view, so use this command even if
the notebook's input-collapse metadata is already set. JupyterLab uses the saved
`jupyter.source_hidden` hints and also provides a left-hand cell collapser. GitHub's
static notebook preview may still show code; hiding inputs is a reader view, not redaction.
See [the notebook presentation guide](docs/NOTEBOOK_PRESENTATION.md).

- `01_boiling_point/02_BP_1_Chemprop_GNN.ipynb` is a 1,500-molecule, 20-epoch demonstration. Increase either setting only after a rehearsal on the actual workshop hardware.
- The boiling-point examples use random molecular splits for speed and conceptual clarity. Do not describe their test metrics as estimates of performance on new chemical scaffolds; use a scaffold split for that claim.
- Run each acquisition notebook before the workshop and retain its output. The IR scraper is intentionally a pre-work step; the workshop starts at the carpentry notebook with its downloaded data. The static IR trainer logs TensorBoard events, saves checkpoints every 25 epochs, restores the lowest eligible validation-loss weights, and writes test metrics and a summary plot. Its 2,000-epoch cap is a resource limit; run documentation records whether the robust stopping rule actually triggered.

## Maintainer automation

Trusted repository collaborators with write access can include `@codex` in an Issue or Issue comment to use the hosted Codex workshop runner. It creates or reuses `dev/<their-GitHub-username>`, can set up the UV environment, access the documented public data/model sources, execute requested notebooks, and commits the resulting code, populated notebooks, and useful output artifacts to that branch. It does not push to `main` or merge branches. See [docs/GITHUB_CODEX_AGENT.md](docs/GITHUB_CODEX_AGENT.md) for operation and troubleshooting.
