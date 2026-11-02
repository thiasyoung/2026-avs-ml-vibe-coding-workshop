# AVS workshop code

This repository is a concise workshop for vibe coding machine learning pipelines, consisting of three examples:

1. molecular structure to boiling point (regression; Chemprop GNN and a frozen foundation-model comparison);
2. IR spectrum to functional-group presence (multi-label classification; data carpentry focus); and
3. HTEM deposition recipes and substrate XY to local thickness/conductivity maps (spatial process emulation; algorithm/architecture experimentation).

Start with `00_Workshop_Roadmap.ipynb`, then work within one numbered example folder at a time. Each sequence begins with executable acquisition code.

## Read the workshop as a website

The Quarto reading view provides searchable navigation, explanations and saved plots,
with Python code folded until you expand it. Keep using the original notebooks in
VS Code to run or edit the examples. Website publishing never reruns the notebooks.
See [preview and GitHub Pages setup](docs/QUARTO_WEBSITE.md); the reading site is
available only after Pages is activated and its first deployment succeeds.
In **Settings → Pages**, set **Source: GitHub Actions**, not **Deploy from a branch**.
The branch-based publisher shows the README and raw notebook links instead of the
Quarto tutorial pages. Run **Actions → Publish workshop website** after choosing
the correct source.

## Notebook order

Notebook numbering restarts at `00` in each module. BP runs from `00` to `04`;
IR and HTEM run from `00` to `03`. Lettered examples such as `01a` or `02a`
are alternatives to the corresponding numbered notebook, not additional main steps.

### 01_boiling_point

- [00 — Get data](01_boiling_point/00_BP_Get_Data_NISTChemPy_v2.ipynb) — NIST acquisition of a new table containing SMILES and boiling point.
- [01 — Chemprop GNN](01_boiling_point/01_BP_Chemprop_GNN.ipynb) — GNN regression demo.
- [02 — Foundation model](01_boiling_point/02_BP_Foundation_Model.ipynb) — frozen foundation-model comparison.
- [03 — One-shot prompt](01_boiling_point/03_BP_One_Shot_Vibe_Coding.ipynb) — prompt-driven baseline.
- [04 — Worked one-shot workflow](01_boiling_point/04_BP_One_Shot_Workflow.ipynb) — completed prediction workflow.

### 02_ir_functional_groups

- [00 — Get data](02_ir_functional_groups/00_IR_Get_Data_NISTChemPy_v2.ipynb) — pre-work NIST acquisition of native IR spectra, SMILES, and functional-group labels.
- [01 — Examine and prepare](02_ir_functional_groups/01_IR_Vibe_Coding_Data_Carpentry.ipynb) — simple baseline for prompt-driven data examination and carpentry.
- [02 — Static training](02_ir_functional_groups/02_IR_Optimized_Training.ipynb) — static neural-network classifier for comparing prepared datasets on fixed evaluation splits.
- [03 — Compare results](02_ir_functional_groups/03_IR_Compare_Carpentry.ipynb) — comparison of the baseline and worked carpentry results.

See [worked examples 01a–01d](02_ir_functional_groups/examples/README.md) for example cleansing strategies and run instructions. The native IR checkpoint is required.

### 03_htem_conductivity

- [00 — Get data](03_htem_conductivity/00_HTEM_Get_Data_NREL.ipynb) — HTEM acquisition of compact process conditions, substrate positions, and scalar properties without full spectra.
- [01 — Prepare data](03_htem_conductivity/01_HTEM_Process_Data_Carpentry.ipynb) — completed preparation of shared chemistry/process features and grouped train/validation/test splits.
- [02 — Build an emulator](03_htem_conductivity/02_HTEM_Process_Emulator_Vibe_Coding.ipynb) — prompt-guided model experiments, starting with linear Ridge regression.
- [03 — Compare outcomes](03_htem_conductivity/03_HTEM_Compare_Architectures.ipynb) — comparison of model errors, with test-set parity plots.

Worked alternatives to `02`: `examples/02a–02d` use kernel Ridge, a feed-forward network, a grid GNN, and Random Forest on the same prepared data.

Predict thickness and conductivity at each substrate location from process settings and XY coordinates—not measured film properties or spectra. See the [HTEM module guide](03_htem_conductivity/README.md) for examples, spatial maps, and run instructions.

## Setup

This is the workshop-content repository. Participants use UV and VS Code, then choose either Codex in VS Code or the separate containerized local-LLM agent workflow. See [SETUP_GUIDE.md](SETUP_GUIDE.md).

After installing UV, open this folder in VS Code and open its integrated terminal.

### Verify the workshop environment

Code Summary: Checks that the locked workshop environment is ready to run the notebooks.

```bash
uv run --locked python test_environment.py
```

The project pins Python 3.11 in `.python-version`, declares all dependencies in `pyproject.toml`, and commits the tested `uv.lock`. The virtual environment is created locally as `.venv` and is ignored by Git. Open the notebooks in VS Code and select the Python interpreter inside `.venv` as the notebook kernel.

Run `00_Test_Installation.ipynb` or the command-line verifier before the workshop. The first run of `01_boiling_point/02_BP_Foundation_Model.ipynb` downloads the public MoLFormer checkpoint (about 188 MB); rehearse it once before the workshop or ensure the Hugging Face cache travels with the workshop computers.

## Notes for presenters

### Follow the workflow with code hidden

Code Summary: Each notebook code cell has a descriptive heading and a concise paragraph
beginning with "Code Summary: " immediately above it, explaining what the following cell
does in the workshop. This includes worked examples and the instructor feature study,
so you can follow the workflow with code hidden.
Blank practice cells are identified as exercises, not completed work. Code and saved outputs
are unchanged by this presentation update.

For an explanations-and-results view in **VS Code**, open the Command Palette and run
**Notebook: Collapse All Cell Inputs** (search `Collapse All Cell Inputs`). Run
**Expand All Cell Inputs** to edit code again, or expand just the cell you want. Outputs
remain visible. VS Code may restore a previous editor view, so use this command even if
the notebook's input-collapse metadata is already set. JupyterLab uses the saved
`jupyter.source_hidden` hints and also provides a left-hand cell collapser. GitHub's
static notebook preview may still show code; hiding inputs is a reader view, not redaction.
See [the notebook presentation guide](docs/NOTEBOOK_PRESENTATION.md).

- Run each acquisition notebook before the workshop and retain its output. The IR scraper is intentionally a pre-work step; the workshop starts at the carpentry notebook with its downloaded data. The static IR trainer logs TensorBoard events, saves checkpoints every 25 epochs, restores the lowest eligible validation-loss weights, and writes test metrics and a summary plot. Its 2,000-epoch cap is a resource limit; run documentation records whether the robust stopping rule actually triggered.

## Maintainer automation

Trusted repository collaborators with write access can include `@codex` in an Issue or Issue comment to use the hosted Codex workshop runner. Requests create or reuse the sender's `dev/<username>` branch, never push directly to main, and queue per user while different users can work in parallel. The runner can set up the UV environment, download public data/models, execute notebooks, and commit populated notebooks and useful outputs. It replies on the Issue with results and branch/commit links. Use `@codex sync with main` to update your branch or `@codex prepare merge` to request a pull request; ordinary work needs no PR or manual approval. The runner never force-pushes or automatically merges. See [docs/GITHUB_CODEX_AGENT.md](docs/GITHUB_CODEX_AGENT.md) for activation, limits, and troubleshooting.
