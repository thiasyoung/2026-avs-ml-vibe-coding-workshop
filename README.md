# AVS workshop code

This repository is a concise workshop for vibe coding machine learning pipelines, consisting of three examples:

1. molecular structure to boiling point (prompt-driven regression, with optional completed model references);
2. IR spectrum to functional-group presence (multi-label classification; data carpentry focus); and
3. HTEM deposition recipes and substrate XY to local thickness/conductivity maps (spatial process emulation; algorithm/architecture experimentation).

Start with `00_Workshop_Roadmap.ipynb`, then work within one numbered example folder at a time. Each sequence begins with acquisition pre-work; prepared data lets the live session focus on the workshop exercises.

> [!IMPORTANT]
> **IR module: download the dataset before the workshop.**
> Open [Workshop data v1](https://github.com/thiasyoung/2026-avs-ml-vibe-coding-workshop/releases/tag/workshop-data-v1), expand **Assets**, and download **ir_raw_checkpoint.parquet** (about 196 MB), not a **Source code** archive.
> Save it inside your cloned repository as `02_ir_functional_groups/outputs/nist_acquisition/ir_raw_checkpoint.parquet` (create the folders if needed).
> Cloning or downloading the repository does **not** include this file. Using it avoids repeating the lengthy NIST acquisition.

## Read the workshop as a website

The Quarto reading view uses the repository's existing notebooks and Markdown files,
with searchable navigation, saved plots, and expandable code. Keep editing and running
the original notebooks in VS Code; do not convert them or add HTML beside them.
GitHub Actions builds the web-only HTML separately when publishing. No local render
is needed, and website publishing never reruns the notebooks.
See [preview and GitHub Pages setup](docs/QUARTO_WEBSITE.md); the reading site is
available only after Pages is activated and its first deployment succeeds.
In **Settings → Pages**, set **Source: GitHub Actions**, not **Deploy from a branch**.
The branch-based publisher shows the README and raw notebook links instead of the
Quarto tutorial pages. Run **Actions → Publish workshop website** after choosing
the correct source.

## Notebook order

Notebook numbering restarts at `00` in each module. All three main paths run `00`–`02`.
Completed references live in each module's `examples/` folder. Lettered examples such as `01a` or `02a`
are alternatives to the corresponding numbered notebook, not additional main steps.

### 01_boiling_point

- [00 — Get data](01_boiling_point/00_BP_Get_Data_NISTChemPy_v2.ipynb) — acquisition pre-work; reuse the prepared SMILES/BP data during the session.
- [01 — Ten steps in prompts](01_boiling_point/01_Ten_Steps_in_Prompts.ipynb) — template for the machine-learning workflow.
- [02 — Vibe-coding workbook](01_boiling_point/02_BP_Vibe_Coding_Workbook.ipynb) — copy and adapt the prompt, then have your assistant populate and execute the workbook.

Optional [BP examples](01_boiling_point/examples/README.md) hold Chemprop, MoLFormer,
the one-shot prompt and completed workflow, and both downloadable prompt text files.
These are references, not additional required workshop steps.

### 02_ir_functional_groups

- [00 — Get data](02_ir_functional_groups/00_IR_Get_Data_NISTChemPy_v2.ipynb) — pre-work NIST acquisition of native IR spectra, SMILES, and functional-group labels.
- [01 — Vibe-coding carpentry workbook](02_ir_functional_groups/01_IR_Vibe_Coding_Data_Carpentry.ipynb) — edit one prompt and have your assistant examine, prepare and validate the trainer's input files.
- [02 — Static training](02_ir_functional_groups/02_IR_Optimized_Training.ipynb) — static neural-network classifier for comparing prepared datasets on fixed evaluation splits.

See [IR references](02_ir_functional_groups/examples/README.md) for the completed 01
baseline, strategies 01a–01d and 03 comparison. The native IR checkpoint is required.

### 03_htem_conductivity

- [00 — Get data](03_htem_conductivity/00_HTEM_Get_Data_NREL.ipynb) — HTEM acquisition of compact process conditions, substrate positions, and scalar properties without full spectra.
- [01 — Prepare data](03_htem_conductivity/01_HTEM_Process_Data_Carpentry.ipynb) — completed preparation of shared chemistry/process features and grouped train/validation/test splits.
- [02 — Vibe-coding workbook](03_htem_conductivity/02_HTEM_Vibe_Coding_Workbook.ipynb) — edit one prompt to build, execute and analyze model experiments on the fixed prepared data.

Optional [HTEM references](03_htem_conductivity/examples/README.md) include the completed
linear Ridge workflow, four alternative architectures and the 03 comparison notebook.

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

Run `00_Test_Installation.ipynb` or the command-line verifier before the workshop. The first run of the optional `01_boiling_point/examples/02_BP_Foundation_Model.ipynb` downloads the public MoLFormer checkpoint (about 188 MB); rehearse it once before the workshop or ensure the Hugging Face cache travels with the workshop computers.

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
