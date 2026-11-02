# How to Use AI to Vibe Code and Build a Scientific Machine Learning Pipeline

**2026 AVS Workshop** — use a coding assistant to build a pipeline, investigate data
quality, and test model ideas on three scientific datasets. You will work with
Jupyter notebooks in VS Code; completed examples are available when you want a reference.

## Start here

1. Download or clone the **whole repository**, not individual notebooks: they use
   local helpers and prepared data.
2. Follow the [setup guide](SETUP_GUIDE.md), then run
   [Check your installation](00_Test_Installation.ipynb) using the workshop `.venv` kernel.
3. Read the [workshop roadmap](00_Workshop_Roadmap.ipynb), then follow the main
   notebooks in each module below. The `00` acquisition notebooks are pre-work;
   you do not need to repeat a full scrape during the session.

Prefer to read first? Open the [workshop website](https://thiasyoung.github.io/2026-avs-ml-vibe-coding-workshop/)
for searchable pages with expandable code and saved figures. Run and edit notebooks
locally; the website is not a live Python session.

> [!IMPORTANT]
> **IR module: download the dataset before the workshop.**
> Open [Workshop data v1](https://github.com/thiasyoung/2026-avs-ml-vibe-coding-workshop/releases/tag/workshop-data-v1), expand **Assets**, and download **ir_raw_checkpoint.parquet** (about 196 MB), not a **Source code** archive.
> Save it inside your cloned repository as `02_ir_functional_groups/outputs/nist_acquisition/ir_raw_checkpoint.parquet` (create the folders if needed).
> Cloning or downloading the repository does **not** include this file. Using it avoids repeating the lengthy NIST acquisition.

## Workshop modules

Numbering restarts in each folder. Follow **00 → 01 → 02** for the main path;
`examples/` contains optional references, not extra required steps.

### 01_boiling_point

**Build a pipeline:** predict reported boiling point in kelvin from a molecule's SMILES string.

- [00 — Get data](01_boiling_point/00_BP_Get_Data_NISTChemPy_v2.ipynb) — acquisition and cleansing; use prepared data in the session.
- [01 — Ten steps in prompts](01_boiling_point/01_Ten_Steps_in_Prompts.ipynb) — the workflow template.
- [02 — Vibe-coding workbook](01_boiling_point/02_BP_Vibe_Coding_Workbook.ipynb) — adapt the prompt and ask your assistant to build and execute your pipeline.

[Module guide](01_boiling_point/README.md) · [Optional examples](01_boiling_point/examples/README.md):
Chemprop, frozen MoLFormer, and one-shot prompting/workflow references.

### 02_ir_functional_groups

**Improve the data:** predict functional-group presence from IR spectra while keeping the classifier fixed.

- [00 — Get data](02_ir_functional_groups/00_IR_Get_Data_NISTChemPy_v2.ipynb) — acquire native spectra and labels, or reuse the release checkpoint.
- [01 — Data-carpentry workbook](02_ir_functional_groups/01_IR_Vibe_Coding_Data_Carpentry.ipynb) — investigate your hypothesis and generate uniform training data.
- [02 — Static training](02_ir_functional_groups/02_IR_Optimized_Training.ipynb) — select your prepared dataset and measure its effect with the same model and holdouts.

[Module guide](02_ir_functional_groups/README.md) · [Optional examples](02_ir_functional_groups/examples/README.md):
a minimal baseline, four carpentry strategies, and their results comparison.

### 03_htem_conductivity

**Improve the model:** predict thickness and conductivity at each substrate XY location from process inputs.

- [00 — Get data](03_htem_conductivity/00_HTEM_Get_Data_NREL.ipynb) — compact process and scalar-property records, without full spectra.
- [01 — Prepare data](03_htem_conductivity/01_HTEM_Process_Data_Carpentry.ipynb) — supplied preparation of shared features and grouped splits.
- [02 — Model-experiment workbook](03_htem_conductivity/02_HTEM_Vibe_Coding_Workbook.ipynb) — develop and analyze model experiments using those fixed inputs.

[Module guide](03_htem_conductivity/README.md) · [Optional examples](03_htem_conductivity/examples/README.md):
linear Ridge, kernel Ridge, a feed-forward network, a grid GNN, Random Forest, and comparison.

## Work with the prompts

The BP workbook uses the ten-step template. The IR and HTEM workbooks provide a
starting approach plus blanks for your hypothesis, plan, vibe-coding strategy,
effort priorities and budget. Edit their **EDITABLE** text freely; keep the
**FIXED input/output contracts** intact. Copy the **whole prompt** into your coding
assistant. Running a prompt-only notebook does not call an AI model.

Ask the assistant to populate the workbook or create a separate notebook. Inspect
its code, run its checks and save genuine results. Use validation data for decisions;
reserve test data for final evaluation. More elaborate preparation or a larger model
is not guaranteed to improve predictions.

## Find your files

- Module roots contain the main notebooks and their guide.
- `examples/` contains optional worked notebooks, prompt text, and example-running scripts.
- `lib/` in IR and HTEM contains reusable helpers; notebook setup cells locate them automatically.
- `data/` contains supplied inputs and frozen references; `outputs/` holds generated runs.
  IR reference carpentry datasets also use `examples/outputs/`.
- Your exercises use `outputs/bp_workbook/`, `outputs/workshop_carpentry/` (IR), or
  `outputs/htem_workbook/<unique_experiment_id>/` (HTEM), separate from reference runs.

Saved notebook figures can be read without retraining. Model weights and full
run folders are not all distributed: regenerating a comparison may require running
its examples. Keep the complete folder layout when downloading the materials.

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

## Read explanations and outputs with code folded

Each code cell has a descriptive heading and a short **Code Summary:** explaining
its purpose. In VS Code, open the Command Palette (`Ctrl+Shift+P` on Windows/Linux,
`Cmd+Shift+P` on macOS) and run **Notebook: Collapse All Cell Inputs**. Leave outputs
expanded; use **Expand All Cell Inputs** or expand one cell when you want to edit.
VS Code may restore a previous view. JupyterLab uses the saved input-collapse hints
and its left-hand cell collapser; GitHub's notebook preview may still show code.
See [reading and notebook controls](docs/NOTEBOOK_PRESENTATION.md).

The website shows saved outputs, not refreshed training runs. Expand **Show Python code**
as needed, or use your browser's **Print → Save as PDF** for a static copy.

## For repository owners and contributors

Website publishing and runner administration are optional, not attendee setup:
see [GitHub Pages instructions](docs/QUARTO_WEBSITE.md) and
[the hosted Issue runner guide](docs/GITHUB_CODEX_AGENT.md). The runner uses each
authorized collaborator's `dev/<username>` branch; changes reach the published
workshop after merging into main.
