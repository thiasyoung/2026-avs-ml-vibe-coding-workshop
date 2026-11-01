# Boiling-point prediction workshop

## Objective

Predict a molecule's reported boiling point from its **SMILES string**, then learn how
to build, evaluate, and question a complete machine-learning workflow. Compare a
Chemprop graph neural network with a frozen MoLFormer encoder plus a small regression
network, and use prompts to develop another approach.

Inputs are molecular structures only; boiling point is the target, not a feature.
The model notebooks report temperatures and prediction errors in **kelvin**.

## Notebook sequence

- [00 — Get data](00_BP_Get_Data_NISTChemPy_v2.ipynb): acquire NIST records, validate
   molecular identities and kelvin values, and save a cleansed SMILES/BP table.
- [01 — Chemprop GNN](01_BP_Chemprop_GNN.ipynb): train a molecular graph model,
   inspect training/validation loss, evaluate predictions, and try a new SMILES.
- [02 — Foundation model](02_BP_Foundation_Model.ipynb): freeze MoLFormer embeddings
   and train a feed-forward head using the same molecular splits as 01.
- [03 — One-shot prompt](03_BP_One_Shot_Vibe_Coding.ipynb): specify a complete
   prediction workflow for a coding assistant. This notebook contains prompts, not training code.
- [04 — Worked one-shot workflow](04_BP_One_Shot_Workflow.ipynb): review a completed
   descriptor/fingerprint and tree-model example, including experiments and saved-model inference.

## Before running

Follow the [workshop setup guide](../SETUP_GUIDE.md) and select the workshop `.venv`
kernel in VS Code. Run each notebook top-to-bottom from its module folder.

Notebook 00 can cleanse an existing acquisition Parquet without repeating the long
download. Full live collection is optional pre-work; one illustrative NIST fetch still
requires internet. Model notebooks prefer `outputs/nist_acquisition/NIST_BP_SMILES.parquet`
when present and otherwise use `data/CompoundData.parquet`. Keep the data source and
split settings unchanged when comparing 01 and 02. MoLFormer downloads its public
checkpoint on first use; allow time for this before the session.

## What to take away

Inspect loss curves, the selected checkpoint, held-out parity plots, and failure cases—not
just a convincing single-molecule prediction. Use validation results for development and
reserve test results for final evaluation. Random molecular splits do not establish
performance on unseen chemical scaffolds, and these labels are not verified to share
one measurement pressure. Generated artifacts are saved under `outputs/`.
