# Boiling-point reference library

These optional references support the main [02 vibe-coding workbook](../02_BP_Vibe_Coding_Workbook.ipynb).
Ask your coding assistant to inspect one or more files, explain the useful ideas,
and implement an appropriate approach in your workbook. Do not copy their reported
scores as your own results.

| Reference | What it demonstrates |
| --- | --- |
| [01 · Chemprop GNN](01_BP_Chemprop_GNN.ipynb) | Molecular graphs, segmented training, checkpoint selection, and evaluation. |
| [02 · Frozen MoLFormer](02_BP_Foundation_Model.ipynb) | Pretrained molecular embeddings with a trainable regression head. |
| [03 · Full one-shot prompt](03_BP_One_Shot_Vibe_Coding.ipynb) | A complete specification for an autonomously coded workflow. |
| [04 · Completed one-shot workflow](04_BP_One_Shot_Workflow.ipynb) | Executed descriptor/fingerprint and tree-model experiments with saved-model inference. |
| [Prompt 1 · Request a prompt](one_shot_prompt_1.txt) | Download the initial request used to ask a coding assistant for a fuller workflow specification. |
| [Prompt 2 · Full specification](one_shot_prompt_2.txt) | Download the detailed one-shot implementation and execution request. |

The neural examples illustrate graph learning and pretrained molecular embeddings;
they are not guaranteed to outperform simpler models on this dataset. Reference numbering
is separate from the main `00` → `01` → `02` workshop sequence.
The live exercise uses the short opening prompt in the main 02 workbook; these
longer prompts and completed demonstrations are optional reference material.

## Running an optional reference

Use the workshop `.venv` kernel. Executable references locate the repository from
the repository root, BP module, or this `examples` folder. Shared datasets remain in
`../data/` and `../outputs/nist_acquisition/`; saved model folders remain under
`../outputs/`. No data redownload is needed to use an existing prepared Parquet.
When copying a prompt into an assistant, open the repository root (`workshop code`)
as its workspace: paths inside the prompts are relative to that root. The text
files open as plain text on the reading website; copy or download them from here.

Read the saved demonstration outputs without training. Rerunning may take time and
replace that reference's generated artifacts; it is not required to read the
examples. Record your own data source, split policy, and validation results before
comparing approaches, and keep the test set sealed until modeling choices are fixed.
