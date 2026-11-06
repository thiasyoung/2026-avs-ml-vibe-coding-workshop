# Boiling-point reference library

These optional references support the main [02 vibe-coding workbook](../02_BP_Vibe_Coding_Workbook.ipynb).
Ask your coding assistant to inspect one or more files, explain the useful ideas,
and implement an appropriate approach in your workbook. Do not copy their reported
scores as your own results.

The library has two parts:

| Folder | Purpose |
| --- | --- |
| [`reference_models/`](#reference-models) | Worked neural-network examples: a Chemprop GNN and a frozen MoLFormer encoder. |
| [`prompt_iteration/`](prompt_iteration/README.md) | The same task generated from four prompts that pull different strategy levers (P0–P3), their executed notebooks, and a prompt-only notebook for mining the strategies. |

## Reference models

| Reference | What it demonstrates |
| --- | --- |
| [01 · Chemprop GNN](reference_models/01_BP_Chemprop_GNN.ipynb) | Molecular graphs, segmented training, checkpoint selection, and evaluation. |
| [02 · Frozen MoLFormer](reference_models/02_BP_Foundation_Model.ipynb) | Pretrained molecular embeddings with a trainable regression head. |

These illustrate graph learning and pretrained molecular embeddings; they are not
guaranteed to outperform simpler models on this dataset.

## Prompt strategies

See the [prompt-strategies guide](prompt_iteration/README.md) for which levers each of the four
prompts (P0–P3) pulled, what showed up in the notebooks they produced, and
[Compare prompt strategies](prompt_iteration/Compare_Prompt_Strategies.ipynb), a copyable
prompt that asks your assistant to analyze them. The levers themselves are listed in the
strategy toolbox in the 02 workbook. These replace the earlier full one-shot prompt and
completed one-shot workflow references.

Reference numbering is separate from the main `00` → `01` → `02` workshop sequence.
The live exercise uses the short opening prompt in the main 02 workbook; everything
here is optional reading.

## Running an optional reference

Use the workshop `.venv` kernel. Executable notebooks locate the repository by walking
up from the working directory, so they run from the repository root, the BP module,
or their own folder. Shared datasets remain in `../data/` and
`../outputs/nist_acquisition/`; generated model folders are written under `../outputs/`
(not committed). No data redownload is needed to use an existing prepared Parquet.
When copying a prompt into an assistant, open the repository root (`workshop code`)
as its workspace: paths inside the prompts are relative to that root. The text
files open as plain text on the reading website; copy or download them from there.

Read the saved outputs without training. Rerunning may take time and replace that
reference's generated artifacts; it is not required to read the examples. Record
your own data source, split policy, and validation results before comparing
approaches, and keep the test set sealed until modeling choices are fixed.
