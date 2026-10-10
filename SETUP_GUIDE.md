# AVS Workshop Setup Guide

This guide is distributed with the workshop-content repository. All participants use **UV + VS Code**, then choose one supported coding-agent workflow: Codex in VS Code or the separate containerized local-LLM agent workflow.

For the full pre-work instructions, including VS Code and extension installation, use the separate environment-setup repository: `<ENVIRONMENT_SETUP_REPOSITORY_URL>`.

## 1. Install the required tools

Install:

- [Visual Studio Code](https://code.visualstudio.com/)
- The Microsoft **Python** extension for VS Code
- The Microsoft **Jupyter** extension for VS Code
- [UV](https://docs.astral.sh/uv/getting-started/installation/)

## 2. Get this repository

Clone the repository:

```bash
git clone <WORKSHOP_CONTENT_REPOSITORY_URL>
cd <WORKSHOP_CONTENT_REPOSITORY_FOLDER>
```

Or download and extract the GitHub ZIP archive. Then open the extracted workshop folder in VS Code using **File → Open Folder**.

## 3. Create and verify the UV environment

Open the VS Code integrated terminal at the root of this workshop folder and run:

```bash
uv run --locked python test_environment.py
```

UV creates a private `.venv` in this folder, provides Python 3.11 when needed, and installs the exact package versions recorded in `uv.lock`. No environment activation is required.

## 4. Select the workshop kernel

Open `00_Test_Installation.ipynb` in VS Code. In the notebook kernel picker, select the Python interpreter located inside this workshop folder's `.venv` directory. Run all cells; every check should report success.

If VS Code does not find it automatically, use **Python: Select Interpreter** from the Command Palette and select the `.venv` Python executable.

## 5. Select a coding-agent workflow

### Path A — Codex in VS Code

Install the OpenAI **Codex** extension for VS Code, open the Codex view, and sign in. Keep this workshop folder open as the VS Code workspace. Codex can inspect the local notebooks and data in that workspace, so you can request tasks against the actual files without uploading them or re-explaining their structure.

For example:

> Inspect `02_ir_functional_groups/01_IR_Vibe_Coding_Data_Carpentry.ipynb` and explain the checks applied before writing the model-ready dataset.

### Path B — Containerized local-LLM agent workflow

Use the separate local-agent repository: `<CONTAINERIZED_LOCAL_LLM_AGENT_REPOSITORY_URL>`. Follow its instructions to set up and run the containerized local-LLM agent, then use it with this same local workshop folder. The local-agent repository owns all container, model, and agent instructions.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'nistchempy'`:** In the VS Code notebook kernel picker, select the Python executable inside this workshop folder's `.venv` directory. In the VS Code terminal at the workshop root, run `uv sync --locked`, restart the notebook kernel, and run the cell again.
- **`uv` is not recognized:** close and reopen the terminal, then run `uv --version`.
- **First installation is slow:** this is normal for the CPU PyTorch, RDKit, and Chemprop stack. Keep `.venv` after the first successful setup.
- **The notebook uses the wrong Python:** use the notebook kernel picker to select this folder's `.venv` interpreter, then restart the kernel.

The workshop is configured to copy packages rather than hard-link them, which makes it reliable in Windows OneDrive folders. An isolated environment prevents dependency conflicts but is not a security sandbox; run only material you trust.
