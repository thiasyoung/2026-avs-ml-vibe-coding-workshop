# AVS Workshop Setup Guide

Complete this setup before the workshop. This guide includes the Windows, macOS, and
Linux installation instructions and Git setup.

## Platform compatibility

These setup instructions are compatible with these native environments:

| Platform | Workshop environment |
| --- | --- |
| Windows, Intel/AMD 64-bit (x86-64) | Supported package selection; locally verified. |
| macOS, Apple Silicon (M-series), native ARM64 | Compatible package selection; use a current macOS release (13 or newer) and the Apple Silicon VS Code build. |
| Linux, x86-64 or ARM64 | Compatible package selection on a modern glibc distribution, such as Ubuntu 22.04 or newer. |
| macOS, Intel (x86-64) | Not supported natively by the pinned PyTorch 2.5.1 package. |


**Intel Mac users:** contact the organizer for access to a compatible Linux environment
rather than changing package versions locally. PyTorch stopped publishing Intel macOS
builds after 2.2.x; silently downgrading would change the workshop environment.
See the [official PyTorch notice](https://pytorch.org/blog/pytorch2-2/).

## 1. Install the required tools

Install:

- [Visual Studio Code](https://code.visualstudio.com/)
- The Microsoft **Python** extension for VS Code
- The Microsoft **Jupyter** extension for VS Code
- [UV](https://docs.astral.sh/uv/getting-started/installation/)

Choose the VS Code download for your operating system and processor. In VS Code's
Extensions view, install **Python** and **Jupyter**, both published by Microsoft.

Install UV using the command for your platform, following its
[official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
Choose one installation method for your platform. The standalone commands download
and run Astral's installer; review it first if required by your organization's policy.

**Windows — PowerShell:**

```powershell
winget install --id=astral-sh.uv -e
```

If WinGet is unavailable, use:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS — Terminal, if you already use Homebrew:**

```bash
brew install uv
```

**Linux, or macOS without Homebrew — Terminal (bash/zsh):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart VS Code after installation so new terminals pick up the updated PATH, then check:

```text
uv --version
```

## 2. Install Git (recommended for cloning)

Git is optional if you use GitHub's ZIP download instead. To clone the repository,
first check whether Git is already installed in PowerShell or Terminal:

```text
git --version
```

If a version number appears, continue to step 3. Otherwise, install it for your platform:

### Windows

Download [Git for Windows](https://git-scm.com/download/win) and run the installer.
The default installation options are generally suitable for the workshop. VS Code's
Git integration still requires Git itself to be installed.

### macOS

Running `git --version` may prompt you to install Apple's Command Line Tools. Complete
that installation, or request it explicitly:

```bash
xcode-select --install
```

If you already use Homebrew, you can instead install Git with:

```bash
brew install git
```

See the [official macOS installation options](https://git-scm.com/install/mac).

### Linux

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install git
```

On Fedora:

```bash
sudo dnf install git
```

For another distribution, use its package manager; see the
[official Linux installation options](https://git-scm.com/install/linux).

After installation, restart VS Code and your terminal, then run `git --version` again.
You do not need to configure a Git author name, email, or GitHub credentials just to
clone a **public** repository and run these notebooks. Committing or pushing changes
is separate from the workshop setup.

## 3. Get this repository

Clone the repository with Git (the same commands work on all three operating systems):

```text
git clone https://github.com/thiasyoung/2026-avs-ml-vibe-coding-workshop.git
cd 2026-avs-ml-vibe-coding-workshop
```

Alternatively, use GitHub **Code → Download ZIP** and extract it instead; no Git
installation is needed for this route. A public repository requires no GitHub sign-in
to clone or download. If the repository is private, ask the organizer for access and
sign in to GitHub to download the ZIP, or use your organization's approved GitHub
authentication method for Git. Your GitHub account password is not a Git HTTPS password.
Open the cloned or extracted workshop folder in VS Code using **File → Open Folder**.
The folder you open must contain `pyproject.toml`, `uv.lock`, and `test_environment.py`.
You can also run `code .` from that folder if the VS Code command is available.

## 4. Create and verify the UV environment

Use **Terminal → New Terminal** in VS Code. At the root of this workshop folder, run
the same command on Windows, macOS, or Linux:

```text
uv run --locked python test_environment.py
```

UV creates a private `.venv` in this folder, provides Python 3.11 when needed, and installs
the packages selected for your platform from `uv.lock`. No environment activation is
required. This is the CPU-capable workshop environment; a GPU or CUDA installation is
not required. Internet is needed for the initial environment installation and any
uncached data/model downloads.

Create the environment separately on each computer—do not copy `.venv` between operating
systems. Share the repository and lockfile instead. Do not replace the locked packages
with platform-specific guesses if an installation fails. Contact an organizer for support.

## 5. Select the workshop kernel

Open `00_Test_Installation.ipynb` in VS Code. In the notebook kernel picker, select the Python interpreter located inside this workshop folder's `.venv` directory. Run all cells; every check should report success.

If VS Code does not find it automatically, use **Python: Select Interpreter** from the
Command Palette, choose **Enter interpreter path**, and browse to the executable below.
Then return to the notebook kernel picker and choose that environment.

| Platform | Interpreter inside the workshop folder |
| --- | --- |
| Windows | `.venv\Scripts\python.exe` |
| macOS / Linux | `.venv/bin/python` |

Open the Command Palette with **Ctrl+Shift+P** on Windows/Linux or **Cmd+Shift+P** on macOS.
The notebook's printed executable should point into this workshop's `.venv`.

After the installation check, open the [workshop roadmap](00_Workshop_Roadmap.ipynb).
Follow the filename numbers in each module: [BP `00`–`04`](01_boiling_point/README.md),
[IR `00`–`03`](02_ir_functional_groups/README.md), or
[HTEM `00`–`03`](03_htem_conductivity/README.md). Lettered examples are alternatives
to their numbered step. The module guides explain the acquisition pre-work and live exercises.

## 6. Select a coding-agent workflow

### Path A (recommended)— Codex in VS Code

Install the OpenAI **Codex** extension for VS Code, open the Codex view, and sign in. Keep this workshop folder open as the VS Code workspace. Codex can inspect the local notebooks and data in that workspace, so you can request tasks against the actual files without uploading them or re-explaining their structure.

For example:

> Inspect `02_ir_functional_groups/01_IR_Vibe_Coding_Data_Carpentry.ipynb` and explain the checks applied before writing the model-ready dataset.

### Path B (advanced users)— Containerized local-LLM agent workflow

Use the separate local-agent repository: `<CONTAINERIZED_LOCAL_LLM_AGENT_REPOSITORY_URL>`. Follow its instructions to set up and run the containerized local-LLM agent, then use it with this same local workshop folder. The local-agent repository owns all container, model, and agent instructions.

## Pre-work checklist

- [ ] VS Code and the Microsoft Python and Jupyter extensions are installed.
- [ ] `uv --version` works.
- [ ] `git --version` works, or I chose the ZIP download instead.
- [ ] The cloned or extracted workshop folder is open in VS Code.
- [ ] `uv run --locked python test_environment.py` completes successfully.
- [ ] `00_Test_Installation.ipynb` runs successfully using the workshop `.venv` kernel.
- [ ] My chosen coding-agent workflow is set up and ready to use.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'nistchempy'`:** In the VS Code notebook kernel picker, select the Python executable inside this workshop folder's `.venv` directory. In the VS Code terminal at the workshop root, run `uv sync --locked`, restart the notebook kernel, and run the cell again.
- **`uv` is not recognized:** close and reopen the terminal, then run `uv --version`.
- **`git` is not recognized:** complete step 2 and restart VS Code, or use the ZIP download in step 3.
- **GitHub says the repository cannot be found:** check the URL and ask the organizer whether the repository is private and your account has access. No GitHub sign-in is needed for a public download.
- **First installation is slow:** this is normal for the CPU PyTorch, RDKit, and Chemprop stack. Keep `.venv` after the first successful setup.
- **The notebook uses the wrong Python:** use the notebook kernel picker to select this folder's `.venv` interpreter, then restart the kernel.
- **Apple Silicon reports an Intel/x86-64 PyTorch error:** use the native ARM64 VS Code/terminal rather than Rosetta, then create a fresh native workshop environment. A `.venv` made using an Intel interpreter cannot be reused as ARM64.
- **No compatible PyTorch wheel:** check the platform table above. In particular, Intel macOS is not supported by this pinned environment; changing installation commands will not supply a missing wheel.
- **Linux cannot load a shared library:** use a supported glibc distribution and send the exact missing-library message to the organizer; package availability alone does not verify system-library compatibility.

