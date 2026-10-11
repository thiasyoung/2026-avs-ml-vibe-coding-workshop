# Codex workshop runner

This repository includes a GitHub Issue-triggered OpenAI Codex runner for trusted workshop collaborators. It is intended to let the GitHub-hosted runner perform real repository work—including public data acquisition, notebook execution, and model training—without requiring a personal workstation to stay online.

## Activation

The workflow uses the existing `OPENAI_API_KEY` repository secret. Do not paste, replace, or reveal that value in a notebook, Issue, workflow log, or source file.

In GitHub repository settings:

1. Enable GitHub Actions.
2. Set **Workflow permissions** to read repository contents and packages. The workflow grants write permission only to its isolated branch-publishing job.
3. Create the Actions variable `CODEX_AGENT_ENABLED` with the value `true` (or leave it unset; the workflow defaults to enabled).
4. Protect `main` against direct pushes.

## Using it

Open an Issue, or add a comment to an existing Issue, containing `@codex` and a concrete request. The authenticated GitHub account that creates the Issue/comment is used; usernames written in the text are ignored.

```text
@codex Set up the locked environment and execute every boiling-point notebook.
Keep the populated notebooks and useful output data and figures in 01_boiling_point.
```

The requester must be a repository collaborator with `write`, `maintain`, or `admin` permission. Each requester gets a `dev/<github-username>` branch. Requests targeting the same branch wait their turn; requests from different collaborators can run in parallel. On completion, the workflow commits to that development branch and replies on the Issue with the branch link and run summary. It never force-pushes, pushes to `main`, or automatically merges.

Two requests have special branch-management behavior:

- `@codex sync with main` merges the current `main` into the requester's development branch and has Codex resolve straightforward conflicts.
- `@codex prepare merge` runs relevant checks and opens (or reuses) a pull request from the requester's development branch to `main`. It never merges that pull request.

The runner uses `openai/codex-action@v1` with `gpt-5.6-terra`, `xhigh` reasoning effort, and its `danger-full-access` sandbox mode. This lets the disposable hosted runner open Jupyter kernels, access public network resources, download models/data, and execute the requested workloads. Codex receives the OpenAI API credential but not the GitHub write token; the subsequent publishing step receives the GitHub token and commits the completed checkout.

Generated `outputs/` folders are ignored by default to keep local work clean. After a runner job completes, it deliberately includes useful generated outputs under 50 MB. It excludes virtual environments, caches, TensorBoard event logs, checkpoints, and large model-weight files because they are poor Git artifacts and can make a push fail. Populated notebooks, compact datasets, tables, and figures are included.

## Limits of a GitHub-hosted runner

GitHub-hosted runners are disposable, CPU-only machines with finite disk, memory, wall-clock, and network allowances. Long deep-learning training and large model downloads can still exceed those platform limits. If that happens, the Issue reply identifies the failure; use a self-hosted GPU runner or the separate containerized local-agent workflow for that workload.

This is intentionally a trusted-collaborator tool. Anyone with write access who triggers it can spend the shared API budget and cause public network/model workloads to run. GitHub Free cannot provide a finer per-user cost quota for the shared secret. Use repository access, OpenAI project spending limits, and the `CODEX_AGENT_ENABLED=false` kill switch to control access and cost.

The configuration follows the official [Codex GitHub Action documentation](https://developers.openai.com/codex/github-action/).
