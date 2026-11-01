# Codex workshop runner

This repository includes a GitHub Issue-triggered OpenAI Codex runner for trusted workshop collaborators. It is intended to let the GitHub-hosted runner perform real repository work—including public data acquisition, notebook execution, and model training—without requiring a personal workstation to stay online.

## Activation

The workflow uses the existing `OPENAI_API_KEY` repository secret. Do not paste, replace, or reveal that value in a notebook, Issue, workflow log, or source file.

In GitHub repository settings:

1. Enable GitHub Actions.
2. Keep default **Workflow permissions** at read repository contents and packages. The workflow explicitly grants contents/Issue write permission to the runner/publishing job; the authorization job remains read-only. A separate publishing step receives the write token after Codex finishes.
3. Create the Actions variable `CODEX_AGENT_ENABLED` with the value `true` (or leave it unset; the workflow defaults to enabled).
4. Allow this workflow's `GITHUB_TOKEN` to push directly to `main`. A rule requiring pull requests, approvals, or checks before every update can reject these pushes. Where supported, configure a narrowly scoped automation bypass; otherwise remove the incompatible requirement for main if you choose direct-write operation. Keep force pushes and branch deletion blocked. This change does not grant the bot a bypass automatically or alter repository settings.

Commit and push the updated workflow, prompt instructions, and guidance to main to activate them. First let old runner jobs finish or cancel them: runs already using the previous workflow keep its old per-user lock and are not coordinated with the new shared lock. Existing `dev/*` branches and their unmerged commits are left untouched; review and merge any work you want to keep separately.

GitHub documents [branch-protection requirements](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) and [ruleset bypasses](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository). Feature availability depends on the repository owner and plan; private repositories on GitHub Free do not have all protected-branch controls. Do not add or expose a personal token merely to bypass a rejected push.

## Using it

Open an Issue, or add a comment to an existing Issue, containing `@codex` and a concrete request. The authenticated GitHub account that creates the Issue/comment is used; usernames written in the text are ignored.

```text
@codex Set up the locked environment and execute every boiling-point notebook.
Keep the populated notebooks and useful output data and figures in 01_boiling_point.
```

The requester must be an authenticated human repository collaborator with `write`, `maintain`, or `admin` permission, or the repository owner. Authorization uses GitHub's event sender and the collaborator API's top-level `permission` field, not usernames written in the request. Bot events, pull-request comments, and users without write access do not start Codex.

All collaborators now work on the same `main` branch. After authorization, requests share one repository-wide execution/publishing queue; different users no longer train or edit in parallel on separate branches. Each job checks out the latest main after acquiring the lock. Successful work is committed and pushed directly to main, with no pull request or manual approval. The Issue reply includes results, the main link, and a commit link when changes were pushed.

The queue uses GitHub's `queue: max`: up to 100 pending jobs can wait without replacing one another. Requests are processed by when they enter the queue, which may differ from Issue/comment creation order. Authorizations can run in parallel, but editing and publishing cannot. See [GitHub's concurrency guide](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).

The legacy standalone commands `@codex sync with main` and `@codex prepare merge` now request validation of main only. They do not merge a development branch or open a pull request. Use a normal `@codex` request for further revisions.

The runner never force-pushes or rebases changes after testing. If a person or another tool updates main while Codex is running, the publishing step refuses the stale update and reports failure; submit the request again so Codex can work from the latest main. A Git push racing the final check is also rejected rather than overwriting commits. Local edits are not part of this queue: commit/push local work between runner jobs, and pull main before editing locally.

The runner retains `openai/codex-action@v1`, `gpt-5.6-terra`, `xhigh` reasoning, the existing `OPENAI_API_KEY`, and `danger-full-access` sandbox mode. This lets the disposable hosted runner open Jupyter kernels, access public network resources, download models/data, and execute requested workloads. Codex is not passed the GitHub write token and checkout does not persist credentials; the subsequent publishing step receives the token and commits the completed checkout. These are step-level boundaries on a trusted shared runner, not a hardened isolation boundary against malicious collaborators, repository code, or persistent background processes. Direct writes also remove the pre-merge human-review checkpoint.

If the Codex step fails, the workflow does not publish its partial checkout. A completed Codex session may still report failed notebook cells or external resource problems; read its reported checks, not just whether a commit was created.

Commits pushed with `GITHUB_TOKEN` do not normally trigger other push workflows. Do not assume separate PR checks ran: Codex must run relevant checks within this request, and the Issue reply reports the checks it actually performed. See [GitHub's workflow-trigger guidance](https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-when-your-workflow-runs/triggering-a-workflow#triggering-a-workflow-from-a-workflow).

Generated `outputs/` folders are ignored by default to keep local work clean. After a runner job completes, it deliberately includes useful generated outputs under 50 MB. It excludes virtual environments, caches, TensorBoard event logs, checkpoints, and large model-weight files because they are poor Git artifacts and can make a push fail. Populated notebooks, compact datasets, tables, and figures are included.

## Limits of a GitHub-hosted runner

GitHub-hosted runners are disposable, CPU-only machines with finite disk, memory, wall-clock, and network allowances. Long deep-learning training and large model downloads can still exceed those platform limits. If that happens, the Issue reply identifies the failure; use a self-hosted GPU runner or the separate containerized local-agent workflow for that workload.

This is intentionally a trusted-collaborator tool. Anyone with write access who triggers it can spend the shared API budget and cause public network/model workloads to run. GitHub Free cannot provide a finer per-user cost quota for the shared secret. Use repository access, OpenAI project spending limits, and the `CODEX_AGENT_ENABLED=false` kill switch to control access and cost.

The configuration follows the official [Codex GitHub Action documentation](https://developers.openai.com/codex/github-action/).
