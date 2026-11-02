# Codex workshop runner

This repository includes a GitHub Issue-triggered OpenAI Codex runner for trusted workshop collaborators. It is intended to let the GitHub-hosted runner perform real repository work—including public data acquisition, notebook execution, and model training—without requiring a personal workstation to stay online.

## Activation

The workflow uses the existing `OPENAI_API_KEY` repository secret. Do not paste, replace, or reveal that value in a notebook, Issue, workflow log, or source file.

In GitHub repository settings:

1. Enable GitHub Actions.
2. Keep default **Workflow permissions** at read repository contents and packages. The workflow explicitly grants contents/Issue write permission to the runner/publishing job; the authorization job remains read-only. A separate publishing step receives the write token after Codex finishes.
3. Create the Actions variable `CODEX_AGENT_ENABLED` with the value `true` (or leave it unset; the workflow defaults to enabled).
4. Allow the workflow's `GITHUB_TOKEN` to create/update `dev/*` branches. Rules requiring a PR before every development-branch update can reject these pushes. Keep main protected where your plan supports it: the runner no longer needs direct-write access to main. Keep force pushes and branch deletion blocked.
5. Enable **Allow GitHub Actions to create and approve pull requests** for the optional `@codex prepare merge` command. The runner only creates PRs; it never approves or merges them. Ordinary development does not need this setting.

Commit and push the updated workflow, prompt instructions, and guidance to main to activate them. First let old runner jobs finish or cancel them: runs already using the previous direct-to-main workflow retain its shared lock and can still publish to main. Existing `dev/*` branches and their unmerged commits are reused, never reset or automatically deleted. These settings do not switch your local VS Code checkout; you can stay on main locally.

GitHub documents [branch-protection requirements](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) and [ruleset bypasses](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository). Feature availability depends on the repository owner and plan; private repositories on GitHub Free do not have all protected-branch controls. Do not add or expose a personal token merely to bypass a rejected push.

## Using it

Open an Issue, or add a comment to an existing Issue, containing `@codex` and a concrete request. The authenticated GitHub account that creates the Issue/comment is used; usernames written in the text are ignored.

```text
@codex Set up the locked environment and execute every boiling-point notebook.
Keep the populated notebooks and useful output data and figures in 01_boiling_point.
```

The requester must be an authenticated human repository collaborator with `write`, `maintain`, or `admin` permission, or the repository owner. Authorization uses GitHub's event sender and the collaborator API's top-level `permission` field, not usernames written in the request. Bot events, pull-request comments, and users without write access do not start Codex.

Each request targets `dev/<username>` using the lowercase authenticated event sender, not the Issue author or a username written in the request. For example, requests from `thiasyoung` update `dev/thiasyoung`. Follow-up comments by another collaborator target that collaborator's branch. After acquiring the branch lock, the runner creates a missing branch from the latest main or reuses the existing branch without resetting it. Successful ordinary work is committed and pushed to that branch without a PR or manual approval. The Issue reply includes results and branch/commit links. No runner command pushes directly to main.

Each user's queue uses GitHub's `queue: max`: up to 100 pending jobs can wait per branch without replacing one another. Requests are processed by when they enter the queue, which may differ from Issue/comment creation order. Requests for the same user never edit/publish concurrently; different users can run in parallel, subject to GitHub's runner quota. See [GitHub's concurrency guide](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).

Two standalone commands manage your development branch:

- `@codex sync with main`: merge main into your development branch, let Codex resolve straightforward conflicts, and validate the result. Unresolved conflicts block publishing and are reported for human review. Existing branches do not otherwise sync automatically.
- `@codex prepare merge`: ask Codex to run relevant tests, then create or reuse a PR from your branch into main after a successful run. If your branch has no commits ahead of main, no PR is needed. Review the reported checks before merging; the runner never automatically merges. Mentioning a command inside a larger ordinary request does not invoke it.

The runner never force-pushes or rebases changes after testing. If a person or another tool updates your development branch while Codex is running, the publishing step refuses the stale update and reports failure; submit the request again to work from its latest revision. A Git push racing the final check is also rejected rather than overwriting commits. Updates to main do not reset existing development branches; request a sync when needed. Local edits are not part of the queue: pull the branch you plan to edit and coordinate local pushes with runner jobs.

The runner retains `openai/codex-action@v1`, `gpt-5.6-terra`, `xhigh` reasoning, the existing `OPENAI_API_KEY`, and `danger-full-access` sandbox mode. This lets the disposable hosted runner open Jupyter kernels, access public network resources, download models/data, and execute requested workloads. Codex is not passed the GitHub write token and checkout does not persist credentials; separate workflow steps create branches, fetch a requested sync, publish changes, and create a requested PR. These are step-level boundaries on a trusted shared runner, not a hardened isolation boundary against malicious collaborators, repository code, or persistent background processes. The publishing job's token is repository-scoped, not limited to one dev branch: branch naming/guards are workflow policy, and branch rules are a separate enforcement layer where available.

Changing branch routing does not repair the separately observed `drop-sudo` startup failure (`Could not restrict access to /run/dbus/system_bus_socket`). The action's safety strategy is unchanged. That runner/action startup problem must be resolved before an API-backed run can edit or publish work; local workflow tests do not establish that hosted startup succeeds.

If the Codex step fails, the workflow does not publish its partial checkout. A completed Codex session may still report failed notebook cells or external resource problems; read its reported checks, not just whether a commit was created.

Commits pushed with `GITHUB_TOKEN` do not normally trigger other push workflows. Do not assume separate PR checks ran: Codex must run relevant checks within this request, and the Issue reply reports the checks it actually performed. See [GitHub's workflow-trigger guidance](https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-when-your-workflow-runs/triggering-a-workflow#triggering-a-workflow-from-a-workflow).

Generated `outputs/` folders are ignored by default to keep local work clean. After a runner job completes, it deliberately includes useful generated outputs under 50 MB. It excludes virtual environments, caches, TensorBoard event logs, checkpoints, and large model-weight files because they are poor Git artifacts and can make a push fail. Populated notebooks, compact datasets, tables, and figures are included.

## Limits of a GitHub-hosted runner

GitHub-hosted runners are disposable, CPU-only machines with finite disk, memory, wall-clock, and network allowances. Long deep-learning training and large model downloads can still exceed those platform limits. If that happens, the Issue reply identifies the failure; use a self-hosted GPU runner or the separate containerized local-agent workflow for that workload.

This is intentionally a trusted-collaborator tool. Anyone with write access who triggers it can spend the shared API budget and cause public network/model workloads to run. GitHub Free cannot provide a finer per-user cost quota for the shared secret. Use repository access, OpenAI project spending limits, and the `CODEX_AGENT_ENABLED=false` kill switch to control access and cost.

The configuration follows the official [Codex GitHub Action documentation](https://developers.openai.com/codex/github-action/).
