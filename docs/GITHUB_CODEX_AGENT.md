# Codex collaborator development agent

This repository has one Issue-triggered workflow, `.github/workflows/codex-issue-agent.yml`. It lets authorized repository collaborators use Codex for ordinary development without creating a pull request for every small change.

## Activation

The workflow uses the existing repository Actions secret `OPENAI_API_KEY`. It is one shared secret for every authorized workflow invocation; collaborators never need, receive, or see its value.

After merging this setup change into `main`:

1. In **Settings → Actions → General**, keep the default `GITHUB_TOKEN` permissions restricted and enable **Allow GitHub Actions to create and approve pull requests**. The latter is needed only for `@codex prepare merge`.
2. In **Settings → Secrets and variables → Actions → Variables**, set `CODEX_AGENT_ENABLED` to `true`.
3. Protect `main` with a pull-request requirement and restrict direct pushes. The workflow never pushes to `main`.

## Everyday use

Create an Issue containing `@codex`, or add a comment containing `@codex` to an Issue. The workflow uses the GitHub account that submitted that event—not a username written in the Issue—to select `dev/<github-login>`.

For an account with repository **write**, **maintain**, or **admin** access, ordinary requests:

1. create `dev/<github-login>` from the current `main` tip only if it does not exist, otherwise reuse it;
2. queue other requests for that same collaborator while allowing other collaborators to run in parallel;
3. run `openai/codex-action@v1` against that development branch with the existing shared API secret;
4. validate Codex's structured patch on a fresh runner, commit it, and push normally to that development branch; and
5. comment on the originating Issue with a branch link, summary, and reported checks.

The publisher uses a normal Git push, never force-pushes, and reports a rejection rather than overwriting a branch changed while a run was active.

Examples:

```text
@codex Update the presenter notes explaining how to select the UV kernel in VS Code. Run the relevant lightweight checks.

@codex sync with main

@codex prepare merge
```

`@codex sync with main` merges the current `main` into the requester's development branch. Git resolves straightforward non-overlapping changes; if conflicts remain, the workflow aborts the merge and reports the paths requiring human review.

`@codex prepare merge` runs the repository's lightweight checks on the requester's development branch. Only when they pass does it create or update a pull request from that branch to `main`. It does not merge it.

## Security model

- Authorization is based on the GitHub event sender and the repository collaborator-permission API. Text in an Issue cannot select another user's branch or grant access.
- Codex runs in a read-only job with `permission-profile: :workspace`, `safety-strategy: drop-sudo`, no persisted checkout credential, and blank `GITHUB_TOKEN`/`GH_TOKEN` values. It returns structured patch text only.
- A later job with narrow write permission validates that patch and rejects changes to workflows, authorization/policy files, Git configuration, binary data, and symlinks before committing to `dev/<github-login>`.
- The visible `codex-agent-request.json` artifact contains only bounded Issue metadata and trigger text—never API keys, GitHub tokens, or credentials. The workflow checks that the file exists before upload.
- Synchronization and PR preparation are separate jobs. A default-branch merge or direct `main` push is never performed by the agent.

## Limitations

This is meaningful defense in depth, not a sandbox with absolute guarantees. In particular, GitHub Free private repositories do not provide every enterprise control, such as required environment approvals, custom organization runner isolation, or advanced ruleset enforcement. A collaborator who can directly push to an unprotected `main` branch can also bypass this workflow entirely. Protect `main`, limit write access to trusted collaborators, review merge-preparation PRs, and disable the agent by setting `CODEX_AGENT_ENABLED` to `false` if needed.

The workflow runs repository test code. The Codex job is isolated from GitHub write credentials, but tests and repository content should still be treated as untrusted. Keep the lightweight checks deterministic and do not add network downloads, model training, or secret-dependent tests to this workflow.

The implementation uses the official [OpenAI Codex GitHub Action documentation](https://developers.openai.com/codex/github-action/).
