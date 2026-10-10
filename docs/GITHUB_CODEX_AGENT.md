# Codex Issue-to-pull-request agent

The repository includes `.github/workflows/codex-issue-agent.yml`, a single OpenAI Codex workflow that turns a trusted Issue request into a feature-branch pull request. It is intentionally inactive until a maintainer enables it in GitHub.

## Activation

The workflow already references the existing `OPENAI_API_KEY` GitHub Actions secret. Do not expose, paste, replace, or test that key in an Issue, pull request, notebook, terminal log, or source file.

After this setup pull request is reviewed and merged to `main`, make these manual GitHub changes:

1. In **Settings → Actions → General**, keep the default workflow token read-only and enable **Allow GitHub Actions to create and approve pull requests**. The agent still cannot merge a pull request.
2. In **Settings → Secrets and variables → Actions → Variables**, create `CODEX_AGENT_ENABLED` with the exact value `true`.
3. Protect `main` with pull requests and required checks. Restrict who may push to it.

The workflow is hard-coded initially to admit only the GitHub account `thiasyoung`, inferred from this repository’s GitHub remote. Change that identity only in a reviewed pull request.

## Operation

Create an Issue whose body contains `@codex`, or add a new comment containing `@codex` to an Issue. Keep the request concrete. For example:

```text
@codex Add a concise presenter note explaining how to select the UV kernel in VS Code. Update the relevant setup documentation and validate it.
```

For an allowed request, the workflow:

1. checks the `@codex` trigger, exact username, enable flag, request length, duplicate marker, and per-Issue run cap;
2. checks out `main`, or the existing `codex/issue-<number>` branch for a follow-up;
3. runs `openai/codex-action@v1` with the existing `OPENAI_API_KEY`, `gpt-5.6-terra` at `xhigh` reasoning effort, and a constrained workspace permission profile;
4. asks Codex to inspect the Issue and repository, make a focused change, and run only lightweight tests;
5. transfers the returned diff to a fresh publisher job that validates it, pushes only `codex/issue-<number>`, and creates or updates a pull request;
6. comments on the originating Issue with the summary, reported checks, and pull-request link.

The agent never pushes directly to `main`, never merges, and never triggers from pull-request comments. Follow-up `@codex` comments on the original Issue update the same feature branch and pull request.

## Safety model and limitations

- The Codex job is isolated from the GitHub write token. Checkout credentials are not persisted, `GITHUB_TOKEN` and `GH_TOKEN` are blanked for the Action step, and the write-capable publisher runs later on a fresh runner.
- Codex runs with `permission-profile: :workspace` so it can edit its disposable checkout and run lightweight tests, and `safety-strategy: drop-sudo` to protect the API-key proxy. The Action has no network permission under that profile.
- Codex is the final step of its job. Its only handoff is structured output containing a unified diff. The publisher rejects changes to `.github/`, `AGENTS.md`, `.gitignore`, and this security document, as well as binary or symlink patches.
- The publisher intentionally does not execute the returned patch’s tests while it holds a write token. Instead, the proposed pull request runs the separate read-only `Fast workshop checks` workflow, which verifies imports, notebook syntax, automation invariants, and small data contracts.
- Issue text, repository content, and test code can be malicious. The exact-user gate, isolated jobs, no persisted credentials, output validation, protected paths, and branch protection reduce that risk but cannot eliminate it. Review every pull request. Disable the workflow by changing `CODEX_AGENT_ENABLED` to `false` if behavior or spend is unexpected.
- The workflow limits a request to 6,000 characters and six completed runs per Issue. `gpt-5.6-terra` balances intelligence and cost, but `xhigh` reasoning can materially increase token use. API-project model access and billing limits are account-specific; use OpenAI project budgets and usage alerts as the real cost control.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| No workflow runs | Confirm the workflow is merged to `main`, `CODEX_AGENT_ENABLED` is `true`, the request includes `@codex`, and the trigger was sent by `thiasyoung`. |
| Codex does not start | Confirm that the existing `OPENAI_API_KEY` secret is still configured and that the API project can use `gpt-5.6-terra`. |
| No pull request | Codex may have reported no focused allowed change, its patch may have been rejected, or GitHub Actions may not be allowed to create pull requests. Read the workflow log and the bot comment. |
| A change is rejected | The request touched protected automation/policy files, used binary/symlink content, or no longer applied cleanly. Make the change through a normal reviewed pull request. |

The implementation follows the official [Codex GitHub Action README](https://github.com/openai/codex-action/blob/main/README.md), its [security guidance](https://github.com/openai/codex-action/blob/main/docs/security.md), and the current [GPT-5.6 Terra model page](https://developers.openai.com/api/docs/models/gpt-5.6-terra).
