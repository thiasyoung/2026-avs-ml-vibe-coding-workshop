"""Static contract checks for the trusted Codex Issue runner."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/codex-issue-agent.yml"
REQUIRED_SNIPPETS = (
    "openai/codex-action@v1",
    "actions/github-script@v8",
    "actions/upload-artifact@v6",
    "actions/download-artifact@v7",
    "OPENAI_API_KEY",
    "model: gpt-5.6-terra",
    "effort: xhigh",
    "CODEX_AGENT_ENABLED",
    "sandbox: danger-full-access",
    "GITHUB_TOKEN: ''",
    "dev/${actor}",
    "getCollaboratorPermissionLevel",
    "GITHUB_WORKSPACE",
    "codex-agent-request.json",
    "issues: write",
    "git push origin \"HEAD:refs/heads/${CODEX_BRANCH}\"",
)
FORBIDDEN_SNIPPETS = (
    "pull_request_target:",
    "allow-users: \"*\"",
    "git push origin main",
    "git push --force",
)


def main() -> int:
    if not WORKFLOW.exists():
        print(f"Missing required workflow: {WORKFLOW.relative_to(ROOT)}")
        return 1

    workflow = WORKFLOW.read_text(encoding="utf-8")
    missing = [item for item in REQUIRED_SNIPPETS if item not in workflow]
    forbidden = [item for item in FORBIDDEN_SNIPPETS if item in workflow]
    if missing or forbidden:
        if missing:
            print("Missing workflow contract elements:", *missing, sep="\n- ")
        if forbidden:
            print("Forbidden workflow content found:", *forbidden, sep="\n- ")
        return 1

    print("Codex trusted-runner workflow contract passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
