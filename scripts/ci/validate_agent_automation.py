"""Static invariants for the constrained collaborator-development automation."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/codex-issue-agent.yml"
REQUIRED_SNIPPETS = (
    "openai/codex-action@v1",
    "actions/github-script@v8",
    "actions/upload-artifact@v6",
    "actions/download-artifact@v6",
    "OPENAI_API_KEY",
    "model: gpt-5.6-terra",
    "effort: xhigh",
    "CODEX_AGENT_ENABLED",
    'permission-profile: ":workspace"',
    "safety-strategy: drop-sudo",
    "GITHUB_TOKEN: \"\"",
    "getCollaboratorPermissionLevel",
    "['admin', 'maintain', 'write']",
    "dev/${actor}",
    "Create dev branch only when it does not exist",
    "codex-dev-${{ github.repository }}-${{ github.actor }}",
    "sync with main",
    "prepare merge",
    "allow-users: ${{ github.actor }}",
    "path === '.git' || path.startsWith('.git/') || path.startsWith('.github/')",
    "GITHUB_WORKSPACE",
    "codex-agent-request.json",
    "pull-requests: write",
    "issues: write",
)
FORBIDDEN_SNIPPETS = (
    "pull_request_target:",
    "allow-users: \"*\"",
    "safety-strategy: unsafe",
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
            print("Missing safety invariants:", *missing, sep="\n- ")
        if forbidden:
            print("Forbidden workflow content found:", *forbidden, sep="\n- ")
        return 1

    print("Codex automation static safety invariants passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
