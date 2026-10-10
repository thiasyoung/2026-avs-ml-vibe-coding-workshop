"""Contract checks for the Issue-triggered collaborator Codex workflow."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github/workflows/codex-issue-agent.yml").read_text(encoding="utf-8")


class CodexWorkflowContractTests(unittest.TestCase):
    def test_uses_event_identity_and_repository_write_permission(self) -> None:
        self.assertIn("const actor = payload.sender?.login", WORKFLOW)
        self.assertIn("getCollaboratorPermissionLevel", WORKFLOW)
        self.assertIn("['admin', 'maintain', 'write']", WORKFLOW)

    def test_creates_or_reuses_a_per_user_development_branch(self) -> None:
        self.assertIn("const branch = actor ? `dev/${actor}` : '';", WORKFLOW)
        self.assertIn("Create dev branch only when it does not exist", WORKFLOW)
        self.assertIn("refs/heads/${branch}", WORKFLOW)
        self.assertIn("Reusing existing branch ${branch}", WORKFLOW)

    def test_serializes_requests_per_user_but_not_globally(self) -> None:
        self.assertIn("group: codex-dev-${{ github.repository }}-${{ github.actor }}", WORKFLOW)
        self.assertIn("cancel-in-progress: false", WORKFLOW)

    def test_supports_sync_and_prepare_merge_without_direct_main_push(self) -> None:
        self.assertIn("sync with main", WORKFLOW)
        self.assertIn("prepare merge", WORKFLOW)
        self.assertIn("Synchronize collaborator branch with main", WORKFLOW)
        self.assertIn("Create or update a pull request into main", WORKFLOW)
        self.assertNotIn("git push origin main", WORKFLOW)
        self.assertNotIn("git push --force", WORKFLOW)


if __name__ == "__main__":
    unittest.main()
