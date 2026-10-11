"""Fast regression checks for the Issue-triggered workshop runner."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github/workflows/codex-issue-agent.yml").read_text(encoding="utf-8")


class CodexWorkflowContractTests(unittest.TestCase):
    def test_write_collaborators_are_authorized_from_event_identity(self) -> None:
        self.assertIn("const actor = event.sender.login", WORKFLOW)
        self.assertIn("getCollaboratorPermissionLevel", WORKFLOW)
        self.assertIn("new Set(['admin', 'maintain', 'write'])", WORKFLOW)

    def test_requester_branch_is_created_or_reused(self) -> None:
        self.assertIn("const branch = `dev/${actor}`", WORKFLOW)
        self.assertIn("github.rest.git.getRef", WORKFLOW)
        self.assertIn("github.rest.git.createRef", WORKFLOW)

    def test_requests_for_a_branch_are_serialized(self) -> None:
        self.assertIn("group: codex-workshop-${{ github.repository }}-${{ github.actor }}", WORKFLOW)
        self.assertIn("cancel-in-progress: false", WORKFLOW)

    def test_special_commands_are_supported(self) -> None:
        self.assertIn("sync_with_main", WORKFLOW)
        self.assertIn("prepare_merge", WORKFLOW)
        self.assertIn("Create merge pull request on request", WORKFLOW)


if __name__ == "__main__":
    unittest.main()
