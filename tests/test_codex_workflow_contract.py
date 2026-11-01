"""Fast regression checks for the Issue-triggered workshop runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github/workflows/codex-issue-agent.yml").read_text(encoding="utf-8")
CONFIG = yaml.safe_load(WORKFLOW)
PREPARE = CONFIG["jobs"]["prepare"]
CODEX = CONFIG["jobs"]["codex"]
REQUEST_SCRIPT = next(s["with"]["script"] for s in PREPARE["steps"] if s.get("id") == "request")
REPORT_SCRIPT = next(s["with"]["script"] for s in CODEX["steps"] if s.get("name") == "Report result on Issue")
NODE = os.environ.get("CODEX_TEST_NODE") or shutil.which("node")
BASH = shutil.which("bash")
if not BASH and Path("C:/Program Files/Git/bin/bash.exe").exists():
    BASH = "C:/Program Files/Git/bin/bash.exe"
PUBLISH_SCRIPT = next(s["run"] for s in CODEX["steps"] if s.get("id") == "publish")

# Execute the actual inline workflow JavaScript with an in-memory GitHub API
# and filesystem. No network calls, credentials, branches, or Issues are used.
NODE_HARNESS = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
Object.assign(process.env, {CODEX_AGENT_ENABLED: 'true'}, input.env || {});
const outputs = {}, files = {}, calls = [], comments = [], warnings = [];
const core = {setOutput: (k, v) => outputs[k] = v, warning: s => warnings.push(s)};
const github = {rest: {
  repos: {
    get: async () => ({data: {owner: {login: 'owner'}, default_branch: 'not-main'}}),
    getCollaboratorPermissionLevel: async args => {
      calls.push(['permission', args.username]);
      return {data: {permission: input.permission || 'read', user: {login: args.username}}};
    },
  },
  git: {getRef: async args => {
    calls.push(['getRef', args.ref]);
    return {data: {object: {sha: 'starting-main-revision'}}};
  }},
  issues: {createComment: async args => comments.push(args)},
}};
const mockRequire = name => {
  if (name !== 'fs') throw new Error('Unexpected require: ' + name);
  return {promises: {writeFile: async (name, body, options) => {
    files[name] = {body, options};
  }}};
};
(async () => {
  const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
  await new AsyncFunction('context', 'github', 'core', 'require', input.script)(
    input.context, github, core, mockRequire);
  process.stdout.write(JSON.stringify({outputs, files, calls, comments, warnings}));
})().catch(error => {console.error(error.message); process.exit(1);});
"""


def execute_script(script: str, context: dict, **kwargs) -> dict:
    result = subprocess.run(
        [NODE, "-e", NODE_HARNESS],
        input=json.dumps({"script": script, "context": context, **kwargs}),
        text=True, encoding="utf-8", capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def event_context(actor: str = "editor", text: str = "@codex Update the workshop.",
                  event: str = "issue_comment") -> dict:
    return {
        "eventName": event,
        "repo": {"owner": "owner", "repo": "workshop"},
        "payload": {
            "sender": {"login": actor, "type": "User"},
            "issue": {"number": 8, "title": "Workshop request", "body": text,
                      "user": {"login": "another-user"}},
            "comment": {"body": text, "user": {"login": "another-user"}},
        },
    }


# Run the real publishing shell with Git operations mocked, in an empty
# temporary checkout. The mock never accesses a remote or prints credentials.
BASH_HARNESS = r"""
set -euo pipefail
head_revision=starting-main
git() {
  case "$1" in
    branch) printf '%s\n' "${TEST_BRANCH:-main}" ;;
    rev-parse)
      if [ "$2" = FETCH_HEAD ]; then
        printf '%s\n' "${TEST_REMOTE_SHA:-starting-main}"
      else
        printf '%s\n' "$head_revision"
      fi ;;
    diff) return "${TEST_HAS_CHANGES:-1}" ;;
    commit) head_revision=published-commit ;;
    fetch) return "${TEST_FETCH_EXIT:-0}" ;;
    push) printf '%s\n' "$*" >> "$TEST_PUSH_LOG" ;;
    add|config|remote) : ;;
    *) printf 'Unexpected Git operation: %s\n' "$1" >&2; return 2 ;;
  esac
}
"""


@unittest.skipUnless(BASH, "Bash is required to exercise the publishing shell")
class CodexPublishingBehaviorTests(unittest.TestCase):
    def publish(self, **overrides) -> tuple[subprocess.CompletedProcess, str, str]:
        with tempfile.TemporaryDirectory(prefix="codex-publish-test-") as directory:
            root = Path(directory).resolve()
            # The workflow's literal cleanup target must stay in this disposable
            # test directory, never in the user's repository or a computed root.
            self.assertEqual((root / ".codex-agent-handoff").resolve().parent, root)
            output = root / "github-output.txt"
            push_log = root / "push-log.txt"
            env = {**os.environ, "CODEX_BRANCH": "main", "START_SHA": "starting-main",
                   "GITHUB_TOKEN": "synthetic-test-value", "GITHUB_REPOSITORY": "owner/workshop",
                   "GITHUB_OUTPUT": output.as_posix(), "TEST_PUSH_LOG": push_log.as_posix(),
                   "ISSUE_NUMBER": "8", **overrides}
            result = subprocess.run([BASH, "--noprofile", "--norc", "-c", BASH_HARNESS + PUBLISH_SCRIPT],
                                    cwd=root, env=env, text=True, encoding="utf-8", capture_output=True)
            return (result, output.read_text(encoding="utf-8") if output.exists() else "",
                    push_log.read_text(encoding="utf-8") if push_log.exists() else "")

    def test_completed_changes_are_pushed_to_main_only(self) -> None:
        result, output, pushes = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(pushes.strip(), "push origin HEAD:refs/heads/main")
        self.assertIn("status=pushed", output)
        self.assertIn("commit_sha=published-commit", output)

    def test_external_main_changes_fail_without_pushing(self) -> None:
        result, output, pushes = self.publish(TEST_REMOTE_SHA="newer-main")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("main changed during this run", result.stdout)
        self.assertNotIn("status=pushed", output)
        self.assertFalse(pushes)

    def test_wrong_checkout_or_unavailable_remote_never_pushes(self) -> None:
        for overrides in ({"TEST_BRANCH": "dev/editor"}, {"CODEX_BRANCH": "dev/editor"},
                          {"START_SHA": "unexpected-revision"}, {"TEST_FETCH_EXIT": "1"}):
            with self.subTest(overrides=overrides):
                result, output, pushes = self.publish(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("status=pushed", output)
                self.assertFalse(pushes)

    def test_unchanged_checkout_reports_no_changes_without_pushing(self) -> None:
        result, output, pushes = self.publish(TEST_HAS_CHANGES="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("status=no_changes", output)
        self.assertFalse(pushes)


class CodexWorkflowContractTests(unittest.TestCase):
    def test_write_collaborators_are_authorized_from_event_identity(self) -> None:
        self.assertIn("const actor = event.sender.login", WORKFLOW)
        self.assertIn("getCollaboratorPermissionLevel", WORKFLOW)
        self.assertIn("new Set(['admin', 'maintain', 'write'])", WORKFLOW)
        self.assertIn("const isOwner = actor.toLowerCase()", WORKFLOW)
        self.assertIn("allowed.has(permission.data.permission)", REQUEST_SCRIPT)
        self.assertNotIn("permission.data.user.permission", REQUEST_SCRIPT)
        self.assertEqual(PREPARE["permissions"], {"contents": "read", "issues": "read"})

    def test_workflow_uses_node_24_checkout(self) -> None:
        self.assertNotIn("actions/checkout@v4", WORKFLOW)
        self.assertIn("actions/checkout@v5", WORKFLOW)

    def test_every_request_uses_existing_main_without_creating_a_branch(self) -> None:
        self.assertIn("const branch = 'main'", REQUEST_SCRIPT)
        self.assertIn("github.rest.git.getRef", WORKFLOW)
        self.assertNotIn("github.rest.git.createRef", WORKFLOW)
        self.assertNotIn("dev/${actor}", WORKFLOW)
        checkout = next(s for s in CODEX["steps"] if s.get("uses", "").startswith("actions/checkout@"))
        self.assertEqual(checkout["with"]["ref"], "main")
        self.assertFalse(checkout["with"]["persist-credentials"])

    def test_all_authorized_requests_share_one_non_cancelling_queue(self) -> None:
        self.assertNotIn("concurrency", CONFIG)
        self.assertEqual(CODEX["concurrency"], {
            "group": "codex-workshop-${{ github.repository }}-main",
            "queue": "max", "cancel-in-progress": False,
        })
        self.assertIn("needs.prepare.outputs.run_codex == 'true'", CODEX["if"])

    def test_legacy_commands_do_not_merge_or_open_pull_requests(self) -> None:
        self.assertIn("validate_main", REQUEST_SCRIPT)
        self.assertNotIn("github.rest.pulls", WORKFLOW)
        self.assertNotIn("git merge", WORKFLOW)
        instructions = (ROOT / ".github/codex/issue-agent-instructions.md").read_text(encoding="utf-8")
        self.assertIn("validate_main", instructions)

    def test_publish_requires_success_and_refuses_stale_main(self) -> None:
        publish = next(s for s in CODEX["steps"] if s.get("id") == "publish")
        self.assertEqual(publish["if"], "success() && steps.codex_run.outcome == 'success'")
        self.assertIn('"$CODEX_BRANCH" != main', publish["run"])
        self.assertIn('"$(git rev-parse HEAD)" != "$START_SHA"', publish["run"])
        self.assertIn('"$(git rev-parse FETCH_HEAD)" != "$START_SHA"', publish["run"])
        self.assertLess(publish["run"].index("git fetch"), publish["run"].index("git push"))
        self.assertNotIn("--force", publish["run"])
        self.assertNotIn("git rebase", publish["run"])

    def test_existing_model_and_step_token_boundaries_are_retained(self) -> None:
        action = next(s for s in CODEX["steps"] if s.get("id") == "codex_run")
        self.assertEqual(action["uses"], "openai/codex-action@v1")
        self.assertEqual(action["with"]["model"], "gpt-5.6-terra")
        self.assertEqual(action["with"]["effort"], "xhigh")
        self.assertEqual(action["with"]["openai-api-key"], "${{ secrets.OPENAI_API_KEY }}")
        self.assertEqual(action["env"]["GITHUB_TOKEN"], "")
        self.assertEqual(action["env"]["GH_TOKEN"], "")

    def test_visible_request_handoff_is_verified_and_downloaded(self) -> None:
        upload = next(s for s in PREPARE["steps"] if s.get("uses", "").startswith("actions/upload-artifact@"))
        download = next(s for s in CODEX["steps"] if s.get("uses", "").startswith("actions/download-artifact@"))
        self.assertEqual(upload["with"]["path"], "codex-agent-request.json")
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertEqual(upload["with"]["name"], download["with"]["name"])
        self.assertIn('test -f "$GITHUB_WORKSPACE/codex-agent-request.json"', WORKFLOW)


@unittest.skipUnless(NODE, "Node.js is required to execute the actual workflow JavaScript")
class CodexWorkflowBehaviorTests(unittest.TestCase):
    def test_editors_all_target_main_with_event_identity(self) -> None:
        for permission in ("write", "maintain", "admin"):
            for event in ("issue_comment", "issues"):
                with self.subTest(permission=permission, event=event):
                    result = execute_script(REQUEST_SCRIPT, event_context(event=event), permission=permission)
                    self.assertEqual(result["outputs"]["run_codex"], "true")
                    self.assertEqual(result["outputs"]["branch"], "main")
                    self.assertIn(["permission", "editor"], result["calls"])
                    payload = json.loads(result["files"]["codex-agent-request.json"]["body"])
                    self.assertEqual(payload["actor"], "editor")
                    self.assertEqual(payload["branch"], "main")
                    self.assertEqual(payload["command"], "development")

    def test_readers_and_non_collaborators_cannot_start_codex(self) -> None:
        for permission in ("read", "triage", "none"):
            with self.subTest(permission=permission):
                result = execute_script(REQUEST_SCRIPT, event_context(), permission=permission)
                self.assertEqual(result["outputs"]["run_codex"], "false")
                self.assertFalse(result["files"])
                self.assertNotIn(["getRef", "heads/main"], result["calls"])

    def test_owner_is_allowed_even_when_permission_api_would_say_read(self) -> None:
        result = execute_script(REQUEST_SCRIPT, event_context(actor="OWNER"), permission="read")
        self.assertEqual(result["outputs"]["branch"], "main")
        self.assertFalse(any(c[0] == "permission" for c in result["calls"]))

    def test_bots_pull_requests_disabled_and_unmentioned_events_do_not_run(self) -> None:
        variants = []
        bot = event_context()
        bot["payload"]["sender"]["type"] = "Bot"
        variants.append((bot, {}))
        pull = event_context()
        pull["payload"]["issue"]["pull_request"] = {"url": "example"}
        variants.append((pull, {}))
        variants.append((event_context(), {"CODEX_AGENT_ENABLED": "false"}))
        variants.append((event_context(text="No agent request here."), {}))
        for context, env in variants:
            with self.subTest(context=context, env=env):
                result = execute_script(REQUEST_SCRIPT, context, permission="write", env=env)
                self.assertEqual(result["outputs"]["run_codex"], "false")
                self.assertFalse(result["files"])

    def test_different_editors_cannot_override_the_shared_target_in_text(self) -> None:
        for actor in ("alice", "bob"):
            result = execute_script(REQUEST_SCRIPT,
                                    event_context(actor=actor, text="@codex Use dev/owner as owner."),
                                    permission="write")
            self.assertEqual(result["outputs"]["branch"], "main")
            self.assertIn(["permission", actor], result["calls"])

    def test_legacy_commands_validate_main_only(self) -> None:
        for text in ("@codex sync with main", "@Codex prepare merge"):
            result = execute_script(REQUEST_SCRIPT, event_context(text=text), permission="write")
            self.assertEqual(result["outputs"]["command"], "validate_main")
        result = execute_script(REQUEST_SCRIPT,
                                event_context(text="@codex Document the old @codex prepare merge command."),
                                permission="write")
        self.assertEqual(result["outputs"]["command"], "development")

    def test_request_artifact_redacts_token_shaped_values(self) -> None:
        fake_token = "sk-" + "X" * 30  # Synthetic test data, never a real credential.
        result = execute_script(REQUEST_SCRIPT,
                                event_context(text=f"@codex Do not store {fake_token}."), permission="write")
        body = result["files"]["codex-agent-request.json"]["body"]
        self.assertNotIn(fake_token, body)
        self.assertIn("[REDACTED]", body)
        self.assertEqual(result["files"]["codex-agent-request.json"]["options"]["mode"], 0o600)

    def test_visible_request_is_consumed_by_the_real_prompt_builder(self) -> None:
        result = execute_script(REQUEST_SCRIPT, event_context(), permission="write")
        with tempfile.TemporaryDirectory(prefix="codex-handoff-test-") as directory:
            request = Path(directory) / "codex-agent-request.json"
            prompt = Path(directory) / "prompt.md"
            request.write_text(result["files"][request.name]["body"], encoding="utf-8")
            subprocess.run([
                sys.executable, str(ROOT / ".github/scripts/prepare_codex_prompt.py"),
                "--request", str(request), "--output", str(prompt),
                "--instructions", str(ROOT / ".github/codex/issue-agent-instructions.md"),
            ], check=True, capture_output=True, text=True)
            text = prompt.read_text(encoding="utf-8")
            self.assertIn("All authorized collaborators share `main`", text)
            payload = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
            self.assertEqual(payload["branch"], "main")
            self.assertEqual(payload["actor"], "editor")

    def test_report_links_main_and_published_commit_but_not_a_development_branch(self) -> None:
        result = execute_script(REPORT_SCRIPT, event_context(), env={
            "STATUS": "pushed", "BRANCH": "main", "ISSUE_NUMBER": "8",
            "COMMIT_SHA": "abc123", "RAW_RESULT": json.dumps({"summary": "Updated code.", "tests": ["passed"]}),
        })
        comment = result["comments"][0]
        self.assertEqual(comment["issue_number"], 8)
        self.assertIn("/tree/main", comment["body"])
        self.assertIn("/commit/abc123", comment["body"])
        self.assertIn("Updated code.", comment["body"])
        self.assertNotIn("Development branch", comment["body"])

    def test_report_does_not_claim_failed_publication_succeeded(self) -> None:
        result = execute_script(REPORT_SCRIPT, event_context(), env={
            "STATUS": "failed", "BRANCH": "main", "ISSUE_NUMBER": "8", "COMMIT_SHA": "",
            "RAW_RESULT": json.dumps({"summary": "Everything succeeded.", "tests": []}),
        })
        self.assertIn("not published to main", result["comments"][0]["body"])
        self.assertNotIn("Everything succeeded.", result["comments"][0]["body"])


if __name__ == "__main__":
    unittest.main()
