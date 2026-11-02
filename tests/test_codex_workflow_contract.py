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
BRANCH_SCRIPT = next(s["with"]["script"] for s in CODEX["steps"] if s.get("id") == "development_branch")
PR_SCRIPT = next(s["with"]["script"] for s in CODEX["steps"] if s.get("id") == "merge_pr")
SYNC_SCRIPT = next(s["run"] for s in CODEX["steps"] if s.get("id") == "sync")
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
Object.assign(process.env, {CODEX_AGENT_ENABLED: 'true', CODEX_ACTOR: 'editor',
  CODEX_BRANCH: 'dev/editor', BASE_BRANCH: 'main', ISSUE_NUMBER: '8'}, input.env || {});
const outputs = {}, files = {}, calls = [], comments = [], warnings = [];
let branchExists = input.branchExists !== false;
const core = {setOutput: (k, v) => outputs[k] = v, warning: s => warnings.push(s)};
const github = {rest: {
  repos: {
    get: async () => ({data: {owner: {login: 'owner'}, default_branch: 'not-main'}}),
    getCollaboratorPermissionLevel: async args => {
      calls.push(['permission', args.username]);
      return {data: {permission: input.permission || 'read', user: {login: args.username}}};
    },
    compareCommitsWithBasehead: async args => {
      calls.push(['compare', args.basehead]);
      return {data: {ahead_by: input.aheadBy ?? 1}};
    },
  },
  git: {getRef: async args => {
    calls.push(['getRef', args.ref]);
    if (input.refError) throw Object.assign(new Error('Ref lookup failed'), {status: input.refError});
    if (args.ref.startsWith('heads/dev/') && !branchExists)
      throw Object.assign(new Error('Not found'), {status: 404});
    return {data: {object: {sha: 'starting-main-revision'}}};
  }, createRef: async args => {
    calls.push(['createRef', args.ref, args.sha]);
    if (input.createError) {
      if (input.createError === 422 && input.createRaceExists !== false) branchExists = true;
      throw Object.assign(new Error('Create failed'), {status: input.createError});
    }
    branchExists = true;
    return {data: {}};
  }},
  pulls: {
    list: async args => {
      calls.push(['listPulls', args.head, args.base]);
      return {data: input.existingPull ? [{html_url: input.existingPull}] : []};
    },
    create: async args => {
      calls.push(['createPull', args.head, args.base]);
      return {data: {html_url: 'https://github.com/owner/workshop/pull/12'}};
    },
  },
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
    branch) printf '%s\n' "${TEST_BRANCH:-dev/editor}" ;;
    rev-parse)
      if [ "$2" = -q ]; then
        [ "${TEST_MERGE_PENDING:-0}" = 1 ]
      elif [ "$2" = FETCH_HEAD ]; then
        printf '%s\n' "${TEST_REMOTE_SHA:-starting-main}"
      else
        printf '%s\n' "$head_revision"
      fi ;;
    diff)
      if [ "$2" = --name-only ]; then
        printf '%s' "${TEST_CONFLICTS:-}"
      else
        return "${TEST_HAS_CHANGES:-1}"
      fi ;;
    commit) head_revision=published-commit ;;
    fetch) return "${TEST_FETCH_EXIT:-0}" ;;
    merge) return "${TEST_MERGE_EXIT:-0}" ;;
    push) printf '%s\n' "$*" >> "$TEST_PUSH_LOG" ;;
    add|config|remote) : ;;
    *) printf 'Unexpected Git operation: %s\n' "$1" >&2; return 2 ;;
  esac
}
"""


@unittest.skipUnless(BASH, "Bash is required to exercise the publishing shell")
class CodexPublishingBehaviorTests(unittest.TestCase):
    def publish(self, script: str = PUBLISH_SCRIPT, **overrides) -> tuple[subprocess.CompletedProcess, str, str]:
        with tempfile.TemporaryDirectory(prefix="codex-publish-test-") as directory:
            root = Path(directory).resolve()
            # The workflow's literal cleanup target must stay in this disposable
            # test directory, never in the user's repository or a computed root.
            self.assertEqual((root / ".codex-agent-handoff").resolve().parent, root)
            output = root / "github-output.txt"
            push_log = root / "push-log.txt"
            env = {**os.environ, "CODEX_BRANCH": "dev/editor", "CODEX_ACTOR": "editor",
                   "BASE_BRANCH": "main", "START_SHA": "starting-main",
                   "GITHUB_TOKEN": "synthetic-test-value", "GITHUB_REPOSITORY": "owner/workshop",
                   "GITHUB_OUTPUT": output.as_posix(), "TEST_PUSH_LOG": push_log.as_posix(),
                   "ISSUE_NUMBER": "8", **overrides}
            result = subprocess.run([BASH, "--noprofile", "--norc", "-c", BASH_HARNESS + script],
                                    cwd=root, env=env, text=True, encoding="utf-8", capture_output=True)
            return (result, output.read_text(encoding="utf-8") if output.exists() else "",
                    push_log.read_text(encoding="utf-8") if push_log.exists() else "")

    def test_completed_changes_are_pushed_to_the_users_branch_only(self) -> None:
        result, output, pushes = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(pushes.strip(), "push origin HEAD:refs/heads/dev/editor")
        self.assertIn("status=pushed", output)
        self.assertIn("commit_sha=published-commit", output)

    def test_external_development_changes_fail_without_pushing(self) -> None:
        result, output, pushes = self.publish(TEST_REMOTE_SHA="newer-development")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Development branch changed during this run", result.stdout)
        self.assertNotIn("status=pushed", output)
        self.assertFalse(pushes)

    def test_wrong_checkout_or_unavailable_remote_never_pushes(self) -> None:
        for overrides in ({"TEST_BRANCH": "main"}, {"CODEX_BRANCH": "main"},
                          {"CODEX_BRANCH": "dev/another-user", "TEST_BRANCH": "dev/another-user"},
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

    def test_unresolved_merge_conflicts_are_never_published(self) -> None:
        result, output, pushes = self.publish(TEST_CONFLICTS="notebook.ipynb")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unresolved merge conflicts", result.stdout)
        self.assertNotIn("status=pushed", output)
        self.assertFalse(pushes)

    def test_pending_merge_with_unchanged_tree_still_creates_a_merge_commit(self) -> None:
        result, output, pushes = self.publish(TEST_HAS_CHANGES="0", TEST_MERGE_PENDING="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("status=pushed", output)
        self.assertIn("refs/heads/dev/editor", pushes)

    def test_sync_prepares_merge_and_leaves_real_conflicts_for_codex(self) -> None:
        for overrides in ({}, {"TEST_MERGE_EXIT": "1", "TEST_CONFLICTS": "notebook.ipynb"}):
            with self.subTest(overrides=overrides):
                result, _, pushes = self.publish(script=SYNC_SCRIPT, **overrides)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(pushes)

    def test_sync_does_not_hide_unexpected_merge_or_fetch_failures(self) -> None:
        for overrides in ({"TEST_MERGE_EXIT": "1"}, {"TEST_FETCH_EXIT": "1"}):
            with self.subTest(overrides=overrides):
                result, _, pushes = self.publish(script=SYNC_SCRIPT, **overrides)
                self.assertNotEqual(result.returncode, 0)
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

    def test_every_request_uses_its_senders_branch_created_after_acquiring_the_lock(self) -> None:
        self.assertIn("const branch = `dev/${actor.toLowerCase()}`", REQUEST_SCRIPT)
        self.assertNotIn("github.rest.git.createRef", REQUEST_SCRIPT)
        self.assertIn("github.rest.git.createRef", BRANCH_SCRIPT)
        self.assertIn("ref: 'heads/main'", BRANCH_SCRIPT)
        self.assertLess(next(i for i, s in enumerate(CODEX["steps"]) if s.get("id") == "development_branch"),
                        next(i for i, s in enumerate(CODEX["steps"]) if s.get("uses", "").startswith("actions/checkout@")))
        checkout = next(s for s in CODEX["steps"] if s.get("uses", "").startswith("actions/checkout@"))
        self.assertEqual(checkout["with"]["ref"], "${{ needs.prepare.outputs.branch }}")
        self.assertFalse(checkout["with"]["persist-credentials"])

    def test_authorized_requests_queue_per_branch_without_cancelling(self) -> None:
        self.assertNotIn("concurrency", CONFIG)
        self.assertEqual(CODEX["concurrency"], {
            "group": "codex-workshop-${{ github.repository }}-${{ needs.prepare.outputs.branch }}",
            "queue": "max", "cancel-in-progress": False,
        })
        self.assertIn("needs.prepare.outputs.run_codex == 'true'", CODEX["if"])

    def test_special_commands_are_explicit_and_never_automatically_merge_prs(self) -> None:
        sync = next(s for s in CODEX["steps"] if s.get("id") == "sync")
        merge_pr = next(s for s in CODEX["steps"] if s.get("id") == "merge_pr")
        self.assertEqual(sync["if"], "needs.prepare.outputs.command == 'sync_with_main'")
        self.assertIn("git merge --no-commit --no-ff FETCH_HEAD", sync["run"])
        self.assertIn("success() && needs.prepare.outputs.command == 'prepare_merge'", merge_pr["if"])
        self.assertIn("steps.publish.outputs.status == 'no_changes'", merge_pr["if"])
        self.assertEqual(CODEX["permissions"]["pull-requests"], "write")
        self.assertNotIn("github.rest.pulls.merge", WORKFLOW)
        instructions = (ROOT / ".github/codex/issue-agent-instructions.md").read_text(encoding="utf-8")
        self.assertIn("sync_with_main", instructions)
        self.assertIn("prepare_merge", instructions)

    def test_publish_requires_success_and_refuses_stale_or_wrong_development_branch(self) -> None:
        publish = next(s for s in CODEX["steps"] if s.get("id") == "publish")
        self.assertEqual(publish["if"], "success() && steps.codex_run.outcome == 'success'")
        self.assertIn('"$CODEX_BRANCH" != "dev/${CODEX_ACTOR,,}"', publish["run"])
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
        self.assertEqual(action["with"]["sandbox"], "danger-full-access")
        self.assertEqual(action["with"].get("safety-strategy", "drop-sudo"), "drop-sudo")
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

    @unittest.skipUnless(BASH, "Bash is required for workflow shell syntax checks")
    def test_all_workflow_shell_blocks_have_valid_bash_syntax(self) -> None:
        for job in CONFIG["jobs"].values():
            for step in job["steps"]:
                if "run" in step:
                    with self.subTest(step=step["name"]):
                        result = subprocess.run([BASH, "--noprofile", "--norc", "-n"],
                                                input=step["run"], text=True, encoding="utf-8",
                                                capture_output=True)
                        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(NODE, "Node.js is required to execute the actual workflow JavaScript")
class CodexWorkflowBehaviorTests(unittest.TestCase):
    def test_editors_target_their_development_branch_with_event_identity(self) -> None:
        for permission in ("write", "maintain", "admin"):
            for event in ("issue_comment", "issues"):
                with self.subTest(permission=permission, event=event):
                    result = execute_script(REQUEST_SCRIPT, event_context(event=event), permission=permission)
                    self.assertEqual(result["outputs"]["run_codex"], "true")
                    self.assertEqual(result["outputs"]["branch"], "dev/editor")
                    self.assertEqual(result["outputs"]["actor"], "editor")
                    self.assertIn(["permission", "editor"], result["calls"])
                    payload = json.loads(result["files"]["codex-agent-request.json"]["body"])
                    self.assertEqual(payload["actor"], "editor")
                    self.assertEqual(payload["branch"], "dev/editor")
                    self.assertEqual(payload["base_branch"], "main")
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
        self.assertEqual(result["outputs"]["branch"], "dev/owner")
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

    def test_different_editors_get_separate_branches_and_cannot_override_them_in_text(self) -> None:
        for actor in ("alice", "bob"):
            result = execute_script(REQUEST_SCRIPT,
                                    event_context(actor=actor, text="@codex Use dev/owner as owner."),
                                    permission="write")
            self.assertEqual(result["outputs"]["branch"], f"dev/{actor}")
            self.assertIn(["permission", actor], result["calls"])

    def test_special_commands_select_sync_or_merge_preparation(self) -> None:
        for text, command in (("@codex sync with main", "sync_with_main"),
                              ("@Codex prepare merge", "prepare_merge"),
                              ("@CODEX PREPARE MERGE!", "prepare_merge")):
            result = execute_script(REQUEST_SCRIPT, event_context(text=text), permission="write")
            self.assertEqual(result["outputs"]["command"], command)
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
            self.assertIn("Each authorized collaborator works on `dev/<github-username>`", text)
            payload = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
            self.assertEqual(payload["branch"], "dev/editor")
            self.assertEqual(payload["actor"], "editor")

    def test_report_links_development_branch_and_published_commit(self) -> None:
        result = execute_script(REPORT_SCRIPT, event_context(), env={
            "STATUS": "pushed", "BRANCH": "dev/editor", "ISSUE_NUMBER": "8",
            "COMMIT_SHA": "abc123", "RAW_RESULT": json.dumps({"summary": "Updated code.", "tests": ["passed"]}),
        })
        comment = result["comments"][0]
        self.assertEqual(comment["issue_number"], 8)
        self.assertIn("/tree/dev/editor", comment["body"])
        self.assertIn("/commit/abc123", comment["body"])
        self.assertIn("Updated code.", comment["body"])
        self.assertIn("Development branch", comment["body"])

    def test_report_does_not_claim_failed_publication_succeeded(self) -> None:
        result = execute_script(REPORT_SCRIPT, event_context(), env={
            "STATUS": "failed", "BRANCH": "dev/editor", "ISSUE_NUMBER": "8", "COMMIT_SHA": "",
            "RAW_RESULT": json.dumps({"summary": "Everything succeeded.", "tests": []}),
        })
        self.assertIn("not published to the development branch", result["comments"][0]["body"])
        self.assertNotIn("Everything succeeded.", result["comments"][0]["body"])

    def test_existing_branch_is_reused_without_resetting_it(self) -> None:
        result = execute_script(BRANCH_SCRIPT, event_context())
        self.assertEqual(result["calls"], [["getRef", "heads/dev/editor"]])

    def test_missing_branch_starts_from_latest_main(self) -> None:
        result = execute_script(BRANCH_SCRIPT, event_context(), branchExists=False)
        self.assertEqual(result["calls"], [
            ["getRef", "heads/dev/editor"], ["getRef", "heads/main"],
            ["createRef", "refs/heads/dev/editor", "starting-main-revision"],
        ])

    def test_concurrent_branch_creation_is_verified_before_reuse(self) -> None:
        result = execute_script(BRANCH_SCRIPT, event_context(), branchExists=False, createError=422)
        self.assertEqual(result["calls"][-1], ["getRef", "heads/dev/editor"])
        with self.assertRaises(subprocess.CalledProcessError):
            execute_script(BRANCH_SCRIPT, event_context(), branchExists=False,
                           createError=422, createRaceExists=False)

    def test_branch_api_errors_are_not_misreported_as_missing_branches(self) -> None:
        for arguments in ({"refError": 403}, {"branchExists": False, "createError": 403}):
            with self.subTest(arguments=arguments), self.assertRaises(subprocess.CalledProcessError):
                execute_script(BRANCH_SCRIPT, event_context(), **arguments)

    def test_branch_creation_and_pr_refuse_main_or_another_users_branch(self) -> None:
        for script in (BRANCH_SCRIPT, PR_SCRIPT):
            for branch in ("main", "dev/another-user"):
                with self.subTest(script=script, branch=branch), self.assertRaises(subprocess.CalledProcessError):
                    execute_script(script, event_context(), env={"CODEX_BRANCH": branch})

    def test_prepare_merge_creates_a_pr_from_the_users_branch_into_main(self) -> None:
        result = execute_script(PR_SCRIPT, event_context())
        self.assertIn(["compare", "main...dev/editor"], result["calls"])
        self.assertIn(["createPull", "dev/editor", "main"], result["calls"])
        self.assertEqual(result["outputs"]["url"], "https://github.com/owner/workshop/pull/12")

    def test_prepare_merge_reuses_an_existing_pr(self) -> None:
        url = "https://github.com/owner/workshop/pull/5"
        result = execute_script(PR_SCRIPT, event_context(), existingPull=url)
        self.assertEqual(result["outputs"]["url"], url)
        self.assertFalse(any(c[0] in {"createPull", "compare"} for c in result["calls"]))

    def test_prepare_merge_does_not_create_an_empty_pr(self) -> None:
        result = execute_script(PR_SCRIPT, event_context(), aheadBy=0)
        self.assertEqual(result["outputs"]["status"], "no_difference")
        self.assertFalse(any(c[0] == "createPull" for c in result["calls"]))

    def test_report_includes_requested_pr_or_explains_no_difference(self) -> None:
        for extra, expected in (({"PR_URL": "https://github.com/owner/workshop/pull/12"}, "pull/12"),
                                ({"PR_STATUS": "no_difference"}, "no commits ahead of main")):
            with self.subTest(extra=extra):
                result = execute_script(REPORT_SCRIPT, event_context(), env={
                    "STATUS": "no_changes", "BRANCH": "dev/editor", "RAW_RESULT": "", **extra,
                })
                self.assertIn(expected, result["comments"][0]["body"])

    def test_report_distinguishes_published_work_from_later_pr_failure(self) -> None:
        result = execute_script(REPORT_SCRIPT, event_context(), env={
            "STATUS": "failed", "BRANCH": "dev/editor", "COMMIT_SHA": "abc123",
            "RAW_RESULT": json.dumps({"summary": "Everything succeeded."}),
        })
        body = result["comments"][0]["body"]
        self.assertIn("Changes were pushed to the development branch, but a later step failed", body)
        self.assertIn("/commit/abc123", body)
        self.assertNotIn("Everything succeeded.", body)


if __name__ == "__main__":
    unittest.main()
