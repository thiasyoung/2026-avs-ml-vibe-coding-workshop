"""Build a Codex prompt from trusted instructions and an Issue artifact.

This script deliberately treats Issue fields as serialized data rather than shell text.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    request_path = Path(".codex-agent-request.json")
    instructions_path = Path(".github/codex/issue-agent-instructions.md")
    output_path = Path(".codex-agent-prompt.md")

    request = json.loads(request_path.read_text(encoding="utf-8"))
    instructions = instructions_path.read_text(encoding="utf-8")
    serialized_request = json.dumps(request, indent=2, ensure_ascii=False)
    output_path.write_text(
        f"{instructions}\n```json\n{serialized_request}\n```\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
