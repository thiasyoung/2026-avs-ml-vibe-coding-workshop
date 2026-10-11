"""Build a Codex prompt from trusted instructions and an Issue artifact.

This script deliberately treats Issue fields as serialized data rather than shell text.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, default=Path("codex-agent-request.json"))
    parser.add_argument(
        "--instructions", type=Path, default=Path(".github/codex/issue-agent-instructions.md")
    )
    parser.add_argument("--output", type=Path, default=Path(".codex-agent-prompt.md"))
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    instructions = args.instructions.read_text(encoding="utf-8")
    serialized_request = json.dumps(request, indent=2, ensure_ascii=False)
    args.output.write_text(
        f"{instructions}\n```json\n{serialized_request}\n```\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
