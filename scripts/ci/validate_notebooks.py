"""Fast, non-executing validation for participant-facing notebooks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from IPython.core.inputtransformer2 import TransformerManager


ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_PARTS = {".git", ".venv", ".quarto", "_site", ".ipynb_checkpoints", "archive", "outputs"}


def notebook_paths() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.ipynb")
        if not any(part in EXCLUDED_PARTS for part in path.parts)
    )


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def main() -> int:
    transformer = TransformerManager()
    failures: list[str] = []

    for path in notebook_paths():
        relative = path.relative_to(ROOT)
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{relative}: invalid notebook JSON ({exc})")
            continue

        if notebook.get("nbformat", 0) < 4:
            failures.append(f"{relative}: expected nbformat 4 or newer")
            continue

        for index, cell in enumerate(notebook.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            source = cell_source(cell)
            if not source.strip():
                continue
            try:
                compile(transformer.transform_cell(source), f"{relative}:cell-{index}", "exec")
            except SyntaxError as exc:
                failures.append(
                    f"{relative}: code cell {index} does not compile after IPython transformation ({exc})"
                )

    if failures:
        print("Notebook validation failed:", *failures, sep="\n- ")
        return 1

    print(f"Validated {len(notebook_paths())} notebooks without executing cells.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
