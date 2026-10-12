"""Execute one notebook in place with the workshop environment."""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1]).resolve()
notebook = nbformat.read(path, as_version=4)
NotebookClient(
    notebook,
    timeout=None,
    kernel_name="python3",
    resources={"metadata": {"path": str(path.parent)}},
).execute()
nbformat.write(notebook, path)
print(f"Executed and saved {path}")
