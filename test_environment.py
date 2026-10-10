"""Verify that the selected workshop environment can run every notebook family."""

from importlib import import_module
from importlib.metadata import version
from pathlib import Path
import sys


PACKAGES = {
    "Beautiful Soup": "bs4",
    "Chemprop": "chemprop",
    "IPython kernel": "ipykernel",
    "JCAMP": "jcamp",
    "JupyterLab": "jupyterlab",
    "Matplotlib": "matplotlib",
    "NISTChemPy": "nistchempy",
    "NumPy": "numpy",
    "Pandas": "pandas",
    "PyArrow": "pyarrow",
    "PyTorch": "torch",
    "RDKit": "rdkit",
    "Requests": "requests",
    "scikit-learn": "sklearn",
    "Seaborn": "seaborn",
    "TensorBoard": "tensorboard",
    "Transformers": "transformers",
}


def main() -> None:
    project_root = Path(__file__).resolve().parent
    print("AVS Workshop Environment Check\n")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    if not sys.version_info[:2] == (3, 11):
        raise RuntimeError("This workshop requires Python 3.11.")
    # ``python`` inside a POSIX virtual environment is commonly a symlink to
    # the managed Python installation. Resolving ``sys.executable`` therefore
    # escapes ``.venv`` on GitHub Actions even though UV selected it correctly.
    # ``sys.prefix`` is the virtual-environment root and is stable on Windows,
    # macOS, and Linux.
    in_project_environment = Path(sys.prefix).resolve() == (project_root / ".venv").resolve()
    if not in_project_environment:
        raise RuntimeError("Launch this check with UV so it uses the .venv inside this workshop folder.")
    print("Environment: project .venv")

    failures = []
    for label, module in PACKAGES.items():
        try:
            import_module(module)
            distribution = {"bs4": "beautifulsoup4", "sklearn": "scikit-learn"}.get(module, module)
            print(f"OK  {label} ({version(distribution)})")
        except Exception as error:  # Report every failed import in one pass.
            failures.append(f"{label}: {error}")
            print(f"FAIL  {label}: {error}")
    if failures:
        raise RuntimeError("\n".join(failures))
    print("\nEnvironment ready.")


if __name__ == "__main__":
    main()
