"""Recoverable snapshot before adding reference DOS features; never move splits."""
from pathlib import Path
import shutil

module = Path(__file__).resolve().parents[1] / "03_htem_conductivity"
archive = module / "archive/pre_source_dos_2026_10_21"
if archive.exists():
    raise RuntimeError("Pre-DOS snapshot already exists; do not overwrite it.")
archive.mkdir(parents=True)
for name in ["htem_prepared", "htem_architectures"]:
    shutil.copytree(module / "outputs" / name, archive / name)
for path in [*module.glob("*.py"), *module.glob("*.ipynb"), module / "README.md",
             *module.glob("data/*reference*"), *(module / "examples").glob("*.ipynb")]:
    relative = path.relative_to(module)
    destination = archive / "source" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
print("Preserved pre-DOS preparation, five models, metrics, populated notebooks, and source:", archive)
