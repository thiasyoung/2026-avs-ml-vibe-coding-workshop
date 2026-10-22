"""Preserve the currently working full HTEM run before feature experiments."""
from pathlib import Path
import shutil

module = Path(__file__).resolve().parents[1] / "03_htem_conductivity"
archive = module / "archive/pre_feature_study_2026_10_21"
if archive.exists():
    raise RuntimeError("Feature-study snapshot already exists; do not overwrite it.")
archive.mkdir(parents=True)
for name in ["htem_prepared", "htem_architectures"]:
    shutil.copytree(module / "outputs" / name, archive / name)
for path in [*module.glob("*.py"), *module.glob("*.ipynb"), module / "README.md",
             *module.glob("data/*reference*"), *(module / "examples").glob("*.ipynb")]:
    destination = archive / "source" / path.relative_to(module)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
print("Preserved preparation, models, metrics, populated notebooks and source:", archive)
