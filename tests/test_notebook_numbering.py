"""Keep workshop filenames, first-cell titles, and participant references aligned."""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "01_boiling_point": ("BP", 3),
    "02_ir_functional_groups": ("IR", 3),
    "03_htem_conductivity": ("HTEM", 3),
}


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def module_notebooks() -> list[Path]:
    return sorted(
        path
        for module in MODULES
        for folder in (ROOT / module, ROOT / module / "examples")
        for path in folder.glob("*.ipynb")
    )


class NotebookNumberingTests(unittest.TestCase):
    def test_main_sequences_are_contiguous_from_zero(self):
        for module, (_, count) in MODULES.items():
            with self.subTest(module=module):
                prefixes = sorted(path.name.split("_", 1)[0]
                                  for path in (ROOT / module).glob("*.ipynb"))
                self.assertEqual(prefixes, [f"{index:02d}" for index in range(count)])

    def test_filenames_have_one_sequence_number_and_the_correct_module(self):
        for path in module_notebooks():
            relative = path.relative_to(ROOT)
            marker, _ = MODULES[relative.parts[0]]
            with self.subTest(notebook=str(relative)):
                if relative.as_posix() == "01_boiling_point/01_Ten_Steps_in_Prompts.ipynb":
                    continue  # The BP-specific prompting guide keeps its descriptive filename.
                self.assertRegex(path.name, rf"^\d{{2}}[a-z]?_{marker}_[A-Za-z].*\.ipynb$")
                if len(relative.parts) == 3:
                    if relative.parts[:2] == ("01_boiling_point", "examples"):
                        self.assertRegex(path.name, r"^0[1-4]_BP_")
                        continue
                    if relative.parts[:2] == ("02_ir_functional_groups", "examples") and path.name in {
                        "01_IR_Vibe_Coding_Data_Carpentry.ipynb", "03_IR_Compare_Carpentry.ipynb"
                    }:
                        continue  # Completed references retain their original step numbers.
                    if relative.parts[:2] == ("03_htem_conductivity", "examples") and path.name in {
                        "02_HTEM_Process_Emulator_Vibe_Coding.ipynb", "03_HTEM_Compare_Architectures.ipynb"
                    }:
                        continue
                    self.assertEqual(relative.parts[1], "examples")
                    prefix = path.name.split("_", 1)[0]
                    self.assertRegex(prefix, r"^\d{2}[a-z]$")
                    self.assertTrue(list(path.parent.parent.glob(f"{prefix[:2]}_*.ipynb")))

    def test_first_cell_title_matches_the_filename_number(self):
        for path in module_notebooks():
            with self.subTest(notebook=str(path.relative_to(ROOT))):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                first = notebook["cells"][0]
                self.assertEqual(first["cell_type"], "markdown")
                prefix = path.name.split("_", 1)[0]
                self.assertTrue(source_text(first).startswith(f"# {prefix} · "))

    def test_boiling_point_references_use_existing_notebook_names(self):
        module = ROOT / "01_boiling_point"
        paths = [
            ROOT / "README.md", ROOT / "00_Workshop_Roadmap.ipynb",
            ROOT / "SETUP_GUIDE.md", ROOT / "AGENTS.md",
            *(ROOT / "docs").rglob("*.md"),
            *(ROOT / name / "README.md" for name in MODULES),
            module / "README.md", module / "examples/README.md",
            *module.glob("*.txt"), *module.glob("examples/*.txt"),
            *module.glob("*.ipynb"), *module.glob("examples/*.ipynb"),
            ROOT / "scripts/prepare_notebook_presentation.py",
            ROOT / "scratch/build_bp_one_shot_notebook.py",
            ROOT / "scratch/build_critiques_and_assumptions_pdf.py",
            ROOT / "scratch/update_segmented_training_notebooks.py",
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".ipynb":
                text = "\n".join(source_text(cell) for cell in json.loads(text)["cells"])
            for name in re.findall(r"\d{2}_BP_[A-Za-z0-9_]+\.ipynb", text):
                with self.subTest(document=str(path.relative_to(ROOT)), reference=name):
                    self.assertTrue((module / name).is_file() or (module / "examples" / name).is_file(),
                                    f"Broken BP reference: {name}")

    def test_readme_sequence_labels_use_the_actual_notebook_numbers(self):
        root_text = (ROOT / "README.md").read_text(encoding="utf-8")
        for module, (_, count) in MODULES.items():
            root_section = root_text.split(f"### {module}\n", 1)[1].split("\n##", 1)[0]
            module_text = (ROOT / module / "README.md").read_text(encoding="utf-8")
            expected = [f"{index:02d}" for index in range(count)]
            for name, text in (("root README", root_section), ("module README", module_text)):
                with self.subTest(module=module, document=name):
                    self.assertEqual(re.findall(r"(?m)^- \[(\d{2}) — ", text), expected)
                    for label, target in re.findall(r"\[(\d{2}[a-z]?) — [^\]]+\]\(([^)]+)\)", text):
                        self.assertEqual(label, Path(target).name.split("_", 1)[0])

    def test_roadmap_lists_each_current_notebook_in_the_participant_path(self):
        notebook = json.loads((ROOT / "00_Workshop_Roadmap.ipynb").read_text(encoding="utf-8"))
        text = "\n".join(source_text(cell) for cell in notebook["cells"])
        # HTEM examples/comparison deliberately stay out of the participant roadmap.
        for module, count in (("01_boiling_point", 3), ("02_ir_functional_groups", 3),
                              ("03_htem_conductivity", 3)):
            references = re.findall(rf"\]\(({module}/[^)]+\.ipynb)\)", text)
            with self.subTest(module=module):
                self.assertEqual([Path(path).name.split("_", 1)[0] for path in references],
                                 [f"{index:02d}" for index in range(count)])
                for path in references:
                    self.assertTrue((ROOT / path).is_file())

    def test_participant_documentation_local_links_exist(self):
        paths = [ROOT / "README.md", ROOT / "SETUP_GUIDE.md",
                 ROOT / "docs/NOTEBOOK_PRESENTATION.md",
                 ROOT / "03_htem_conductivity/TECHNICAL_NOTES.md",
                 ROOT / "01_boiling_point/examples/README.md",
                 ROOT / "02_ir_functional_groups/examples/README.md",
                 ROOT / "03_htem_conductivity/examples/README.md",
                 *(ROOT / module / "README.md" for module in MODULES)]
        for path in paths:
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if target.startswith(("https://", "http://", "#")):
                    continue
                with self.subTest(document=str(path.relative_to(ROOT)), target=target):
                    self.assertTrue((path.parent / target.split("#", 1)[0]).exists())


if __name__ == "__main__":
    unittest.main()
