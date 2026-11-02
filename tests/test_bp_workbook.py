"""Guard the prompt-driven BP path and reference paths without training models."""

import ast
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "01_boiling_point"
EXAMPLES = MODULE / "examples"
REFERENCES = (
    "01_BP_Chemprop_GNN.ipynb",
    "02_BP_Foundation_Model.ipynb",
    "03_BP_One_Shot_Vibe_Coding.ipynb",
    "04_BP_One_Shot_Workflow.ipynb",
)


def read_notebook(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source(cell):
    text = cell.get("source", "")
    return "".join(text) if isinstance(text, list) else text


class BoilingPointWorkbookTests(unittest.TestCase):
    def test_main_path_and_optional_reference_library(self):
        self.assertEqual(sorted(p.name for p in MODULE.glob("*.ipynb")), [
            "00_BP_Get_Data_NISTChemPy_v2.ipynb",
            "01_Ten_Steps_in_Prompts.ipynb",
            "02_BP_Vibe_Coding_Workbook.ipynb",
        ])
        self.assertEqual(sorted(p.name for p in EXAMPLES.glob("*.ipynb")), list(REFERENCES))
        self.assertTrue((EXAMPLES / "README.md").is_file())
        self.assertFalse((MODULE / "extras").exists())
        for name in ("one_shot_prompt_1.txt", "one_shot_prompt_2.txt"):
            with self.subTest(prompt=name):
                self.assertTrue((EXAMPLES / name).is_file())
                self.assertFalse((MODULE / name).exists())
                self.assertIn(f"]({name})", (EXAMPLES / "README.md").read_text(encoding="utf-8"))

    def test_workbook_contains_one_copyable_prompt_instead_of_prescribed_exercises(self):
        notebook = read_notebook(MODULE / "02_BP_Vibe_Coding_Workbook.ipynb")
        populated = [c for c in notebook["cells"] if source(c).strip()]
        self.assertEqual(len(populated), 1)  # Empty Markdown cells are harmless editor state.
        self.assertEqual(populated[0]["cell_type"], "markdown")
        text = source(populated[0])
        prompts = re.findall(r"```text\n(.*?)\n```", text, flags=re.DOTALL)
        self.assertEqual(len(prompts), 1)
        prompt = prompts[0]
        self.assertIn("SMILES", prompt)
        self.assertIn("kelvin", prompt)
        self.assertIn("Build and populate the 02_BP_Vibe_Coding_Workbook.ipynb", prompt)
        self.assertIn("01_boiling_point/01_Ten_Steps_in_Prompts.ipynb", prompt)
        self.assertIn("ten step names and order", prompt)
        self.assertIn("test set only after the model is frozen", prompt)
        self.assertIn("execute and debug", prompt)
        self.assertIn("rather than inventing results", prompt)
        self.assertNotIn("## Step", text)

    def test_documented_template_and_prompt_destinations_match_the_module_layout(self):
        prompt = (EXAMPLES / "one_shot_prompt_2.txt").read_text(encoding="utf-8")
        self.assertIn("01_boiling_point/01_Ten_Steps_in_Prompts.ipynb", prompt)
        self.assertIn("01_boiling_point/examples/04_BP_One_Shot_Workflow.ipynb", prompt)
        self.assertNotIn("01_boiling_point/extras/", prompt)

    def test_bp_guide_uses_prepared_parquet_without_requiring_the_reaction_csv(self):
        notebook = read_notebook(MODULE / "01_Ten_Steps_in_Prompts.ipynb")
        code = "\n".join(source(c) for c in notebook["cells"] if c["cell_type"] == "code")
        self.assertIn("read_parquet", code)
        self.assertNotIn("read_csv", code)
        self.assertNotIn("reaction_conversion_dataset", code)
        text = "\n".join(source(c) for c in notebook["cells"])
        self.assertIn("02_BP_Vibe_Coding_Workbook.ipynb", text)
        self.assertIn("final test evaluation belongs in Step 10", text)

    def test_moved_references_resolve_shared_data_and_outputs_from_all_supported_folders(self):
        expected_output = {
            REFERENCES[0]: "bp_chemprop",
            REFERENCES[1]: "bp_molformer",
            REFERENCES[3]: "bp_one_shot_2",
        }
        for name, output in expected_output.items():
            notebook = read_notebook(EXAMPLES / name)
            first = next(c for c in notebook["cells"] if c["cell_type"] == "code")
            # Execute only path configuration, not imports, data writes, or training.
            tree = ast.parse(source(first))
            selected = []
            names = {"REPO", "MODULE", "ACQUIRED_DATA", "ACQUIRED_DATA_FILE",
                     "FALLBACK_DATA", "PACKAGED_DATA_FILE", "DATA_FILE", "OUTPUT"}
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "find_repository_root":
                    selected.append(node)
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in names for target in node.targets
                ):
                    selected.append(node)
            setup = compile(ast.Module(body=selected, type_ignores=[]), str(EXAMPLES / name), "exec")
            for directory in (ROOT, MODULE, EXAMPLES):
                with self.subTest(reference=name, cwd=directory), patch.object(Path, "cwd", return_value=directory):
                    scope = {"Path": Path}
                    exec(setup, scope)
                    self.assertEqual(scope["REPO"], ROOT)
                    self.assertEqual(scope["MODULE"], MODULE)
                    self.assertEqual(scope["OUTPUT"], MODULE / "outputs" / output)
                    self.assertTrue(scope["DATA_FILE"].is_file())
                    self.assertNotIn("examples", scope["DATA_FILE"].relative_to(MODULE).parts)

    def test_notebook_markdown_links_still_point_to_existing_local_files(self):
        for path in [*MODULE.glob("*.ipynb"), *EXAMPLES.glob("*.ipynb")]:
            for cell in read_notebook(path)["cells"]:
                if cell["cell_type"] != "markdown":
                    continue
                for target in re.findall(r"\]\(([^)]+)\)", source(cell)):
                    if target.startswith(("https://", "http://", "#", "data:", "attachment:")):
                        continue
                    with self.subTest(notebook=path.name, target=target):
                        self.assertTrue((path.parent / target.split("#", 1)[0]).is_file())


if __name__ == "__main__":
    unittest.main()
