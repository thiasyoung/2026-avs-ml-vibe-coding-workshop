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
REFERENCE_MODELS = EXAMPLES / "reference_models"
PROMPT_ITERATION = EXAMPLES / "prompt_iteration"
REFERENCES = (
    "01_BP_Chemprop_GNN.ipynb",
    "02_BP_Foundation_Model.ipynb",
)
PROMPT_LEVELS = ("P0", "P1", "P2", "P3")


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
        self.assertEqual(sorted(p.name for p in EXAMPLES.glob("*.ipynb")), [])
        self.assertEqual(sorted(p.name for p in REFERENCE_MODELS.glob("*.ipynb")), list(REFERENCES))
        self.assertTrue((EXAMPLES / "README.md").is_file())
        self.assertTrue((PROMPT_ITERATION / "README.md").is_file())
        self.assertTrue((PROMPT_ITERATION / "Compare_Prompt_Strategies.ipynb").is_file())
        self.assertFalse((MODULE / "extras").exists())
        # Retired in favor of the prompt-iteration set; they must not reappear unnoticed.
        for retired in ("03_BP_One_Shot_Vibe_Coding.ipynb", "04_BP_One_Shot_Workflow.ipynb",
                        "one_shot_prompt_1.txt", "one_shot_prompt_2.txt"):
            self.assertFalse((EXAMPLES / retired).exists(), retired)
        readme = (PROMPT_ITERATION / "README.md").read_text(encoding="utf-8")
        prompts = sorted((PROMPT_ITERATION / "prompts").glob("*.txt"))
        notebooks = sorted((PROMPT_ITERATION / "notebooks").glob("*.ipynb"))
        self.assertEqual(sorted(p.name.split("_one_shot")[0].split("_zero")[0] for p in prompts), sorted(PROMPT_LEVELS))
        self.assertEqual(len(notebooks), len(PROMPT_LEVELS))
        for path in (*prompts, *notebooks):
            with self.subTest(path=path.name):
                self.assertIn(f"]({path.relative_to(PROMPT_ITERATION).as_posix()})", readme)
        # Generated run artifacts are local-only (git-ignored outputs/), never stored beside the notebooks.
        self.assertEqual([p for p in EXAMPLES.rglob("*") if p.suffix in {".joblib", ".parquet", ".pyc"}], [])

    def test_workbook_contains_one_copyable_prompt_instead_of_prescribed_exercises(self):
        notebook = read_notebook(MODULE / "02_BP_Vibe_Coding_Workbook.ipynb")
        populated = [c for c in notebook["cells"] if source(c).strip()]
        # One prompt cell, then a short strategy-toolbox table.
        self.assertEqual(len(populated), 2)  # Empty Markdown cells are harmless editor state.
        self.assertEqual({c["cell_type"] for c in populated}, {"markdown"})
        text = source(populated[0])
        ladder = source(populated[1])
        self.assertNotIn("```", ladder)
        for lever in ("Hypothesis first", "Plan before code", "Self-consistency", "Critique and revise"):
            self.assertIn(f"| {lever} |", ladder)
        self.assertIn("Working approach:", text)
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

    def test_prompts_point_to_the_template_and_valid_module_paths(self):
        for path in sorted((PROMPT_ITERATION / "prompts").glob("*.txt")):
            with self.subTest(prompt=path.name):
                prompt = path.read_text(encoding="utf-8")
                self.assertNotIn("01_boiling_point/extras/", prompt)
                if not path.name.startswith("P0"):
                    self.assertIn("01_boiling_point/01_Ten_Steps_in_Prompts.ipynb", prompt)

    def test_comparison_notebook_is_one_prompt_and_runs_no_code(self):
        notebook = read_notebook(PROMPT_ITERATION / "Compare_Prompt_Strategies.ipynb")
        self.assertFalse([c for c in notebook["cells"] if c["cell_type"] == "code"])
        text = "\n".join(source(c) for c in notebook["cells"])
        self.assertEqual(len(re.findall(r"```text\n(.*?)\n```", text, flags=re.DOTALL)), 1)
        for level in ("P0", "P1", "P2", "P3"):
            self.assertIn(level, text)

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
        }
        for name, output in expected_output.items():
            notebook = read_notebook(REFERENCE_MODELS / name)
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
            setup = compile(ast.Module(body=selected, type_ignores=[]), str(REFERENCE_MODELS / name), "exec")
            for directory in (ROOT, MODULE, EXAMPLES, REFERENCE_MODELS):
                with self.subTest(reference=name, cwd=directory), patch.object(Path, "cwd", return_value=directory):
                    scope = {"Path": Path}
                    exec(setup, scope)
                    self.assertEqual(scope["REPO"], ROOT)
                    self.assertEqual(scope["MODULE"], MODULE)
                    self.assertEqual(scope["OUTPUT"], MODULE / "outputs" / output)
                    self.assertTrue(scope["DATA_FILE"].is_file())
                    self.assertNotIn("examples", scope["DATA_FILE"].relative_to(MODULE).parts)

    def test_notebook_markdown_links_still_point_to_existing_local_files(self):
        for path in [*MODULE.glob("*.ipynb"), *EXAMPLES.rglob("*.ipynb")]:
            for cell in read_notebook(path)["cells"]:
                if cell["cell_type"] != "markdown":
                    continue
                # Ignore inline code such as SMILES like `[N+](=O)[O-]`, which resembles a link.
                for target in re.findall(r"\]\(([^)]+)\)", re.sub(r"`[^`]*`", "", source(cell))):
                    if target.startswith(("https://", "http://", "#", "data:", "attachment:")):
                        continue
                    with self.subTest(notebook=path.name, target=target):
                        self.assertTrue((path.parent / target.split("#", 1)[0]).is_file())


if __name__ == "__main__":
    unittest.main()
