"""Presentation changes must never change scientific code or populated outputs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import nbformat
from bs4 import BeautifulSoup
from nbconvert.filters.markdown import markdown2html

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_notebook_presentation as presentation


class NotebookPresentationTests(unittest.TestCase):
    def test_every_code_cell_has_fresh_plain_summary_and_visible_outputs(self):
        code_count = 0
        for path in presentation.notebook_paths():
            with self.subTest(notebook=path.relative_to(ROOT)):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                # The generic course template predates nbformat cell IDs; validate
                # without inventing IDs or changing its existing metadata.
                nbformat.validate(nbformat.from_dict(notebook))
                ids = [c["id"] for c in notebook["cells"] if "id" in c]
                self.assertEqual(len(ids), len(set(ids)))
                for i, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    code_count += 1
                    self.assertGreater(i, 0)
                    notes = notebook["cells"][i - 1]
                    self.assertEqual(notes["cell_type"], "markdown")
                    metadata = notes["metadata"][presentation.MARKER]
                    self.assertEqual(metadata["kind"], "notes")
                    self.assertEqual(metadata["code_cell"], cell["id"])
                    self.assertEqual(metadata["source_sha256"], hashlib.sha256(
                        presentation.source_text(cell).encode("utf-8")).hexdigest())
                    rendered = BeautifulSoup(markdown2html(presentation.source_text(notes)), "html.parser")
                    for anchor in rendered.select("a.anchor-link"):
                        anchor.decompose()  # nbconvert adds a permalink glyph to headings.
                    headings = [h.get_text() for h in rendered.select("h3")]
                    self.assertEqual(len(headings), 1)
                    self.assertTrue(headings[0].strip())
                    self.assertNotEqual(headings[0].lower(), "code summary")
                    paragraphs = rendered.select("p")
                    self.assertEqual(len(paragraphs), 1)
                    self.assertTrue(paragraphs[0].get_text().startswith("Code Summary: "))
                    self.assertEqual(paragraphs[0].get_text().count("Code Summary: "), 1)
                    self.assertLessEqual(len(paragraphs[0].get_text().split()), 60)
                    self.assertFalse(rendered.select("ul, ol, pre, code, strong, em"))
                    self.assertNotIn("what to notice", presentation.source_text(notes).lower())
                    self.assertTrue(cell["metadata"]["jupyter"]["source_hidden"])
                    self.assertFalse(cell["metadata"]["jupyter"]["outputs_hidden"])
                    self.assertFalse(cell["metadata"]["collapsed"])
        self.assertGreater(code_count, 100)

    def test_catalog_matches_code_notebooks_and_transform_is_idempotent(self):
        paths_with_code = set()
        for path in presentation.notebook_paths():
            relative = path.relative_to(ROOT).as_posix()
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if any(c["cell_type"] == "code" for c in notebook["cells"]):
                paths_with_code.add(relative)
                guides = [c for c in notebook["cells"] if c.get("metadata", {}).get(
                    presentation.MARKER, {}).get("kind") == "guide"]
                self.assertEqual(len(guides), 1, relative)
            self.assertEqual(presentation.prepare_notebook(notebook, relative), notebook, relative)
        self.assertEqual(set(presentation.NOTES), paths_with_code)
        self.assertEqual(set(presentation.HEADINGS), paths_with_code)

    def test_transform_preserves_original_code_outputs_prose_and_metadata(self):
        fixture = {
            "cells": [
                {"cell_type": "markdown", "id": "intro", "metadata": {},
                 "source": ["# Demo\n", "\n", "![source](source.png)\n"],
                 "attachments": {"example": {"image/png": "opaque attachment"}}},
                {"cell_type": "code", "id": "example", "execution_count": 7,
                 "source": ["print('saved result')\n"],
                 "metadata": {"execution": {"timestamp": "keep"}, "custom": 123,
                              "jupyter": {"custom": "keep", "outputs_hidden": True}, "collapsed": True},
                 "outputs": [{"output_type": "stream", "name": "stdout", "text": ["saved result\n"]}]},
            ], "metadata": {"custom": "keep"}, "nbformat": 4, "nbformat_minor": 5,
        }
        before = deepcopy(fixture)
        with patch.dict(presentation.NOTES, {"fixture.ipynb": ["Prints the saved result."]}), \
                patch.dict(presentation.HEADINGS, {"fixture.ipynb": ["Display the saved result"]}):
            result = presentation.prepare_notebook(fixture, "fixture.ipynb")
            self.assertEqual(presentation.prepare_notebook(result, "fixture.ipynb"), result)
        self.assertEqual(fixture, before, "Transform must not mutate its input")
        original_cells = [c for c in result["cells"] if presentation.MARKER not in c.get("metadata", {})]
        expected = deepcopy(before)
        expected["cells"][1]["metadata"]["jupyter"].update(source_hidden=True, outputs_hidden=False)
        expected["cells"][1]["metadata"]["collapsed"] = False
        self.assertEqual(original_cells, expected["cells"])
        self.assertEqual(result["metadata"], before["metadata"])

    def test_notes_mismatch_fails_instead_of_assigning_wrong_explanations(self):
        fixture = {"cells": [{"cell_type": "code", "source": ["x = 1"], "metadata": {}}]}
        with self.assertRaisesRegex(ValueError, "review teaching notes"):
            presentation.prepare_notebook(fixture, "unreviewed.ipynb")

    def test_headings_mismatch_fails_instead_of_assigning_wrong_titles(self):
        fixture = {"cells": [{"cell_type": "code", "source": ["x = 1"], "metadata": {}}]}
        for headings in (None, []):
            with self.subTest(headings=headings), \
                    patch.dict(presentation.NOTES, {"fixture.ipynb": ["Sets an example value."]}), \
                    patch.dict(presentation.HEADINGS, {"fixture.ipynb": headings}):
                with self.assertRaisesRegex(ValueError, "review descriptive headings"):
                    presentation.prepare_notebook(fixture, "fixture.ipynb")

    def test_code_first_notebook_has_only_one_reading_guide(self):
        fixture = {"cells": [
            {"cell_type": "code", "source": ["x = 1"], "metadata": {}, "id": "a"},
            {"cell_type": "code", "source": [], "metadata": {}, "id": "b"},
        ]}
        with patch.dict(presentation.NOTES, {"fixture.ipynb": ["Sets an example value.", "This practice cell is blank."]}), \
                patch.dict(presentation.HEADINGS, {"fixture.ipynb": ["Set an example value", "Practice: add a calculation"]}):
            result = presentation.prepare_notebook(fixture, "fixture.ipynb")
        self.assertEqual(sum(c.get("metadata", {}).get(presentation.MARKER, {}).get("kind") == "guide"
                             for c in result["cells"]), 1)

    def test_markdown_only_notebook_is_unchanged(self):
        fixture = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Roadmap"]}]}
        self.assertEqual(presentation.prepare_notebook(fixture, "roadmap.ipynb"), fixture)

    def test_legacy_notes_are_replaced_without_duplication(self):
        for legacy in ("**What to notice**\n\n- Old presenter instruction.\n",
                       "### Code Summary\n\nSets an example value.\n"):
            fixture = {"cells": [
                {"cell_type": "markdown", "metadata": {presentation.MARKER: {"kind": "notes"}},
                 "source": [legacy]},
                {"cell_type": "code", "source": ["x = 1"], "metadata": {}, "id": "a"},
            ]}
            with self.subTest(legacy=legacy), \
                    patch.dict(presentation.NOTES, {"fixture.ipynb": ["Sets an example value."]}), \
                    patch.dict(presentation.HEADINGS, {"fixture.ipynb": ["Set an example value"]}):
                result = presentation.prepare_notebook(fixture, "fixture.ipynb")
                self.assertEqual(presentation.prepare_notebook(result, "fixture.ipynb"), result)
            summaries = [c for c in result["cells"] if c.get("metadata", {}).get(
                presentation.MARKER, {}).get("kind") == "notes"]
            self.assertEqual(len(summaries), 1)
            self.assertEqual(presentation.source_text(summaries[0]),
                             "### Set an example value\n\nCode Summary: Sets an example value.\n")

    def test_participant_docs_describe_code_summaries(self):
        for relative in ("README.md", "docs/NOTEBOOK_PRESENTATION.md"):
            with self.subTest(document=relative):
                text = (ROOT / relative).read_text(encoding="utf-8")
                self.assertIn("Code Summary:", text)
                self.assertNotIn("what to notice", text.lower())
                self.assertNotRegex(text, r"(?im)^\s*#{1,6}\s+Code Summary\s*$")

    def test_blank_practice_cells_are_not_described_as_completed_work(self):
        for path in presentation.notebook_paths():
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code" and not presentation.source_text(cell).strip():
                    with self.subTest(notebook=path.name, cell=index):
                        summary = presentation.source_text(notebook["cells"][index - 1]).lower()
                        self.assertIn("blank", summary)
                        self.assertIn("practice", summary)


if __name__ == "__main__":
    unittest.main()
