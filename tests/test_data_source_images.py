"""Source illustrations render locally without running acquisition code."""
import json
from pathlib import Path
import re
import unittest

import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from nbconvert.filters.markdown import markdown2html

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "01_boiling_point/00_BP_Get_Data_NISTChemPy_v2.ipynb": "webbook.nist.gov",
    "02_ir_functional_groups/00_IR_Get_Data_NISTChemPy_v2.ipynb": "webbook.nist.gov",
    "03_htem_conductivity/00_HTEM_Get_Data_NREL.ipynb": "htem.nlr.gov",
}


class DataSourceImageTests(unittest.TestCase):
    def test_images_and_captions_are_at_the_top(self):
        for relative, domain in SOURCES.items():
            with self.subTest(notebook=relative):
                notebook = json.loads((ROOT / relative).read_text(encoding="utf-8"))
                cell = notebook["cells"][0]
                self.assertEqual(cell["cell_type"], "markdown")
                text = "".join(cell["source"])
                self.assertLess(text.index("!["), text.index("Figure 1."))
                self.assertIn(domain, text)
                self.assertIn("credit:", text.lower())
                self.assertRegex(text, r"^# .+\n\n!\[")

    def test_local_images_render_from_each_notebook_folder(self):
        for relative in SOURCES:
            with self.subTest(notebook=relative):
                notebook_path = ROOT / relative
                notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
                rendered = BeautifulSoup(markdown2html("".join(notebook["cells"][0]["source"])), "html.parser")
                image = rendered.find("img")
                self.assertIsNotNone(image)
                self.assertTrue(image.get("alt"))
                self.assertFalse(re.match(r"https?://", image["src"]))
                path = (notebook_path.parent / image["src"]).resolve()
                self.assertTrue(path.is_relative_to(ROOT / "docs/data_sources"))
                self.assertTrue(path.is_file())
                self.assertLess(path.stat().st_size, 1_000_000)

    def test_images_are_valid_and_not_blank_captures(self):
        for path in (ROOT / "docs/data_sources").glob("*.png"):
            with self.subTest(image=path.name), Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreaterEqual(image.width, 1000)
                self.assertGreaterEqual(image.height, 500)
                thumbnail = np.asarray(image.convert("RGB").resize((140, 90)))
                self.assertGreater(float(thumbnail.std()), 15.)
                self.assertGreater(len(np.unique(thumbnail.reshape(-1, 3), axis=0)), 100)


if __name__ == "__main__":
    unittest.main()
