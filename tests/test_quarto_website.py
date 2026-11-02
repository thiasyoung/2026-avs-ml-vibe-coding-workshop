"""Keep website publishing read-only, curated, and separate from the coding runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "_quarto.yml").read_text(encoding="utf-8"))
WORKFLOW_TEXT = (ROOT / ".github/workflows/publish-website.yml").read_text(encoding="utf-8")
WORKFLOW = yaml.safe_load(WORKFLOW_TEXT)
SPEC = importlib.util.spec_from_file_location("validate_reading_site", ROOT / "scripts/ci/validate_website.py")
site_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(site_check)


def navigation_links(items: list) -> list[str]:
    links = []
    for item in items:
        if isinstance(item, str):
            links.append(item)
        else:
            if "href" in item:
                links.append(item["href"])
            links.extend(navigation_links(item.get("contents", [])))
    return links


class QuartoConfigurationTests(unittest.TestCase):
    def test_homepage_heading_matches_the_workshop_website_title(self):
        title = "How to Use AI to Vibe Code and Build a Scientific Machine Learning Pipeline"
        homepage = (ROOT / "index.qmd").read_text(encoding="utf-8")
        frontmatter = yaml.safe_load(homepage.split("---", 2)[1])
        self.assertEqual(CONFIG["website"]["title"], title)
        self.assertEqual(frontmatter["title"], title)

    def test_module_navigation_separates_main_workbooks_from_optional_examples(self):
        sidebar = CONFIG["website"]["sidebar"]["contents"]
        for module in site_check.MODULES:
            with self.subTest(module=module):
                section = next(item for item in sidebar if item.get("href") == f"{module}/README.md")
                main = [item["href"] for item in section["contents"]
                        if item.get("href", "").endswith(".ipynb")]
                expected = sorted(p.relative_to(ROOT).as_posix()
                                  for p in (ROOT / module).glob("*.ipynb"))
                self.assertEqual(main, expected)
                self.assertEqual([Path(path).name[:2] for path in main], ["00", "01", "02"])
                examples = [item for item in section["contents"] if "contents" in item]
                self.assertEqual(len(examples), 1)
                self.assertEqual(examples[0]["section"], "Examples (optional)")
                self.assertEqual(examples[0]["href"], f"{module}/examples/README.md")
                notebooks = [path for path in navigation_links(examples[0]["contents"])
                             if path.endswith(".ipynb")]
                expected_examples = sorted(p.relative_to(ROOT).as_posix()
                                           for p in (ROOT / module / "examples").glob("*.ipynb"))
                self.assertEqual(sorted(notebooks), expected_examples)

    def test_homepage_local_page_links_are_in_the_reading_site(self):
        homepage = (ROOT / "index.qmd").read_text(encoding="utf-8")
        links = re.findall(r"\]\(([^)]+)\)", homepage)
        rendered = set(CONFIG["project"]["render"])
        for link in links:
            if not link.startswith(("https://", "http://", "#")):
                self.assertIn(link.split("#", 1)[0], rendered, link)

    def test_only_participant_pages_are_rendered_and_every_page_is_in_navigation(self):
        rendered = CONFIG["project"]["render"]
        expected = {p.relative_to(ROOT).as_posix() for p in site_check.page_sources()}
        self.assertEqual(set(rendered), expected)
        self.assertEqual(len(rendered), len(expected))
        self.assertEqual(set(navigation_links(CONFIG["website"]["sidebar"]["contents"])), expected)
        for path in rendered:
            self.assertTrue((ROOT / path).is_file(), path)
            self.assertFalse(site_check.FORBIDDEN_PARTS.intersection(Path(path).parts), path)
        self.assertNotIn("scratch/HTEM_Feature_Selection_Study.ipynb", rendered)

    def test_code_folds_but_outputs_and_original_notebooks_remain_available(self):
        self.assertIs(CONFIG["execute"]["enabled"], False)
        self.assertIs(CONFIG["execute"]["echo"], True)
        self.assertIs(CONFIG["execute"]["output"], True)
        self.assertIs(CONFIG["format"]["html"]["code-fold"], True)
        self.assertEqual(CONFIG["format"]["html"]["code-summary"], "Show Python code")
        self.assertIs(CONFIG["website"]["sidebar"]["search"], True)
        resources = CONFIG["project"]["resources"]
        self.assertFalse(any("*" in path or "?" in path for path in resources),
                         "Quarto resource globs can include nested archives and checkpoints")
        self.assertTrue(all(path.startswith("/") for path in resources),
                        "Anchor downloads at the project root, not a recursive basename search")
        downloads = {(ROOT / path.lstrip("/")).resolve() for path in resources if path.endswith(".ipynb")}
        self.assertEqual(downloads, set(site_check.notebook_sources()))
        prompts = {(ROOT / path.lstrip("/")).resolve() for path in resources if path.endswith(".txt")}
        self.assertEqual(prompts, set(site_check.prompt_sources()))
        self.assertFalse(any("/extras/" in path for path in resources))
        self.assertEqual(site_check.SITE_PATH, "/" + CONFIG["website"]["site-url"].split(".io/", 1)[1])

    def test_pages_build_cannot_execute_notebooks_or_publish_from_a_pr(self):
        build = WORKFLOW["jobs"]["build"]
        deploy = WORKFLOW["jobs"]["deploy"]
        self.assertEqual(WORKFLOW["permissions"], {"contents": "read"})
        self.assertNotIn("permissions", build)
        self.assertEqual(deploy["permissions"], {"pages": "write", "id-token": "write"})
        self.assertEqual(deploy["needs"], "build")
        self.assertEqual(deploy["environment"]["name"], "github-pages")
        self.assertIn("github.ref == 'refs/heads/main'", deploy["if"])
        self.assertIn("github.event_name != 'pull_request'", deploy["if"])
        self.assertIn("quarto render --no-execute", WORKFLOW_TEXT)
        self.assertIn("python scripts/ci/validate_website.py", WORKFLOW_TEXT)
        self.assertIn("git diff --exit-code -- '*.ipynb'", WORKFLOW_TEXT)
        self.assertNotIn("secrets.", WORKFLOW_TEXT)
        self.assertNotIn("uv sync", WORKFLOW_TEXT)
        self.assertNotIn("openai/", WORKFLOW_TEXT)
        self.assertFalse(any(s.get("with", {}).get("persist-credentials", False) for s in build["steps"]))
        quarto = next(s for s in build["steps"] if s.get("uses", "").startswith("quarto-dev/"))
        self.assertEqual(quarto["with"]["version"], "1.9.38")

    def test_generated_site_and_cache_are_ignored(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/_site/", ignore)
        self.assertIn("/.quarto/", ignore)
        self.assertEqual(CONFIG["project"]["output-dir"], "_site")

    def test_notebook_maintenance_never_rewrites_or_validates_generated_copies(self):
        for filename, setting in (("scripts/ci/validate_notebooks.py", "EXCLUDED_PARTS"),
                                  ("scripts/prepare_notebook_presentation.py", "EXCLUDED")):
            with self.subTest(script=filename):
                spec = importlib.util.spec_from_file_location("notebook_maintenance", ROOT / filename)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertTrue({"_site", ".quarto"}.issubset(getattr(module, setting)))


class WebsiteValidatorTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path]:
        sources, site = root / "source", root / "site"
        sources.mkdir()
        site.mkdir()
        notebook = sources / "00_Demo.ipynb"
        notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["raise RuntimeError('never run')"],
            "outputs": [{"output_type": "display_data", "data": {"image/png": "saved plot"}}]}]}), encoding="utf-8")
        html = ('<details class="code-fold"><summary>Show Python code</summary><pre>code</pre></details>'
                '<div class="cell-output-display"><img src="saved.png"></div>'
                '<a href="00_Demo.ipynb" download>Download notebook</a>')
        for source in site_check.page_sources(sources):
            output = site / source.relative_to(sources).with_suffix(".html")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(html if source.suffix == ".ipynb" else '<a href="https://example.com">External</a>',
                              encoding="utf-8")
        (site / notebook.name).write_bytes(notebook.read_bytes())
        for source in site_check.prompt_sources(sources):
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("Example prompt; never executed by the website.", encoding="utf-8")
            download = site / source.relative_to(sources)
            download.parent.mkdir(parents=True, exist_ok=True)
            download.write_bytes(source.read_bytes())
        (site / "saved.png").write_bytes(b"saved figure")
        (site / "search.json").write_text('[{"title":"Demo"}]', encoding="utf-8")
        return sources, site

    def test_saved_outputs_and_byte_identical_download_validate_without_running_code(self):
        with tempfile.TemporaryDirectory() as directory:
            sources, site = self.fixture(Path(directory))
            self.assertEqual(site_check.validate_site(site, sources), [])

    def test_missing_page_image_download_or_search_is_reported(self):
        for name in ("index.html", "saved.png", "00_Demo.ipynb", "search.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                sources, site = self.fixture(Path(directory))
                (site / name).unlink()
                self.assertTrue(site_check.validate_site(site, sources))

    def test_missing_or_changed_prompt_download_is_reported(self):
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                sources, site = self.fixture(Path(directory))
                source = site_check.prompt_sources(sources)[0]
                download = site / source.relative_to(sources)
                if changed:
                    download.write_text("Outdated prompt", encoding="utf-8")
                else:
                    download.unlink()
                errors = site_check.validate_site(site, sources)
                self.assertTrue(any("prompt download" in e.lower() for e in errors), errors)

    def test_expanded_or_missing_code_and_missing_outputs_are_reported(self):
        for text in ('<details class="code-fold" open></details>', '<p>Code and outputs missing</p>'):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                sources, site = self.fixture(Path(directory))
                (site / "00_Demo.html").write_text(text, encoding="utf-8")
                errors = site_check.validate_site(site, sources)
                self.assertTrue(any("code" in e.lower() for e in errors))
                self.assertTrue(any("outputs" in e.lower() for e in errors))

    def test_changed_download_and_non_presentation_artifacts_block_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            sources, site = self.fixture(Path(directory))
            (site / "00_Demo.ipynb").write_text("changed", encoding="utf-8")
            (site / "scratch").mkdir()
            (site / ".ipynb_checkpoints").mkdir()
            (site / "weights.pt").write_bytes(b"not a presentation resource")
            (site / "unlisted.ipynb").write_text("{}", encoding="utf-8")
            errors = site_check.validate_site(site, sources)
            self.assertTrue(any("differs" in e for e in errors))
            self.assertTrue(any("scratch" in e for e in errors))
            self.assertTrue(any(".ipynb_checkpoints" in e for e in errors))
            self.assertTrue(any("weights.pt" in e for e in errors))
            self.assertTrue(any("Unlisted notebook" in e for e in errors))

    def test_project_subpath_and_nested_links_resolve_within_the_site(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            page = root / "module/demo.html"
            self.assertEqual(site_check.local_target(page, "../index.html#intro", root), root / "index.html")
            self.assertEqual(site_check.local_target(page, site_check.SITE_PATH + "index.html", root), root / "index.html")
            self.assertIsNone(site_check.local_target(page, "https://example.com/image.png", root))
            self.assertIsNone(site_check.local_target(page, "#dynamic-menu", root))
            with self.assertRaises(ValueError):
                site_check.local_target(page, "../../outside.html", root)


if __name__ == "__main__":
    unittest.main()
