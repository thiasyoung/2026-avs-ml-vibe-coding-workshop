"""Check a rendered reading site without importing or executing workshop code.

Uses only Python's standard library so Pages CI needs no ML environment.
"""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
MODULES = ("01_boiling_point", "02_ir_functional_groups", "03_htem_conductivity")
SITE_PATH = "/2026-avs-ml-vibe-coding-workshop/"
PROMPT_FILES = ("P0_zero_shot_prompt.txt", "P1_one_shot_prompt.txt", "P2_one_shot_prompt.txt", "P3_one_shot_prompt.txt")
FORBIDDEN_PARTS = {".git", ".github", ".venv", ".ipynb_checkpoints", "__pycache__", "scratch", "archive", "outputs"}
FORBIDDEN_SUFFIXES = {".parquet", ".pt", ".pth", ".ckpt", ".env"}


def notebook_sources(root: Path = ROOT) -> list[Path]:
    paths = list(root.glob("00_*.ipynb"))
    for module in MODULES:
        paths.extend((root / module).glob("*.ipynb"))
        paths.extend((root / module / "examples").rglob("*.ipynb"))
    return sorted(path for path in paths
                  if not FORBIDDEN_PARTS.intersection(path.relative_to(root).parts))


def prompt_sources(root: Path = ROOT) -> list[Path]:
    return [root / "01_boiling_point/examples/prompt_iteration/prompts" / name for name in PROMPT_FILES]


def page_sources(root: Path = ROOT) -> list[Path]:
    return [root / "index.qmd", root / "SETUP_GUIDE.md", *notebook_sources(root),
            *(root / module / "README.md" for module in MODULES),
            root / "01_boiling_point/examples/README.md",
            root / "01_boiling_point/examples/prompt_iteration/README.md",
            root / "02_ir_functional_groups/examples/README.md",
            root / "03_htem_conductivity/examples/README.md",
            root / "03_htem_conductivity/TECHNICAL_NOTES.md"]


class Page(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.references: list[str] = []
        self.folds: list[dict] = []
        self.output_count = 0
        self.image_count = 0
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        for name in ("href", "src"):
            if attributes.get(name):
                self.references.append(attributes[name])
        classes = (attributes.get("class") or "").split()
        if tag == "details" and "code-fold" in classes:
            self.folds.append(attributes)
        if any(c.startswith("cell-output") for c in classes):
            self.output_count += 1
        if tag == "img":
            self.image_count += 1


def local_target(page: Path, reference: str, site: Path) -> Path | None:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None  # External resources and same-page/dynamic fragments are not fetched.
    path = unquote(parsed.path)
    if path.startswith(SITE_PATH):
        path = path[len(SITE_PATH):]
        target = site / path
    elif path.startswith("/"):
        target = site / path.lstrip("/")
    else:
        target = page.parent / path
    target = target.resolve()
    if not target.is_relative_to(site.resolve()):
        raise ValueError(f"Link escapes the published site: {reference}")
    if target.is_dir():
        target /= "index.html"
    return target


def validate_site(site: Path, root: Path = ROOT) -> list[str]:
    site = site.resolve()
    errors: list[str] = []
    if not (site / "index.html").is_file():
        return [f"No website at {site}; run quarto render --no-execute first."]

    for source in page_sources(root):
        relative = source.relative_to(root)
        html = site / relative.with_suffix(".html")
        if not html.is_file():
            errors.append(f"Missing rendered page: {relative.with_suffix('.html')}")
            continue
        if source.suffix != ".ipynb":
            continue
        download = site / relative
        if not download.is_file():
            errors.append(f"Missing notebook download: {relative}")
        elif hashlib.sha256(download.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
            errors.append(f"Notebook download differs from the original: {relative}")
        notebook = json.loads(source.read_text(encoding="utf-8"))
        page = Page(html.read_text(encoding="utf-8"))
        code = [c for c in notebook["cells"] if c["cell_type"] == "code" and "".join(c["source"]).strip()]
        if code and not page.folds:
            errors.append(f"No expandable code found: {relative}")
        if any("open" in fold for fold in page.folds):
            errors.append(f"Code is not folded by default: {relative}")
        outputs = [o for c in notebook["cells"] for o in c.get("outputs", [])]
        if outputs and not page.output_count:
            errors.append(f"Saved outputs are absent: {relative}")
        if any("image/png" in o.get("data", {}) for o in outputs) and not page.image_count:
            errors.append(f"Saved plots are absent: {relative}")

    for source in prompt_sources(root):
        relative = source.relative_to(root)
        download = site / relative
        if not source.is_file():
            errors.append(f"Missing prompt source: {relative}")
        elif not download.is_file():
            errors.append(f"Missing prompt download: {relative}")
        elif download.read_bytes() != source.read_bytes():
            errors.append(f"Prompt download differs from the original: {relative}")

    for html in site.rglob("*.html"):
        for reference in Page(html.read_text(encoding="utf-8")).references:
            try:
                target = local_target(html, reference, site)
            except ValueError as error:
                errors.append(f"{html.relative_to(site)}: {error}")
                continue
            if target is not None and not target.is_file():
                errors.append(f"Broken local link in {html.relative_to(site)}: {reference}")

    for path in site.rglob("*"):
        relative = path.relative_to(site)
        if path.is_symlink():
            errors.append(f"Symlink in Pages artifact: {relative}")
        if FORBIDDEN_PARTS.intersection(relative.parts) or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"Non-presentation artifact copied into website: {relative}")
        if path.is_file() and path.suffix == ".ipynb" and root / relative not in notebook_sources(root):
            errors.append(f"Unlisted notebook copied into website: {relative}")

    search = site / "search.json"
    if not search.is_file() or not json.loads(search.read_text(encoding="utf-8")):
        errors.append("Missing or empty site search index.")
    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    errors = validate_site(args.site)
    if errors:
        print("Website validation failed:", *(f"- {error}" for error in errors), sep="\n")
        return 1
    print(f"Validated {len(page_sources())} pages, {len(notebook_sources())} unchanged notebook downloads, "
          f"and {len(prompt_sources())} unchanged prompt downloads; "
          "local links, folded code, saved outputs, and search passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
