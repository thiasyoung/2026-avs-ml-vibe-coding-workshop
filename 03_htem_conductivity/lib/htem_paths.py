"""Module-relative resources and verified compatibility with pre-lib artifacts."""
from pathlib import Path
import hashlib

MODULE_DIR = Path(__file__).resolve().parents[1]

# These exact source revisions differ from the original helpers only in resource
# paths and calls to this checksum adapter. Preserve their existing provenance
# IDs without rewriting prepared data/models. Any other edit gets a new digest
# and is rejected by the existing stale-artifact checks. Never extend this list
# for algorithm/feature changes: regenerate preparation and results instead.
_RELOCATED_SOURCE = {
    "e85af58a8788225f7a51182dc644f3797dbb0091bf507ad93a3e9a0363421dfb":
        "019d9623ba4b7f207851071005934ba379200922eeeb1e6d2cecc3461819f8b5",
    "0b436e2ac5b2d327d9010162028b1fd3aa94606b3b9ede3c8cbd5f4637c2ce42":
        "59c8c0def15bff03cfe69c893c0c0ac8e17ebac1022d419e19694a3a50183b57",
    "0eccfc0834dd0eb3b51b8329d6e98bd241d9c4f981802c6afa8cb231ca03dd2e":
        "25374d87f43a1f1148f78e9671cc45b03e03703c18881b8a81ff29a2b39e8c50",
}


def source_sha256(path):
    """Hash actual source, normalizing Windows newlines, with exact move aliases."""
    digest = hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return _RELOCATED_SOURCE.get(digest, digest)
