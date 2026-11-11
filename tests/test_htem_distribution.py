"""Frozen HTEM assets must retain their pinned bytes through Git distribution."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "03_htem_conductivity"
sys.path.insert(0, str(MODULE / "lib"))
import htem_features as features


@unittest.skipUnless(shutil.which("git"), "Git is required to test distribution filters")
class GitDistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "03_htem_conductivity/data"], cwd=ROOT
        ).decode().split("\0")
        cls.assets = [name for name in names if Path(name).suffix in {".json", ".csv"}]

    def test_frozen_text_assets_disable_line_ending_conversion(self):
        self.assertTrue(self.assets)
        for name in self.assets:
            with self.subTest(asset=name):
                attribute = subprocess.check_output(
                    ["git", "check-attr", "text", "--", name], cwd=ROOT, text=True
                ).strip()
                self.assertTrue(attribute.endswith(": text: unset"), attribute)

    def test_git_stores_exact_asset_bytes_with_windows_or_unix_settings(self):
        # hash-object without -w performs Git's actual clean conversion but
        # changes neither the repository index nor its stored objects.
        for name in self.assets:
            raw = subprocess.check_output(
                ["git", "hash-object", "--no-filters", name], cwd=ROOT
            ).strip()
            for autocrlf in ("true", "false", "input"):
                with self.subTest(asset=name, autocrlf=autocrlf):
                    filtered = subprocess.check_output(
                        ["git", "-c", f"core.autocrlf={autocrlf}", "hash-object",
                         f"--path={name}", name], cwd=ROOT
                    ).strip()
                    self.assertEqual(filtered, raw)


class FrozenAssetIntegrityTests(unittest.TestCase):
    def test_distributed_source_table_matches_its_existing_checksum(self):
        expected = json.loads(features.SPECIFICATION.read_text(encoding="utf-8"))[
            "source_feature_table_sha256"]
        self.assertEqual(hashlib.sha256(features.TABLE.read_bytes()).hexdigest(), expected)

    def test_changing_a_source_value_still_fails_the_integrity_check(self):
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "changed_table.json"
            data = json.loads(features.TABLE.read_text(encoding="utf-8"))
            data["compositions"] = {}
            table.write_text(json.dumps(data), encoding="utf-8")
            features.source_table.cache_clear()
            try:
                with patch.object(features, "TABLE", table):
                    with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                        features.source_table()
            finally:
                features.source_table.cache_clear()


if __name__ == "__main__":
    unittest.main()
