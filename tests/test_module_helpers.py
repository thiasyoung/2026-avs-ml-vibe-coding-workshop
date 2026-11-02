"""Library relocation: imports, resources, provenance and saved-result compatibility."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
IR = ROOT / '02_ir_functional_groups'
HTEM = ROOT / '03_htem_conductivity'
sys.path.insert(0, str(HTEM / 'lib'))
sys.path.insert(0, str(IR / 'lib'))
import htem_chemistry as chemistry
import htem_features as features
import htem_paths as paths
import htem_source_dos as dos


class ModuleHelperLayoutTests(unittest.TestCase):
    def test_module_roots_have_no_helper_scripts(self):
        for module in (IR, HTEM):
            self.assertEqual(list(module.glob('*.py')), [])
        for name in ('ir_carpentry.py', 'ir_example_support.py'):
            self.assertTrue((IR / 'lib' / name).is_file())
        for name in ('run_examples.py', 'run_carpentry_comparison.py',
                     'verify_carpentry_comparison.py', 'strategies.py'):
            self.assertTrue((IR / 'examples' / name).is_file())

    def test_imports_in_fresh_process_from_supported_directories(self):
        for module, imports in (
            (IR, 'import ir_carpentry, ir_example_support, run_examples; '
                 'assert ir_example_support.module_directory() == module'),
            (HTEM, 'import htem_workshop, htem_experiments, htem_download, '
                   'htem_graph, htem_features, htem_chemistry, htem_source_dos, htem_feature_study; '
                   'assert htem_feature_study.MODULE == module'),
        ):
            for directory in (ROOT, module, module / 'examples'):
                with self.subTest(module=module.name, cwd=directory.name):
                    code = ('from pathlib import Path; import sys; '
                            f'module = Path({str(module)!r}); '
                            'sys.path[:0] = [str(module / "lib"), str(module / "examples")]; '
                            + imports)
                    result = subprocess.run([sys.executable, '-W', 'ignore', '-c', code],
                                            cwd=directory, text=True, capture_output=True,
                                            encoding='utf-8', errors='replace', timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_resource_paths_still_point_to_module_data(self):
        self.assertEqual(paths.MODULE_DIR, HTEM)
        self.assertEqual(features.DATA, HTEM / 'data')
        for reference in (chemistry.REFERENCE, dos.REFERENCE, features.SPECIFICATION, features.TABLE):
            self.assertEqual(reference.parent, HTEM / 'data')
            self.assertTrue(reference.is_file())

    def test_exact_relocation_aliases_reconstruct_the_original_source(self):
        # Prove these aliases preserve the complete old source, not just its
        # numerical functions or an unchecked version label.
        replacements = {
            'htem_chemistry.py': [
                ('from htem_paths import MODULE_DIR, source_sha256\n\n', ''),
                ('MODULE_DIR / "data/htem_chemistry_reference.json"',
                 'Path(__file__).with_name("data") / "htem_chemistry_reference.json"'),
                ('source_sha256(__file__)', 'hashlib.sha256(Path(__file__).read_bytes()).hexdigest()'),
                ('source_sha256(dos.__file__)', 'hashlib.sha256(Path(dos.__file__).read_bytes()).hexdigest()'),
            ],
            'htem_source_dos.py': [
                ('from htem_paths import MODULE_DIR\n\n', ''),
                ('MODULE_DIR / "data/htem_source_dos_reference.json"',
                 'Path(__file__).with_name("data") / "htem_source_dos_reference.json"'),
                ('module = MODULE_DIR', 'module = Path(__file__).resolve().parent'),
            ],
            'htem_features.py': [
                ('from htem_paths import MODULE_DIR, source_sha256\n\n', ''),
                ('MODULE_DIR / "data"', 'Path(__file__).with_name("data")'),
                ('source_sha256(__file__)', 'digest(__file__)'),
            ],
        }
        for name, changes in replacements.items():
            with self.subTest(helper=name):
                path = HTEM / 'lib' / name
                original = path.read_text(encoding='utf-8')
                for current, previous in changes:
                    self.assertIn(current, original)
                    original = original.replace(current, previous)
                self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), paths.source_sha256(path))

    def test_changed_helper_source_is_not_accepted_as_a_relocation(self):
        source = (HTEM / 'lib/htem_features.py').read_bytes()
        expected = paths.source_sha256(HTEM / 'lib/htem_features.py')
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / 'helper.py'
            temporary.write_bytes(source.replace(b'\n', b'\r\n'))
            self.assertEqual(paths.source_sha256(temporary), expected)
            temporary.write_bytes(source + b'\n# arbitrary subsequent edit\n')
            self.assertNotEqual(paths.source_sha256(temporary), expected)

    def test_distributed_preparation_still_has_matching_source_fingerprints(self):
        manifest = json.loads((HTEM / 'data/frozen_features/manifest.json').read_text())
        for key, value in chemistry.reference_fingerprint().items():
            self.assertEqual(manifest['chemistry'][key], value, key)
        self.assertIn('03_htem_conductivity/lib/*.py text eol=lf',
                      (ROOT / '.gitattributes').read_text())


if __name__ == '__main__':
    unittest.main()
