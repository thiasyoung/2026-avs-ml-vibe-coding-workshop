"""Prompt-only IR workflow, moved references, and the real trainer input contract."""

import ast
from contextlib import chdir, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / '02_ir_functional_groups'
EXAMPLES = MODULE / 'examples'
sys.path.insert(0, str(MODULE / 'lib'))
sys.path.insert(0, str(EXAMPLES))
import ir_carpentry as carpentry
from ir_example_support import module_directory, validate_tables
from run_examples import select_dataset


def notebook(path):
    return json.loads(path.read_text(encoding='utf-8'))


def source(cell):
    value = cell['source']
    return ''.join(value) if isinstance(value, list) else value


def first_code(path):
    return next(source(c) for c in notebook(path)['cells'] if c['cell_type'] == 'code')


class IRWorkbookTests(unittest.TestCase):
    def test_main_sequence_and_completed_references(self):
        self.assertEqual(sorted(p.name for p in MODULE.glob('*.ipynb')), [
            '00_IR_Get_Data_NISTChemPy_v2.ipynb',
            '01_IR_Vibe_Coding_Data_Carpentry.ipynb',
            '02_IR_Optimized_Training.ipynb',
        ])
        for name in ('01_IR_Vibe_Coding_Data_Carpentry.ipynb', '03_IR_Compare_Carpentry.ipynb'):
            self.assertTrue((EXAMPLES / name).is_file())
            self.assertTrue(any(c['cell_type'] == 'code' for c in notebook(EXAMPLES / name)['cells']))
        self.assertFalse((MODULE / '03_IR_Compare_Carpentry.ipynb').exists())

    def test_one_editable_prompt_describes_actual_training_handoff(self):
        cells = [c for c in notebook(MODULE / '01_IR_Vibe_Coding_Data_Carpentry.ipynb')['cells']
                 if source(c).strip()]
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]['cell_type'], 'markdown')
        prompts = re.findall(r'```text\n(.*?)\n```', source(cells[0]), re.DOTALL)
        self.assertEqual(len(prompts), 1)
        text = prompts[0]
        for expected in (
            '[EDIT THIS:', 'Execute and debug', 'ir_raw_checkpoint.parquet',
            'IR_train.parquet', 'IR_validation.parquet', 'IR_test.parquet',
            'data_manifest.json', 'cleaning_audit.csv', 'outputs/workshop_carpentry/',
            '500-point', 'ordered 12 binary labels', 'training_pool', 'save_carpentry',
            'validate_frozen_evaluation', 'validate_tables', 'byte-for-byte',
            'never regenerate or resplit', 'actual first code cell',
            'Reference files and outputs/carpentry/ are not output destinations',
            "CARPENTRY_DIR = Path('outputs/workshop_carpentry')",
            'model/training settings stay unchanged', 'report limitations honestly',
            'first code cell of 02_IR_Optimized_Training.ipynb',
            'examples/01_IR_Vibe_Coding_Data_Carpentry.ipynb', 'ir_carpentry.py',
            'examples/README.md',
        ):
            with self.subTest(requirement=expected):
                self.assertIn(expected, text)
        self.assertLessEqual(len(text.split()), 430)
        self.assertNotIn(', '.join(carpentry.LABELS), text)  # The actual schema stays in 02/helpers.

    def test_editable_plan_suggests_label_distribution_and_native_measurement_demonstrations(self):
        text = source(notebook(MODULE / '01_IR_Vibe_Coding_Data_Carpentry.ipynb')['cells'][0])
        text = text.split('========== END EDITABLE SECTION ==========', 1)[0]
        for requirement in (
            'functional-group counts/percentages', 'rare groups', 'labels per molecule',
            'co-occurrence', 'labels overlap', 'distinct molecules', 'before/after carpentry',
            'raw x-range', 'y-range', 'unit variability', 'percent versus fractional units',
            'transmittance versus absorbance', 'do not guess units or fabricate absent categories',
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, text)

    def test_editable_fields_are_separated_from_the_fixed_trainer_contract(self):
        text = source(notebook(MODULE / '01_IR_Vibe_Coding_Data_Carpentry.ipynb')['cells'][0])
        prompt = re.search(r'```text\n(.*?)\n```', text, re.DOTALL).group(1)
        editable, fixed = prompt.split('========== END EDITABLE SECTION ==========', 1)
        self.assertIn('EDITABLE: YOUR APPROACH', editable)
        for field in ('My hypothesis', 'My plan', 'Vibe-coding strategy', 'Focus the effort',
                      'Compute budget and extra questions'):
            self.assertIn(field, editable)
        self.assertEqual(editable.count('[EDIT THIS:'), 4)
        self.assertIn('[OPTIONAL EDIT:', editable)
        self.assertNotIn('[EDIT THIS:', fixed)
        self.assertEqual(re.findall(r'FIXED: ([A-Z ]+) =', fixed),
                         ['INPUT CONTRACT', 'OUTPUT CONTRACT'])
        for requirement in ('IR_train.parquet', 'data/frozen_evaluation/',
                            '02_IR_Optimized_Training.ipynb',
                            "CARPENTRY_DIR = Path('outputs/workshop_carpentry')"):
            self.assertIn(requirement, fixed)
        self.assertIn('Copy the **entire prompt block**', text)
        self.assertIn('Starting plan (edit freely):', editable)
        self.assertIn('not a prescribed method', editable)
        for method in ('training_pool', 'prepare_training', 'save_carpentry',
                       'validate_tables', 'validate_frozen_evaluation', 'Smoke-test',
                       'report limitations honestly', 'Code Summary:'):
            with self.subTest(method=method):
                self.assertIn(method, editable)
                self.assertNotIn(method, fixed)

    def test_download_instructions_are_prominent_and_use_the_real_checkpoint_path(self):
        release = 'https://github.com/thiasyoung/2026-avs-ml-vibe-coding-workshop/releases/tag/workshop-data-v1'
        destination = '02_ir_functional_groups/outputs/nist_acquisition/ir_raw_checkpoint.parquet'
        pages = [ROOT / 'README.md', MODULE / 'README.md', ROOT / 'SETUP_GUIDE.md', ROOT / 'index.qmd',
                 MODULE / '00_IR_Get_Data_NISTChemPy_v2.ipynb',
                 MODULE / '01_IR_Vibe_Coding_Data_Carpentry.ipynb']
        for page in pages:
            text = page.read_text(encoding='utf-8')
            if page.suffix == '.ipynb':
                text = '\n'.join(source(cell) for cell in json.loads(text)['cells'])
            with self.subTest(page=page.relative_to(ROOT)):
                self.assertIn(release, text)
                self.assertIn(destination, text)
                self.assertIn('Source code', text)
        for page in pages[:3]:
            self.assertIn('> [!IMPORTANT]', page.read_text(encoding='utf-8'))
        self.assertIn('callout-important', (ROOT / 'index.qmd').read_text(encoding='utf-8'))

    def test_reference_paths_work_from_repository_module_and_examples(self):
        for name in ('01_IR_Vibe_Coding_Data_Carpentry.ipynb', '03_IR_Compare_Carpentry.ipynb'):
            tree = ast.parse(first_code(EXAMPLES / name))
            selected = [node for node in tree.body if
                        isinstance(node, ast.FunctionDef) and node.name == 'find_module_directory' or
                        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                            t.id in {'MODULE', 'OUTPUT_DIR', 'OUTPUT'} for t in node.targets)]
            setup = compile(ast.Module(body=selected, type_ignores=[]), name, 'exec')
            for directory in (ROOT, MODULE, EXAMPLES):
                with self.subTest(notebook=name, cwd=directory), patch.object(Path, 'cwd', return_value=directory):
                    scope = {'Path': Path}
                    exec(setup, scope)
                    self.assertEqual(scope['MODULE'], MODULE)
                    output = scope.get('OUTPUT_DIR', scope.get('OUTPUT'))
                    self.assertEqual(output.parent, MODULE / 'outputs')
                    self.assertEqual(module_directory(), MODULE)

    def test_batch_selectors_ignore_the_students_current_dataset(self):
        for current in ("Path('outputs/carpentry')", "Path('outputs/workshop_carpentry')",
                        'Path("outputs/my_experiment")'):
            loader = f"# CARPENTRY_DIR is configurable.\nCARPENTRY_DIR = {current}  # keep comment\n"
            chosen = select_dataset(loader, 'examples/outputs/01a_omit_transmittance')
            scope = {'Path': Path}
            exec(chosen, scope)
            self.assertEqual(scope['CARPENTRY_DIR'], Path('examples/outputs/01a_omit_transmittance'))
            self.assertIn('# keep comment', chosen)
        for loader in ('nothing = 1', "CARPENTRY_DIR = Path('a')\nCARPENTRY_DIR = Path('b')"):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                select_dataset(loader, 'examples/outputs/01b_convert_transmittance')

    def test_batch_and_verifier_use_completed_references_not_the_prompt(self):
        for name in ('run_carpentry_comparison.py', 'verify_carpentry_comparison.py'):
            text = (EXAMPLES / name).read_text(encoding='utf-8')
            self.assertIn('examples/01_IR_Vibe_Coding_Data_Carpentry.ipynb', text)
            self.assertIn('examples/03_IR_Compare_Carpentry.ipynb', text)
            self.assertNotIn("MODULE / '03_IR_Compare_Carpentry.ipynb'", text)

    def test_saved_contract_loads_in_the_actual_static_trainer_without_training(self):
        # These deliberately synthetic schema fixtures are not scientific results.
        grid = np.asarray(json.loads((MODULE / 'data/frozen_evaluation/evaluation_contract.json')
                                    .read_text(encoding='utf-8'))['wavenumber'])
        records = pd.DataFrame([
            dict(NIST_molecule=f'fixture-{index}', CAS_NO=f'fixture-{index}', Name='schema fixture',
                 molform='', SMILES=smiles, x_raw=[500., 2000., 4000.], y_raw=signal,
                 xunits='1/CM', yunits='ABSORBANCE', **{label: 1 for label in carpentry.LABELS})
            for index, (smiles, signal) in enumerate([
                ('CCO', [.1, .9, .2]), ('CCN', [.2, .1, .8]),
                ('CC', [.1, .4, .2]), ('CCC', [.3, .2, .9]),
            ])
        ])
        with tempfile.TemporaryDirectory() as directory:
            module = Path(directory)
            (module / 'lib').mkdir()
            import shutil
            shutil.copy2(MODULE / 'lib/ir_carpentry.py', module / 'lib/ir_carpentry.py')
            frozen = module / 'data/frozen_evaluation'
            frozen.mkdir(parents=True)
            prepared, _ = carpentry.prepare_training(records, grid)
            holdouts = {'validation': prepared.iloc[[2]].copy(), 'test': prepared.iloc[[3]].copy()}
            for name, frame in holdouts.items():
                frame.to_parquet(frozen / f'IR_{name}.parquet', index=False)
            hashes = {name: hashlib.sha256((frozen / f'IR_{name}.parquet').read_bytes()).hexdigest()
                      for name in holdouts}
            contract = dict(version=1, origin='synthetic input-contract smoke test',
                            labels=carpentry.LABELS, wavenumber=grid.tolist(), sha256=hashes)
            (frozen / 'evaluation_contract.json').write_text(json.dumps(contract), encoding='utf-8')
            raw = module / 'outputs/nist_acquisition/ir_raw_checkpoint.parquet'
            raw.parent.mkdir(parents=True)
            records.to_parquet(raw, index=False)
            native, source_file = carpentry.load_native(module)
            pool, excluded = carpentry.training_pool(native, holdouts)
            self.assertEqual(excluded, 2)
            train, audit = carpentry.prepare_training(pool, grid)
            output = module / 'outputs/workshop_carpentry'
            manifest = carpentry.save_carpentry(module, output, train, source_file,
                {'fixture': 'schema verification, not model training'}, audit, len(native))
            tables = {name: pd.read_parquet(output / f'IR_{name}.parquet')
                      for name in ('train', 'validation', 'test')}
            validate_tables(tables, grid)
            carpentry.validate_frozen_evaluation(module, output, grid)
            self.assertEqual(manifest['feature_count'], 500)
            self.assertEqual(manifest['frozen_holdout_sha256'], hashes)
            self.assertTrue((output / 'cleaning_audit.csv').is_file())
            loader = select_dataset(first_code(MODULE / '02_IR_Optimized_Training.ipynb'),
                                    'outputs/workshop_carpentry')
            with chdir(module), redirect_stdout(io.StringIO()):
                scope = {}
                exec(compile(loader, 'IR 02 actual loader smoke test', 'exec'), scope)
                self.assertEqual(scope['X_train'].shape, (2, 500))
                self.assertEqual(scope['y_train'].shape, (2, 12))
                np.testing.assert_allclose(scope['scaler'].mean_, scope['X_train_raw'].mean(axis=0),
                                           atol=1e-7)
                np.testing.assert_allclose(scope['X_validation'],
                    scope['scaler'].transform(scope['X_validation_raw']), atol=1e-7)
                # A malformed student handoff must fail before any model is created.
                broken = tables['train'].copy()
                broken[carpentry.LABELS[0]] = 0
                broken.to_parquet(output / 'IR_train.parquet', index=False)
                with self.assertRaisesRegex(ValueError, 'without a positive example'):
                    exec(compile(loader, 'IR 02 missing-label smoke test', 'exec'), {})


if __name__ == '__main__':
    unittest.main()
