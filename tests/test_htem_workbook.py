"""Prompt-driven HTEM model development and intact completed reference notebooks."""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / '03_htem_conductivity'
EXAMPLES = MODULE / 'examples'
WORKBOOK = '02_HTEM_Vibe_Coding_Workbook.ipynb'
BASELINE = '02_HTEM_Process_Emulator_Vibe_Coding.ipynb'
COMPARISON = '03_HTEM_Compare_Architectures.ipynb'
sys.path.insert(0, str(MODULE / 'lib'))
import htem_experiments as exp
import htem_workshop as htem


def notebook(path):
    return json.loads(path.read_text(encoding='utf-8'))


def source(cell):
    value = cell['source']
    return ''.join(value) if isinstance(value, list) else value


def first_code(path):
    return next(source(c) for c in notebook(path)['cells'] if c['cell_type'] == 'code')


class HTEMWorkbookTests(unittest.TestCase):
    def test_main_sequence_and_completed_references(self):
        self.assertEqual(sorted(p.name for p in MODULE.glob('*.ipynb')), [
            '00_HTEM_Get_Data_NREL.ipynb', '01_HTEM_Process_Data_Carpentry.ipynb', WORKBOOK,
        ])
        self.assertEqual(len(list(EXAMPLES.glob('02*.ipynb'))), 5)
        for name in (BASELINE, COMPARISON):
            self.assertFalse((MODULE / name).exists())
            self.assertTrue((EXAMPLES / name).is_file())
            self.assertTrue(any(c['cell_type'] == 'code' for c in notebook(EXAMPLES / name)['cells']))

    def test_one_copyable_prompt_with_editable_choices_and_fixed_contract(self):
        cells = [c for c in notebook(MODULE / WORKBOOK)['cells'] if source(c).strip()]
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]['cell_type'], 'markdown')
        text = source(cells[0])
        self.assertTrue(text.startswith('# 02 · HTEM:'))
        prompts = re.findall(r'```text\n(.*?)\n```', text, re.DOTALL)
        self.assertEqual(len(prompts), 1)
        editable, fixed = prompts[0].split('========== END EDITABLE SECTION ==========', 1)
        self.assertIn('EDITABLE: YOUR APPROACH', editable)
        for field in ('My hypothesis', 'My plan', 'Vibe-coding strategy', 'Focus the effort',
                      'Compute budget and search space'):
            self.assertIn(field, editable)
        self.assertEqual(editable.count('[EDIT THIS:'), 5)
        self.assertNotIn('[EDIT THIS:', fixed)
        self.assertIn('entire prompt block', text)
        self.assertIn('Starting plan (edit freely):', editable)
        self.assertIn('not a prescribed method', editable)
        self.assertEqual(re.findall(r'FIXED: ([A-Z ]+) =', fixed),
                         ['INPUT CONTRACT', 'OUTPUT CONTRACT'])
        for stage in ('Baseline', 'Hypothesis and experiment', 'Validation analysis',
                      'Final evaluation', 'Saved-model inference'):
            self.assertIn(stage, editable)
            self.assertNotIn(stage, fixed)
        for requirement in (
            'outputs/htem_prepared/', 'data/frozen_features/',
            'label masks', 'Missing targets remain NaN', 'whole-deposition/recipe splits',
            'substrate XY location', 'planned local-temperature inputs',
            'never measured film outputs or IDs',
            'standardized/log-target and inverse-transform conventions',
            'outputs/htem_workbook/<unique_experiment_id>/',
            'never outputs/htem_architectures/', '01_HTEM_Process_Data_Carpentry.ipynb',
            'thickness_um', 'conductivity_S_cm', 'deposition/position identities',
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, fixed)
        for reference in (BASELINE, COMPARISON):
            self.assertIn('examples/' + reference, editable)
        for method in (
            'ctx = exp.load_context(HERE)', 'Ridge(alpha=10)', 'validation only',
            'freeze choices before evaluating test', 'restore best validation-selected weights',
            'Do not invent epoch curves for non-neural models', 'parity plots',
            'Reload the saved model/preprocessing', 'Code Summary:', 'evaluate_and_save',
            'lib/htem_workshop.py and lib/htem_experiments.py',
        ):
            with self.subTest(method=method):
                self.assertIn(method, editable)
                self.assertNotIn(method, fixed)
        self.assertLessEqual(len(prompts[0].split()), 500)

    def test_moved_references_resolve_paths_from_all_supported_folders(self):
        for name in (BASELINE, COMPARISON):
            nodes = []
            for node in ast.parse(first_code(EXAMPLES / name)).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                        t.id in {'HERE', 'EXPERIMENT_ID', 'RESULTS', 'COMPARISON'} for t in node.targets):
                    nodes.append(node)
                elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'parent':
                    nodes.append(node)
            setup = compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec')
            for directory in (ROOT, MODULE, EXAMPLES):
                with self.subTest(reference=name, cwd=directory), patch.object(Path, 'cwd', return_value=directory):
                    scope = {'Path': Path}
                    exec(setup, scope)
                    self.assertEqual(scope['HERE'], MODULE)
                    output = scope.get('RESULTS', scope.get('COMPARISON'))
                    self.assertEqual(output.parent, MODULE / 'outputs/htem_architectures')

    def test_instructor_generator_cannot_overwrite_the_prompt_workbook(self):
        path = ROOT / 'scratch/build_htem_algorithm_notebooks.py'
        spec = importlib.util.spec_from_file_location('htem_reference_generator', path)
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        captured = {}
        with patch.object(generator.nbf, 'write', side_effect=lambda nb, path: captured.update({Path(path): nb})), \
                redirect_stdout(io.StringIO()):
            generator.build()
        self.assertEqual(len(captured), 6)
        self.assertTrue(all(path.parent == EXAMPLES for path in captured))
        self.assertIn(EXAMPLES / BASELINE, captured)
        self.assertIn(EXAMPLES / COMPARISON, captured)
        self.assertNotIn(MODULE / WORKBOOK, captured)

    def test_workbook_documentation_links_exist(self):
        for path in (MODULE / 'README.md', EXAMPLES / 'README.md'):
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                if target.startswith(('https://', 'http://', '#')):
                    continue
                self.assertTrue((path.parent / target.split('#', 1)[0]).is_file(), target)

    def test_real_prepared_context_has_the_documented_prediction_interface(self):
        ctx = exp.load_context(MODULE)
        self.assertEqual(ctx.manifest['targets'], ['log10_thickness_um', 'log10_conductivity_S_cm'])
        self.assertEqual(len(ctx.manifest['encoded_features']), ctx.data['train']['X'].shape[1])
        library_sets = []
        for split in htem.SPLITS:
            part = ctx.data[split]
            self.assertTrue(np.isfinite(part['X']).all())
            self.assertEqual(part['y'].shape, (len(part['X']), 2))
            np.testing.assert_array_equal(part['y_mask'], np.isfinite(part['y']))
            physical = htem.inverse_predictions(np.zeros((2, 2)), ctx.transforms['target_scaler'])
            self.assertEqual(physical.columns.tolist(), ['thickness_um', 'conductivity_S_cm'])
            self.assertTrue(np.isfinite(physical.to_numpy()).all())
            self.assertTrue((physical.to_numpy() > 0).all())
            library_sets.append(set(part['library_id']))
        self.assertFalse(any(a & b for i, a in enumerate(library_sets) for b in library_sets[i + 1:]))


if __name__ == '__main__':
    unittest.main()
