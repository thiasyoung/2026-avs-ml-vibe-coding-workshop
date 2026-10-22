"""Exercise the actual notebook's robust stopping slopes on controlled loss traces."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np

NOTEBOOK = Path(__file__).resolve().parents[1] / '02_ir_functional_groups/02_IR_Optimized_Training.ipynb'


class IRTrainingTemplateTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        function = next(n for n in ast.parse(code).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'robust_validation_slopes')
        namespace = {'np': np, 'VALIDATION_SLOPE_WINDOW': 20, 'TRIMMED_SLOPE_EXCLUDED_POINTS': 5}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
        self.slopes = namespace['robust_validation_slopes']

    def test_sustained_increase_survives_trimming(self):
        raw, trimmed = self.slopes(np.linspace(.1, .3, 20))
        self.assertGreater(raw, 0)
        self.assertGreater(trimmed, 0)

    def test_five_high_spikes_do_not_end_a_decreasing_trend(self):
        losses = np.linspace(1, .1, 20)
        losses[-5:] += 5
        raw, trimmed = self.slopes(losses)
        self.assertGreater(raw, 0)
        self.assertLess(trimmed, 0)

    def test_original_epoch_coordinates_are_preserved(self):
        losses = 1 + .1 * np.arange(20)
        losses[[1, 5, 9, 13, 17]] += 10
        _, trimmed = self.slopes(losses)
        self.assertAlmostEqual(trimmed, .1)

    def test_incomplete_or_nonfinite_window_is_rejected(self):
        for losses in [np.ones(19), np.r_[np.ones(19), np.nan]]:
            with self.subTest(losses=losses), self.assertRaises(ValueError):
                self.slopes(losses)


if __name__ == '__main__':
    unittest.main()
