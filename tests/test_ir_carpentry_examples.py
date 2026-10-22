"""Small scientific-contract tests for the shared IR example utilities."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SUPPORT = Path(__file__).resolve().parents[1] / '02_ir_functional_groups/examples/ir_example_support.py'
spec = importlib.util.spec_from_file_location('ir_example_support', SUPPORT)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class IRCarpentryTests(unittest.TestCase):
    def test_sort_and_average_repeated_coordinates(self):
        grid, signal = support.parse_spectrum([2000, 1000, 2000, 3000], [.4, .1, .6, .2])
        np.testing.assert_array_equal(grid, [1000, 2000, 3000])
        np.testing.assert_allclose(signal, [.1, .5, .2])

    def test_invalid_spectra_are_not_imputed(self):
        for grid, values in [([1000, 2000], [.1, np.nan]),
                             ([1000, 2000], [.1, np.inf]),
                             ([1000], [.1]), ([1000, 2000], [.1])]:
            with self.subTest(grid=grid, values=values), self.assertRaises(ValueError):
                support.parse_spectrum(grid, values)

    def test_equivalent_smiles_training_copy_removed_without_touching_holdouts(self):
        tables = {
            'train': pd.DataFrame({'NIST_molecule': ['ethanol_copy', 'other'], 'SMILES': ['OCC', 'CC']}),
            'validation': pd.DataFrame({'SMILES': ['CCO']}),
            'test': pd.DataFrame({'SMILES': ['CCC']}),
        }
        original_validation, original_test = tables['validation'].copy(), tables['test'].copy()
        audit = []
        train = support.guard_training_identities(tables, audit)
        self.assertEqual(train.NIST_molecule.tolist(), ['other'])
        self.assertEqual(audit[0]['reason'], 'canonical_molecule_in_holdout')
        pd.testing.assert_frame_equal(tables['validation'], original_validation)
        pd.testing.assert_frame_equal(tables['test'], original_test)

    def test_stereochemistry_is_preserved(self):
        self.assertNotEqual(support.canonical_smiles('N[C@H](C)C(=O)O'),
                            support.canonical_smiles('N[C@@H](C)C(=O)O'))


if __name__ == '__main__':
    unittest.main()
