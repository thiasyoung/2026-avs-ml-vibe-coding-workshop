"""Native signal/bounds policies and immutable holdout regression checks."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / '02_ir_functional_groups'
spec = importlib.util.spec_from_file_location('ir_native_contract', MODULE / 'lib/ir_carpentry.py')
carpentry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(carpentry)


def native_row(**changes):
    row = dict(NIST_molecule='synthetic', CAS_NO='synthetic', Name='synthetic fixture', molform='',
               SMILES='CCO', x_raw=[1000., 1500., 2000.], y_raw=[0., .5, 1.],
               xunits='1/CM', yunits='ABSORBANCE', **{label: 1 for label in carpentry.LABELS})
    row.update(changes)
    return pd.Series(row)


class NativeCarpentryTests(unittest.TestCase):
    def test_naive_baseline_preserves_recorded_signal_without_unit_changes(self):
        row = native_row(yunits='TRANSMITTANCE', y_raw=[.1, .2, .3])
        x, y, actions = carpentry.native_arrays(row, 'as_recorded')
        np.testing.assert_array_equal(x, row.x_raw)
        np.testing.assert_array_equal(y, row.y_raw)
        self.assertEqual(actions, ['as_recorded_no_unit_handling'])

    def test_baseline_does_not_discuss_or_branch_on_signal_representations(self):
        notebook = json.loads((MODULE / 'examples/01_IR_Vibe_Coding_Data_Carpentry.ipynb').read_text(encoding='utf-8'))
        source = '\n'.join(''.join(c['source']) for c in notebook['cells']).lower()
        self.assertNotIn('transmittance', source)
        self.assertNotIn('absorbance', source)
        self.assertIn("signal_policy='as_recorded'", source)

    def test_sparse_upsampling_and_dense_bin_downsampling_are_finite_and_uniform(self):
        grid = np.linspace(1000, 2000, 500)
        values, padded = carpentry.regrid(np.array([1000, 2000]), np.array([0, 1]), grid,
                                         resampling='bin_average')
        np.testing.assert_allclose(values, np.linspace(0, 1, 500))
        self.assertEqual(padded, 0)
        dense = np.linspace(1, 11, 1001)
        signal = .01 * dense + np.maximum(0, 1 - np.abs(dense - 5.5) / .2)
        coarse = np.arange(1, 12)
        linear, _ = carpentry.regrid(dense, signal, coarse)
        averaged, _ = carpentry.regrid(dense, signal, coarse, resampling='bin_average')
        self.assertFalse(np.allclose(linear, averaged))
        self.assertEqual(averaged.shape, coarse.shape)
        self.assertTrue(np.isfinite(averaged).all())
        self.assertEqual(float(averaged.min()), 0)
        self.assertEqual(float(averaged.max()), 1)

    def test_preparation_records_actual_upsampling_and_downsampling(self):
        rows = pd.DataFrame([native_row(NIST_molecule='sparse', x_raw=[1000, 2000], y_raw=[0, 1]),
                             native_row(NIST_molecule='dense', x_raw=np.linspace(1000, 2000, 1000),
                                        y_raw=np.linspace(0, 1, 1000))])
        prepared, _ = carpentry.prepare_training(rows, np.linspace(1000, 2000, 500), resampling='bin_average')
        self.assertEqual(prepared.resampling_action.tolist(), ['upsampled_linear', 'downsampled_bin_average'])
        self.assertEqual(prepared.IR.map(len).tolist(), [500, 500])

    def test_two_stage_resampling_restores_dimensions_not_native_values(self):
        x = np.linspace(1000, 2000, 1500)
        y = .001 * x + np.sin(x / 3)**2
        grid = np.linspace(1000, 2000, 500)
        values, _ = carpentry.regrid(x, y, grid, resampling='bin_average_then_upsample')
        linear, _ = carpentry.regrid(x, y, grid)
        self.assertEqual(values.shape, (500,))
        self.assertTrue(np.isfinite(values).all())
        self.assertFalse(np.allclose(values, linear))

    def test_absorbance_fraction_and_explicit_percent_agree(self):
        reference = native_row()
        x, a, _ = carpentry.native_arrays(reference)
        for units, scale in [('TRANSMITTANCE', 1), ('PERCENT TRANSMITTANCE', 100)]:
            row = native_row(yunits=units, y_raw=scale * 10 ** -a,
                             xunits='MICROMETERS', x_raw=1e4 / x)
            converted_x, converted_a, _ = carpentry.native_arrays(row, 'convert_transmittance')
            np.testing.assert_allclose(converted_x, x)
            np.testing.assert_allclose(converted_a, a)

    def test_absorbance_policy_does_not_accept_transmittance_or_reflectance(self):
        for units in ('TRANSMITTANCE', 'REFLECTANCE'):
            with self.subTest(units=units), self.assertRaises(ValueError):
                carpentry.native_arrays(native_row(yunits=units))

    def test_conversion_never_guesses_percent_or_clips_bad_values(self):
        for values in ([0, .5, 1], [-.1, .5, 1], [.1, .5, 1.1], [1, 50, 100], [.1, np.nan, 1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                carpentry.native_arrays(native_row(yunits='TRANSMITTANCE', y_raw=values), 'convert_transmittance')

    def test_crop_and_interpolation_are_in_carpentry(self):
        y, padded = carpentry.regrid(np.array([500, 1000, 1500, 2000]), np.array([0, 1, 2, 3]),
                                    np.array([1000, 1250, 1500]))
        np.testing.assert_allclose(y, [0, .5, 1])
        self.assertEqual(padded, 0)

    def test_padding_is_explicit_and_does_not_change_measured_normalization(self):
        x, y, grid = np.array([1000, 2000]), np.array([2, 4]), np.array([500, 1000, 1500, 2000, 2500])
        with self.assertRaisesRegex(ValueError, 'incomplete_measured_coverage'):
            carpentry.regrid(x, y, grid)
        padded, count = carpentry.regrid(x, y, grid, 'zero_pad', .6)
        np.testing.assert_allclose(padded, [0, 0, .5, 1, 0])
        self.assertEqual(count, 2)
        with self.assertRaisesRegex(ValueError, 'insufficient_measured_coverage'):
            carpentry.regrid(x, y, grid, 'zero_pad', .8)

    def test_heldout_equivalent_smiles_excluded_before_preparation(self):
        source = pd.DataFrame([native_row(SMILES='OCC'), native_row(NIST_molecule='other', CAS_NO='other', SMILES='CC')])
        holdouts = {'test': pd.DataFrame([native_row(SMILES='CCO', NIST_molecule='held', CAS_NO='held')])}
        pool, excluded = carpentry.training_pool(source, holdouts)
        self.assertEqual(excluded, 1)
        self.assertEqual(pool.NIST_molecule.tolist(), ['other'])

    def test_existing_holdouts_migrate_once_and_survive_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            module = Path(directory)
            baseline = module / 'outputs/carpentry'
            baseline.mkdir(parents=True)
            for name, smiles in [('validation', 'CCC'), ('test', 'CCCC')]:
                frame = pd.DataFrame([native_row(SMILES=smiles, NIST_molecule=name, CAS_NO=name)])
                frame['wavenumber'] = [[1000., 1500., 2000.]]
                frame['IR'] = [[0., .5, 1.]]
                frame.to_parquet(baseline / f'IR_{name}.parquet', index=False)
            original = {s: carpentry.sha256_file(baseline / f'IR_{s}.parquet') for s in ('validation', 'test')}
            holdouts, grid, contract = carpentry.ensure_frozen_evaluation(module, pd.DataFrame())
            self.assertEqual(contract['sha256'], original)
            # Reruns do not regenerate the benchmark even if the acquisition changes.
            _, again, repeated = carpentry.ensure_frozen_evaluation(module, pd.DataFrame([native_row()]))
            np.testing.assert_array_equal(again, grid)
            self.assertEqual(repeated, contract)
            attempt = module / 'attempt'
            attempt.mkdir()
            for name in ('validation', 'test'):
                shutil.copyfile(baseline / f'IR_{name}.parquet', attempt / f'IR_{name}.parquet')
            carpentry.validate_frozen_evaluation(module, attempt, grid)
            frame = holdouts['test'].copy()
            frame['IR'] = [[1., .5, 0.]]
            frame.to_parquet(attempt / 'IR_test.parquet', index=False)
            with self.assertRaisesRegex(ValueError, 'test differs'):
                carpentry.validate_frozen_evaluation(module, attempt, grid)
            with self.assertRaisesRegex(ValueError, 'grid or labels'):
                carpentry.validate_frozen_evaluation(module, attempt, grid + 1)

    def test_fresh_reference_splits_by_canonical_molecule_once(self):
        source = pd.DataFrame([native_row(NIST_molecule=str(i), CAS_NO=str(i), SMILES='C' * (i + 2),
                             x_raw=[500., 4000.], y_raw=[0., 1.]) for i in range(30)])
        # A second observation of the same canonical molecule must share its split.
        duplicate = source.iloc[[0]].assign(NIST_molecule='duplicate', CAS_NO='duplicate')
        source = pd.concat([source, duplicate], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            heldouts, grid, contract = carpentry.ensure_frozen_evaluation(Path(directory), source)
            pool, _ = carpentry.training_pool(source, heldouts)
            groups = [set(f.SMILES.map(carpentry.canonical_smiles)) for f in [pool, *heldouts.values()]]
            self.assertFalse(any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]))
            self.assertEqual(contract['wavenumber'], grid.tolist())
            _, _, repeated = carpentry.ensure_frozen_evaluation(Path(directory), source.sample(frac=1, random_state=3))
            self.assertEqual(repeated['sha256'], contract['sha256'])


if __name__ == '__main__':
    unittest.main()
