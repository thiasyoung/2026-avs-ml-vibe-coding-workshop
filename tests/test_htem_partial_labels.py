"""Regression checks for expanded observations, frozen holdouts and local temperature."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

MODULE = Path(__file__).resolve().parents[1] / '03_htem_conductivity'
sys.path.insert(0, str(MODULE / 'lib'))
import htem_workshop as h
import htem_experiments as exp
from test_htem_process_emulator import example_record


class PartialLabelTests(unittest.TestCase):
    def test_partial_labels_survive_but_neither_label_does_not(self):
        rows = [example_record(i, i) for i in range(1, 5)]
        rows[0]['thickness'] = None
        rows[1]['fpm_conductivity'] = None
        rows[2].update(thickness=0, fpm_conductivity=-1)
        clean, audit = h.cleanse(pd.DataFrame(rows))
        self.assertEqual(clean.id.tolist(), [1, 2, 4])
        self.assertTrue(np.isnan(clean.iloc[0].thickness_um))
        self.assertTrue(np.isnan(clean.iloc[1].conductivity_S_cm))
        self.assertTrue(audit.iloc[2].excluded)

    def test_masked_loss_equal_weights_properties_and_has_zero_missing_gradient(self):
        estimate = torch.tensor([[1., 100.], [3., 4.]], requires_grad=True)
        target = torch.tensor([[0., float('nan')], [0., 0.]])
        loss = h.masked_mse(estimate, target)
        self.assertEqual(float(loss.detach()), (5 + 16) / 2)
        loss.backward()
        self.assertEqual(float(estimate.grad[0, 1]), 0.)
        self.assertTrue(torch.isfinite(estimate.grad).all())
        np.testing.assert_allclose(h.observed_mse(estimate.detach().numpy(), target.numpy()), [5, 16])

    def test_ridge_each_property_fits_all_and_only_observed_labels(self):
        x = np.arange(12, dtype=float).reshape(6, 2)
        y = np.column_stack([np.arange(6), np.arange(6) * 3.]).astype(float)
        y[[0, 1], 0] = np.nan
        y[[3, 4, 5], 1] = np.nan
        model = exp.fit_observed_regressors(Ridge(alpha=1), x, y)
        for j in range(2):
            known = np.isfinite(y[:, j])
            ref = Ridge(alpha=1).fit(x[known], y[known, j])
            np.testing.assert_allclose(model.predict(x)[:, j], ref.predict(x))

    def test_local_temperature_varies_by_node_and_is_not_broadcast_from_first(self):
        row = example_record()
        second = {**row, 'absolute_temp_c': 450}
        pd.testing.assert_series_equal(pd.Series(h.engineer_recipe(row)), pd.Series(h.engineer_recipe(second)))
        first = h.engineer_observation(row)
        self.assertEqual(first['local_temperature_C'], 300.)
        self.assertEqual(h.engineer_observation(second)['local_temperature_C'], 450.)
        grid = pd.DataFrame({'position': [1, 2], 'x_mm': [4., 8.], 'y_mm': [2., 2.],
                             'local_temperature_C': [250., 450.]})
        recipe = {k: row[k] for k in h.RECIPE_FIELDS}
        _, features = h.inference_points(recipe, grid)
        np.testing.assert_array_equal(features.local_temperature_C, [250, 450])
        expanded = h.grid_feature_frame(pd.DataFrame([first]), grid)
        np.testing.assert_array_equal(expanded.local_temperature_C, [250, 450])
        _, unknown = h.inference_points(recipe, grid.drop(columns='local_temperature_C'))
        self.assertTrue(unknown.local_temperature_C.isna().all())

    def test_expansion_preserves_original_rows_labels_and_group_assignments(self):
        raw = pd.DataFrame([example_record(i, i) for i in range(1, 31)])
        with tempfile.TemporaryDirectory() as directory:
            old, _, _ = h.prepare(raw, Path(directory) / 'old', 'old')
            anchors = old[['id', 'sample_library_id', 'position', 'x_mm', 'y_mm', 'split',
                           'thickness_um', 'conductivity_S_cm']].copy()
            anchors.id = anchors.id.astype(str)
            extra = pd.DataFrame([example_record(i, i) for i in range(31, 61)])
            extra.loc[extra.index % 2 == 0, 'thickness'] = np.nan
            extra.loc[extra.index % 2 == 1, 'fpm_conductivity'] = np.nan
            new, _, manifest = h.prepare(pd.concat([raw, extra]), Path(directory) / 'new', 'new', frozen_split=anchors)
            self.assertEqual(len(new), 60)
            pd.testing.assert_series_equal(new.set_index('id').loc[old.id, 'split'], old.set_index('id')['split'])
            self.assertEqual(new.legacy_evaluation.sum(), len(old))
            self.assertEqual(manifest['frozen_evaluation']['counts'], old.split.value_counts().to_dict())
            data, transforms, _ = h.load_prepared(Path(directory) / 'new')
            for a in data.values():
                np.testing.assert_array_equal(a['y_mask'], np.isfinite(a['y']))
            np.testing.assert_allclose(transforms['target_scaler'].mean_, new.loc[new.split.eq('train'), h.TARGETS].mean())
            damaged = pd.concat([raw, extra]).copy()
            damaged.loc[damaged.id.eq(1), 'thickness'] *= 2
            with self.assertRaises(AssertionError):
                h.prepare(damaged, Path(directory) / 'bad', 'bad', frozen_split=anchors)

    def test_exact_kernel_rejects_unconverged_linear_system(self):
        x = np.arange(6.).reshape(3, 2)
        with patch('scipy.sparse.linalg.cg', return_value=(np.zeros(3), 4)):
            with self.assertRaisesRegex(RuntimeError, 'did not converge'):
                exp.ExactRBFRegressor(alpha=1, gamma=.1, kernel='rbf').fit_kernel(x, np.ones(3), np.eye(3))


if __name__ == '__main__':
    unittest.main()
