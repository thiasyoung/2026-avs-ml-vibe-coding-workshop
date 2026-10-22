"""Algorithm-comparison contract: same data, validation-only KRR, isolated outputs."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

MODULE = Path(__file__).resolve().parents[1] / "03_htem_conductivity"
sys.path.insert(0, str(MODULE))
import htem_workshop as htem
import htem_graph as gnn
import htem_experiments as exp


class HTEMExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.module = Path(cls.temp.name) / "module"
        acquisition = cls.module / "outputs/htem_acquisition"
        acquisition.mkdir(parents=True)
        rows = []
        for library in range(1, 21):
            for point in range(3):
                row = {key: None for key in htem.RAW_FIELDS}
                row.update(id=f"{library}:{point}", sample_library_id=library, position=point,
                           x_mm=point * 4.0, y_mm=0, pdac="tool", deposition_sample_time_min=30 + library,
                           deposition_initial_temp_c=300, deposition_compounds=["ZnO"], deposition_power=[50],
                           deposition_gases=["Ar"], deposition_gas_flow_sccm=[10],
                           deposition_substrate_material="glass", thickness=.3 + point * .01,
                           fpm_conductivity=10 + library if point < 2 else None)
                rows.append(row)
        raw = pd.DataFrame(rows)
        raw.to_parquet(acquisition / "htem_process_raw.parquet", index=False)
        htem.prepare(raw, cls.module / "outputs/htem_prepared", "fixture")
        cls.context = exp.load_context(cls.module)
        training = cls.context.data["train"]
        linear = exp.fit_observed_regressors(Ridge(alpha=10), training["X"], training["y"])
        kernel, cls.trials = exp.fit_kernel_ridge(cls.context, [.1, 1], [.01, .1])
        mlp, history, note = htem.fit_neural_head(training["X"], training["y"],
            cls.context.data["validation"]["X"], cls.context.data["validation"]["y"], hidden=(8,), epochs=3, patience=2)
        graphs = exp.ensure_graphs(cls.context)
        graph, graph_history, graph_note = gnn.fit_graph_model(graphs["train"], graphs["validation"],
            hidden_dim=8, message_layers=1, epochs=3, patience=2)
        forest = exp.fit_observed_regressors(RandomForestRegressor(
            n_estimators=12, max_depth=5, min_samples_leaf=2, random_state=42, n_jobs=1),
            training["X"], training["y"])
        cls.models = dict(linear_ridge=linear, kernel_ridge=kernel, mlp=mlp, grid_gnn=graph, random_forest=forest)
        for key, model in cls.models.items():
            exp.evaluate_and_save(cls.context, model, key, {"test_fixture": True}, .1,
                history=history if key == "mlp" else graph_history if key == "grid_gnn" else None,
                training_note=note if key == "mlp" else graph_note if key == "grid_gnn" else None,
                trials=cls.trials if key == "kernel_ridge" else None)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_runs_have_same_rows_and_preparation_without_overwriting(self):
        comparison = exp.collect_comparison(self.context)
        self.assertEqual(set(comparison.experiment_id), set(exp.EXPERIMENTS))
        self.assertEqual(len(comparison), len(exp.EXPERIMENTS))
        self.assertEqual(comparison.map_library_id.nunique(), 1)
        self.assertTrue(comparison.validation_score.is_monotonic_increasing)
        self.assertEqual(self.context.fingerprint, exp.load_context(self.module).fingerprint)
        for key in exp.EXPERIMENTS:
            self.assertTrue((self.module / "outputs/htem_architectures" / key / "model.joblib").exists())

    def test_kernel_is_exact_multioutput_and_selected_only_by_validation(self):
        kernel = self.models["kernel_ridge"]
        self.assertIsInstance(kernel, exp.ObservedTargetRegressors)
        for index, estimator in enumerate(kernel.estimators_):
            self.assertIsInstance(estimator, KernelRidge)
            self.assertEqual(estimator.kernel, "rbf")
            known = np.isfinite(self.context.data['train']['y'][:, index])
            self.assertEqual(len(estimator.X_fit_), int(known.sum()))
            self.assertEqual(estimator.X_fit_.dtype, np.dtype("float64"))
            self.assertEqual(estimator.dual_coef_.dtype, np.dtype("float64"))
            reference = KernelRidge(alpha=kernel.alpha, gamma=kernel.gamma, kernel='rbf').fit(
                estimator.X_fit_, self.context.data['train']['y'][known, index].astype('float64'))
            np.testing.assert_allclose(estimator.predict(estimator.X_fit_), reference.predict(estimator.X_fit_), atol=1e-7)
            self.assertLessEqual(estimator.solver_relative_residual_, 1.1e-8)
        self.assertAlmostEqual(exp.validation_score(self.context, kernel), self.trials.validation_score.min(), places=6)
        without_test = replace(self.context, data={**self.context.data, "test": None})
        model, _ = exp.fit_kernel_ridge(without_test, [.1, 1], [.01, .1])
        self.assertEqual((kernel.alpha, kernel.gamma), (model.alpha, model.gamma))

    def test_kernel_predictions_are_batch_invariant_without_changing_prepared_values(self):
        kernel = self.models["kernel_ridge"]
        X = self.context.data["validation"]["X"]
        promoted = exp.solver_inputs(kernel, X)
        np.testing.assert_array_equal(promoted, X)
        whole = exp.predict_scaled(self.context, kernel, "validation")
        separate = np.vstack([exp.grid_prediction(kernel, row[None, :], None) for row in X])
        np.testing.assert_allclose(separate, whole, rtol=1e-10, atol=1e-10)
        self.assertEqual(X.dtype, np.dtype("float32"))

    def test_kernel_search_rejects_invalid_hyperparameters(self):
        for alphas, gammas in [([], [.1]), ([0], [.1]), ([1], [np.nan])]:
            with self.assertRaises(ValueError):
                exp.fit_kernel_ridge(self.context, alphas, gammas)

    def test_random_forest_fits_each_property_only_on_observed_training_labels(self):
        forest = self.models["random_forest"]
        training = self.context.data["train"]
        self.assertIsNone(forest.alpha)
        for index, estimator in enumerate(forest.estimators_):
            self.assertIsInstance(estimator, RandomForestRegressor)
            known = np.isfinite(training["y"][:, index])
            reference = RandomForestRegressor(**estimator.get_params()).fit(
                training["X"][known], training["y"][known, index])
            np.testing.assert_allclose(estimator.predict(training["X"]), reference.predict(training["X"]))
            self.assertEqual(estimator.n_features_in_, training["X"].shape[1])
        notebook = nbformat.read(MODULE / exp.EXPERIMENTS["random_forest"][1], 4)
        source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
        self.assertIn("fit_observed_regressors(RandomForestRegressor(**configuration), X_train, y_train)", source)
        self.assertNotIn(".fit(X_val", source)

    def test_all_reloaded_models_predict_full_grid_without_measured_inputs(self):
        for key in exp.EXPERIMENTS:
            output = self.module / "outputs/htem_architectures" / key
            saved = joblib.load(output / "model.joblib")
            full = pd.read_csv(output / "heldout_map.csv")
            recipe = exp.example_recipe(self.context, int(full.library_id.iloc[0]))
            grid = full[["position", "x_mm", "y_mm"]]
            actual = exp.predict_substrate(saved, recipe, grid)
            self.assertEqual(len(actual), 3)  # Two labeled locations plus one unlabeled.
            np.testing.assert_allclose(actual[["predicted_thickness_um", "predicted_conductivity_S_cm"]],
                                       full[["predicted_thickness_um", "predicted_conductivity_S_cm"]], rtol=5e-5)
            with self.assertRaises(ValueError):
                exp.predict_substrate(saved, recipe, grid.assign(thickness=1))

    def test_point_models_predict_each_location_independently_of_query_grid(self):
        """One point, shuffled rows, and a batch must give the same local predictions."""
        columns = ["predicted_thickness_um", "predicted_conductivity_S_cm"]
        for key in ["linear_ridge", "kernel_ridge", "mlp", "random_forest"]:
            with self.subTest(model=key):
                output = self.module / "outputs/htem_architectures" / key
                saved = joblib.load(output / "model.joblib")
                full = pd.read_csv(output / "heldout_map.csv")
                recipe = exp.example_recipe(self.context, int(full.library_id.iloc[0]))
                grid = full[["position", "x_mm", "y_mm", "local_temperature_C"]]
                batch = exp.predict_substrate(saved, recipe, grid)
                separate = pd.concat([exp.predict_substrate(saved, recipe, grid.iloc[[i]])
                                      for i in range(len(grid))], ignore_index=True)
                pd.testing.assert_frame_equal(batch[grid.columns], separate[grid.columns])
                np.testing.assert_allclose(batch[columns], separate[columns], rtol=5e-5, atol=1e-7)
                shuffled = grid.sample(frac=1, random_state=42).reset_index(drop=True)
                reordered = exp.predict_substrate(saved, recipe, shuffled)
                pd.testing.assert_frame_equal(reordered[grid.columns], shuffled)
                np.testing.assert_allclose(
                    batch.sort_values("position")[columns],
                    reordered.sort_values("position")[columns], rtol=5e-5, atol=1e-7)

    def test_map_reconstruction_places_predictions_at_xy_without_changing_values(self):
        """Grid reconstruction is plotting only, independent of incoming row order."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        table = pd.DataFrame({
            "library_id": [1, 1, 1], "position": [90, 3, 17],
            "x_mm": [4., 0., 0.], "y_mm": [12.5, 0., 12.5],
            "predicted_thickness_um": [.7, .2, .4],
            "predicted_conductivity_S_cm": [70., 20., 40.],
        }).sample(frac=1, random_state=42).reset_index(drop=True)
        unchanged = table.copy(deep=True)
        figure = htem.plot_substrate_maps(table)
        try:
            # Ascending y, then x; the unrequested (4, 0) cell must stay blank.
            for axis, values in zip(figure.axes[:2], [[.2, np.nan, .4, .7], [20., np.nan, 40., 70.]]):
                mesh = axis.collections[0].get_array()
                np.testing.assert_allclose(np.ma.filled(mesh, np.nan).ravel(), values, equal_nan=True)
            pd.testing.assert_frame_equal(table, unchanged)
        finally:
            plt.close(figure)

    def isolated_copy(self, destination):
        module = Path(destination) / "copy"
        shutil.copytree(self.module, module)
        return exp.load_context(module)

    def test_missing_run_reports_which_notebook_to_run(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx = self.isolated_copy(temp)
            (ctx.module / "outputs/htem_architectures/kernel_ridge/experiment.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "02a_HTEM_Kernel_Ridge"):
                exp.collect_comparison(ctx)

    def test_stale_or_different_data_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx = self.isolated_copy(temp)
            marker = ctx.module / "outputs/htem_architectures/mlp/experiment.json"
            value = json.loads(marker.read_text())
            value["fingerprint"]["seed"] = 999
            htem.write_json(marker, value)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                exp.collect_comparison(ctx)

    def test_changed_result_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx = self.isolated_copy(temp)
            path = ctx.module / "outputs/htem_architectures/linear_ridge/predictions.csv"
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "artifact"):
                exp.collect_comparison(ctx)

    def test_even_rechecksummed_different_labeled_points_cannot_be_compared(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx = self.isolated_copy(temp)
            output = ctx.module / "outputs/htem_architectures/linear_ridge"
            path = output / "predictions.csv"
            points = pd.read_csv(path)
            points.loc[0, "position"] = 99999
            points.to_csv(path, index=False)
            marker = output / "experiment.json"
            value = json.loads(marker.read_text())
            value["artifacts_sha256"][path.name] = htem.sha256(path)
            htem.write_json(marker, value)
            with self.assertRaisesRegex(ValueError, "Different labeled point"):
                exp.collect_comparison(ctx)

    def test_preparation_changed_during_fit_does_not_certify_stale_results(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx = self.isolated_copy(temp)
            path = ctx.prepared / "manifest.json"
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "Preparation changed"):
                exp.evaluate_and_save(ctx, self.models["linear_ridge"], "linear_ridge", {}, .1)

    def test_default_notebook_is_linear_and_comparison_does_not_fit_models(self):
        default = nbformat.read(MODULE / exp.EXPERIMENTS["linear_ridge"][1], 4)
        default_code = "\n".join(c.source for c in default.cells if c.cell_type == "code")
        self.assertIn("fit_observed_regressors(Ridge(alpha=ALPHA)", default_code)
        self.assertNotIn("fit_kernel_ridge(", default_code)
        self.assertNotIn("fit_graph_model(", default_code)
        comparison = nbformat.read(MODULE / "03_HTEM_Compare_Architectures.ipynb", 4)
        comparison_code = "\n".join(c.source for c in comparison.cells if c.cell_type == "code")
        self.assertIn("collect_comparison", comparison_code)
        self.assertNotIn(".fit(", comparison_code)
        self.assertNotIn("prepare(", comparison_code)
        for _, relative in exp.EXPERIMENTS.values():
            self.assertTrue((MODULE / relative).exists())

    def test_comparison_parity_has_all_test_points_and_shared_property_limits(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from unittest.mock import patch
        notebook = nbformat.read(MODULE / '03_HTEM_Compare_Architectures.ipynb', 4)
        source = next(c.source for c in notebook.cells
                      if c.cell_type == 'code' and c.source.startswith('# Test-set parity:'))
        with tempfile.TemporaryDirectory() as output, patch.object(plt, 'show'):
            namespace = dict(np=np, pd=pd, plt=plt, HERE=self.module, COMPARISON=Path(output),
                             comparison=exp.collect_comparison(self.context))
            try:
                exec(compile(source, '<test-set parity cell>', 'exec'), namespace)
                axes = namespace['axes']
                self.assertEqual(axes.shape, (len(exp.EXPERIMENTS), 2))
                for column in range(2):
                    for ax in axes[:, column]:
                        target = 'actual_thickness_um' if column == 0 else 'actual_conductivity_S_cm'
                        self.assertEqual(len(ax.collections[0].get_offsets()), int(namespace['table'][target].notna().sum()))
                        self.assertEqual(ax.get_xlim(), axes[0, column].get_xlim())
                        self.assertEqual(ax.get_ylim(), ax.get_xlim())
                first = next(iter(namespace['test_predictions'].values()))
                np.testing.assert_allclose(axes[0, 1].collections[0].get_offsets()[:, 0],
                                           np.log10(first.actual_conductivity_S_cm.dropna()))
                self.assertTrue((Path(output) / 'test_set_parity.png').exists())
            finally:
                plt.close('all')


if __name__ == "__main__":
    unittest.main()
