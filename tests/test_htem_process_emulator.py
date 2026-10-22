"""Small offline tests for the HTEM process-only workshop contract."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "03_htem_conductivity"
sys.path.insert(0, str(MODULE))
spec = importlib.util.spec_from_file_location("htem_workshop", MODULE / "htem_workshop.py")
htem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(htem)


def example_record(library=1, sample=1):
    return {
        "id": sample, "sample_library_id": library, "pdac": "4",
        "position": sample % 10 + 1, "x_mm": (sample % 10 + 1) * 4.0, "y_mm": 6.65,
        "absolute_temp_c": 300.0,
        **{key: None for key in htem.EXTRA_PROCESS_FIELDS},
        "deposition_sample_time_min": 30.0 + library,
        "deposition_initial_temp_c": 300.0,
        "deposition_base_pressure_mtorr": 0.001,
        "deposition_growth_pressure_mtorr": 10.0,
        "deposition_compounds": ["ZnO", "CoO", None],
        "deposition_power": [20.0, 40.0, None],
        "deposition_gases": ["Ar", "O2", None],
        "deposition_gas_flow_sccm": [10.0, 2.0, None],
        "deposition_substrate_material": " Quartz ",
        "thickness": 0.2 + library * 0.01,
        "opt_direct_bandgap": 1.0 + library * 0.01,
        "fpm_conductivity": 10.0 ** (library / 20),
    }


class HTEMPreparationTests(unittest.TestCase):
    def test_process_features_do_not_depend_on_outcomes(self):
        row = example_record()
        changed = copy.deepcopy(row)
        changed.update(thickness=7, fpm_conductivity=1e6, opt_direct_bandgap=4,
                       xrf_concentration=[0.8, 0.2], opt_energy=[1, 2], quality=5,
                       absolute_temp_c=200, position=99, id=999, sample_library_id=999)
        a, b = htem.engineer_recipe(row), htem.engineer_recipe(changed)
        pd.testing.assert_series_equal(pd.Series(a), pd.Series(b))
        self.assertEqual(set(a), set(htem.RECIPE_FEATURES))
        self.assertTrue(set(htem.LABEL_FIELDS).isdisjoint(htem.FEATURES))

    def test_source_and_gas_pairing_and_missing_semantics(self):
        features = htem.engineer_recipe(example_record())
        self.assertEqual(features["target_2"], "CoO")
        self.assertEqual(features["target_2_power_W"], 40)
        self.assertEqual(features["target_3_power_W"], 0)
        self.assertEqual(features["gas_1"], "ARGON")
        self.assertEqual(features["gas_2"], "OXYGEN")
        self.assertEqual(features["gas_2_flow_sccm"], 2)
        self.assertTrue(np.isnan(features["gas_3_flow_sccm"]))
        with self.assertRaisesRegex(ValueError, "More than three"):
            htem.slots([1, 2, 3, 4])

    def test_physical_cleansing_does_not_guess_unit_repairs(self):
        records = [example_record(i + 1, i + 1) for i in range(6)]
        records[1]["thickness"] = 0  # zero thickness is not repaired
        records[2]["fpm_conductivity"] = -1
        records[3]["deposition_power"] = [None, None, None]  # active targets unknown
        records[4]["deposition_sample_time_min"] = 0
        records[5]["deposition_growth_pressure_mtorr"] = -20  # optional value -> unknown
        clean, audit = htem.cleanse(pd.DataFrame(records))
        self.assertEqual(clean.id.tolist(), [1, 2, 3, 6])
        self.assertEqual(int(audit.excluded.sum()), 2)
        self.assertTrue(np.isnan(clean.loc[clean.id.eq(2), 'thickness_um'].iloc[0]))
        self.assertTrue(np.isnan(clean.loc[clean.id.eq(3), 'conductivity_S_cm'].iloc[0]))
        self.assertAlmostEqual(clean.iloc[0].thickness_um, 0.21)
        self.assertTrue(np.isnan(clean.iloc[-1].growth_pressure_mTorr))
        self.assertTrue(audit.iloc[5].negative_optional_setting_set_missing)

    def test_conflicting_duplicate_ids_fail_instead_of_averaging(self):
        first, second = example_record(), example_record()
        second["thickness"] = 1.0
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            htem.cleanse(pd.DataFrame([first, second]))

    def test_identical_recipes_and_whole_libraries_share_a_split(self):
        records = [example_record(library, library * 10 + j) for library in range(1, 31) for j in range(2)]
        # A separate run with exactly the same recipe as library 1 must remain with it.
        repeated = example_record(1, 999)
        repeated["sample_library_id"] = 31
        records.append(repeated)
        clean, _ = htem.cleanse(pd.DataFrame(records))
        split, groups = htem.split_groups(clean)
        clean["split"], clean["group"] = split, groups
        self.assertTrue((clean.groupby("sample_library_id").split.nunique() == 1).all())
        self.assertEqual(clean[clean.sample_library_id.eq(1)].group.iloc[0],
                         clean[clean.sample_library_id.eq(31)].group.iloc[0])
        np.testing.assert_array_equal(split, htem.split_groups(clean)[0])

    def test_preprocessing_fits_training_only_and_checks_artifacts(self):
        raw = pd.DataFrame([example_record(i, i) for i in range(1, 31)])
        with tempfile.TemporaryDirectory() as temp:
            clean, _, manifest = htem.prepare(raw, temp, "synthetic-fixture")
            data, transforms, loaded_manifest = htem.load_prepared(temp)
            self.assertEqual(manifest, loaded_manifest)
            self.assertEqual(manifest["version"], 5)
            self.assertIn("material_stratification", manifest)
            self.assertIn("material_split_balance.csv", manifest["files_sha256"])
            training = clean[clean.split.eq("train")]
            np.testing.assert_allclose(transforms["target_scaler"].mean_, training[htem.TARGETS].mean())
            self.assertTrue(np.allclose(data["train"]["y"].mean(axis=0), 0, atol=1e-6))
            numeric = transforms["preprocessor"].named_transformers_["numeric"]
            expected = training[htem.NUMERIC_FEATURES].median().fillna(0)
            np.testing.assert_allclose(numeric["simpleimputer"].statistics_, expected)
            for part in data.values():
                self.assertTrue(np.isfinite(part["X"]).all())
                self.assertEqual(part["y"].shape[1], 2)
                self.assertTrue(np.isfinite(part["x_mm"]).all())
            path = Path(temp) / "split_manifest.csv"
            path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                htem.load_prepared(temp)

    def test_inverse_transforms_restore_units(self):
        from sklearn.preprocessing import StandardScaler
        target = np.array([[np.log10(0.5), 3.0], [0.0, 2.0]])
        scaler = StandardScaler().fit(target)
        restored = htem.inverse_predictions(scaler.transform(target), scaler)
        np.testing.assert_allclose(restored.thickness_um, [0.5, 1.0])
        np.testing.assert_allclose(restored.conductivity_S_cm, [1000, 100])

    def test_inference_rejects_measured_fields_and_invalid_core_settings(self):
        recipe = {key: example_record()[key] for key in htem.RECIPE_FIELDS}
        self.assertEqual(htem.inference_recipe(recipe).shape, (1, len(htem.RECIPE_FEATURES)))
        with self.assertRaisesRegex(ValueError, "Only pre-deposition"):
            htem.inference_recipe({**recipe, "thickness": 0.5})
        with self.assertRaisesRegex(ValueError, "duration"):
            htem.inference_recipe({**recipe, "deposition_sample_time_min": 0})
        with self.assertRaisesRegex(ValueError, "power"):
            htem.inference_recipe({**recipe, "deposition_power": [-1, 20, None]})

    def test_distributed_snapshot_is_raw_compact_and_has_provenance(self):
        path = MODULE / "data/htem_spatial_snapshot.parquet"
        raw = pd.read_parquet(path)
        provenance = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(set(raw), set(htem.RAW_FIELDS))
        self.assertEqual(provenance["sha256"], htem.sha256(path))
        self.assertEqual(len(raw), provenance["rows"])
        self.assertFalse(any("email" in c or "spectrum" in c for c in raw))


class HTEMMaterialSplitTests(unittest.TestCase):
    def cohort(self):
        records, library = [], 0
        for material, count in [("A", 20), ("B", 10), ("C", 3), ("RareOne", 1), ("RareTwo", 2)]:
            for _ in range(count):
                library += 1
                for point in range(2):
                    row = example_record(library, library * 10 + point)
                    row.update(deposition_compounds=[material, None, None], deposition_power=[50, None, None])
                    records.append(row)
        clean, _ = htem.cleanse(pd.DataFrame(records))
        return clean

    def test_common_materials_occur_in_all_splits_with_representative_quotas(self):
        clean = self.cohort()
        clean["split"], clean["leakage_group"] = htem.split_groups(clean)
        balance = htem.material_split_report(clean).set_index("material_class")
        for material in ["A", "B", "C"]:
            self.assertTrue(balance.loc[material, "all_splits_present"])
        self.assertEqual(balance.loc["A", [name + "_groups" for name in htem.SPLITS]].tolist(), [12, 4, 4])
        self.assertEqual(balance.loc["B", [name + "_groups" for name in htem.SPLITS]].tolist(), [6, 2, 2])
        self.assertEqual(balance.loc["C", [name + "_groups" for name in htem.SPLITS]].tolist(), [1, 1, 1])
        self.assertTrue(clean.groupby("sample_library_id").split.nunique().eq(1).all())
        self.assertTrue(clean.groupby("leakage_group").split.nunique().eq(1).all())
        # Balancing does not add, remove, or duplicate points.
        self.assertEqual(sum(balance[name + "_points"].sum() for name in htem.SPLITS), len(clean))

    def test_rare_classes_stay_in_train_and_limitations_are_explicit(self):
        clean = self.cohort()
        clean["split"], clean["leakage_group"] = htem.split_groups(clean)
        balance = htem.material_split_report(clean).set_index("material_class")
        self.assertEqual(balance.loc["RareOne", "train_groups"], 1)
        self.assertFalse(balance.loc["RareOne", "all_splits_feasible"])
        self.assertFalse(balance.loc["RareOne", "all_splits_present"])
        self.assertEqual(balance.loc["RareTwo", "train_groups"], 1)
        self.assertEqual(balance.loc["RareTwo", "validation_groups"] + balance.loc["RareTwo", "test_groups"], 1)
        self.assertIn("two independent", balance.loc["RareTwo", "coverage_note"])

    def test_strata_ignore_source_order_and_inactive_material_but_features_do_not(self):
        rows = [example_record(1, 1), example_record(2, 2)]
        rows[0].update(deposition_compounds=["ZnO", "Al", "Unused"], deposition_power=[100, 20, 0])
        rows[1].update(deposition_compounds=["Al", "ZnO", None], deposition_power=[20, 100, None])
        clean, _ = htem.cleanse(pd.DataFrame(rows))
        self.assertEqual(htem.material_classes(clean).tolist(), ["Al + ZnO", "Al + ZnO"])
        self.assertEqual(clean.target_1.tolist(), ["ZnO", "Al"])
        self.assertEqual(clean.target_1_power_W.tolist(), [100, 20])

    def test_assignments_are_reproducible_row_order_independent_and_outcome_blind(self):
        clean = self.cohort()
        expected, groups = htem.split_groups(clean, seed=12)
        np.testing.assert_array_equal(expected, htem.split_groups(clean, seed=12)[0])
        shuffled = clean.sample(frac=1, random_state=3).copy()
        shuffled[htem.TARGETS] = 9999
        shuffled["thickness_um"], shuffled["conductivity_S_cm"] = 123, 456
        actual, _ = htem.split_groups(shuffled, seed=12)
        self.assertEqual(dict(zip(clean.id, expected)), dict(zip(shuffled.id, actual)))

    def test_inconsistent_materials_within_library_fail_without_breaking_group(self):
        clean = self.cohort()
        clean.loc[clean.index[0], "target_1"] = "Unexpected"
        with self.assertRaisesRegex(ValueError, "multiple material classes"):
            htem.split_groups(clean)

    def test_singleton_only_cohort_does_not_invent_safe_holdouts(self):
        rows = [example_record(i, i) for i in range(1, 11)]
        for i, row in enumerate(rows):
            row.update(deposition_compounds=["Unique" + str(i), None, None], deposition_power=[50, None, None])
        clean, _ = htem.cleanse(pd.DataFrame(rows))
        with self.assertRaisesRegex(ValueError, "empty holdout"):
            htem.split_groups(clean)

    def test_repeated_recipe_is_one_group_even_with_multiple_depositions(self):
        clean = self.cohort()
        repeat = clean[htem.material_classes(clean).eq("RareOne")].copy()
        repeat["sample_library_id"] = 999
        repeat["id"] += 100000
        clean = pd.concat([clean, repeat], ignore_index=True)
        clean["split"], clean["leakage_group"] = htem.split_groups(clean)
        row = htem.material_split_report(clean).set_index("material_class").loc["RareOne"]
        self.assertEqual(row.total_groups, 1)
        self.assertEqual(row.train_libraries, 2)
        self.assertEqual(row.validation_groups + row.test_groups, 0)

    def test_loader_rejects_old_unstratified_manifest(self):
        raw = pd.DataFrame([example_record(i, i) for i in range(1, 21)])
        with tempfile.TemporaryDirectory() as temp:
            htem.prepare(raw, temp, "fixture")
            path = Path(temp) / "manifest.json"
            old = json.loads(path.read_text())
            old["version"] = 2
            path.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, "predates material stratification"):
                htem.load_prepared(temp)


class HTEMDownloadTests(unittest.TestCase):
    def test_downloader_merges_recipes_and_labels_and_resumes_from_cache(self):
        record = example_record()
        library = {key: record[key] for key in htem.RECIPE_FIELDS}
        library.update(id=1, sample_ids=[1])
        block = {"sample_library_id": 1, "position": [1], "x_mm": [4.0], "y_mm": [6.65], "absolute_temp_c": [350.0],
                 **{key: [record[key]] for key in htem.LABEL_FIELDS},
                 "xrf_concentration": [[0.5, 0.5]], "xrd_intensity": [[1, 2, 3]]}
        session = Mock()
        responses = {"sample_library": [{"id": 1}], "sample_library/1": library,
                     "sample_library/prop?ids=1": [block]}

        def get(url, **kwargs):
            response = Mock()
            response.json.return_value = responses[url.split("/api/")[1]]
            return response

        session.get.side_effect = get
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            frame = htem.download_htem(temp / "raw.parquet", temp / "cache", session=session, delay=0)
            self.assertEqual(session.get.call_count, 3)
            self.assertEqual(frame.absolute_temp_c.iloc[0], 350.0)
            self.assertEqual(frame.pdac.iloc[0], "4")
            self.assertEqual(frame.thickness.iloc[0], record["thickness"])
            self.assertEqual(frame.id.iloc[0], "1:1")
            self.assertEqual(frame.x_mm.iloc[0], 4.0)
            for path in (temp / "cache").glob("*.json"):
                self.assertNotIn("xrd_intensity", path.read_text())
                self.assertNotIn("xrf_concentration", path.read_text())
            session.get.reset_mock()
            htem.download_htem(temp / "raw.parquet", temp / "cache", session=session, delay=0)
            self.assertEqual(session.get.call_count, 1)  # public discovery refreshed; cached data reused

    def test_scalar_arrays_align_by_position_not_sample_id_order(self):
        from htem_download import compact_library, expand_scalar_block
        library = compact_library({**example_record(), "id": 17, "sample_ids": [999, 15]})
        block = {"sample_library_id": 17, "position": [2, 1], "x_mm": [8, 4], "y_mm": [1, 1],
                 "thickness": [0.8, 0.4], "fpm_conductivity": [80, 40], "opt_direct_bandgap": None}
        records = expand_scalar_block(library, block)
        self.assertEqual([(r['position'], r['x_mm'], r['thickness']) for r in records], [(2, 8, 0.8), (1, 4, 0.4)])
        block['thickness'] = [0.8]
        with self.assertRaisesRegex(ValueError, 'Misaligned'):
            expand_scalar_block(library, block)

    def test_empty_sample_id_placeholder_is_not_a_fake_observation(self):
        from htem_download import compact_library
        metadata = compact_library({**example_record(), 'sample_ids': [None]})
        self.assertEqual(metadata['expected_sample_count'], 0)
        self.assertEqual(metadata['null_sample_id_placeholders'], 1)

    def test_exceptional_sample_fallback_strips_spectra_and_never_guesses_xy(self):
        record = example_record()
        library = {**record, 'id': 1, 'sample_ids': [7], 'owner_email': 'not-retained@example.org'}
        sample = {**record, 'id': 7, 'position': 1, 'sample_library_id': 1,
                  'xyz_mm': [47, 49, 3.5], 'xrd_intensity': [9, 8, 7], 'opt_wavelength': [400, 500]}
        responses = {'sample_library': [{'id': 1}], 'sample_library/1': library,
                     'sample_library/prop?ids=1': [], 'sample/7': sample}
        session = Mock()
        def get(url, **kwargs):
            response = Mock()
            response.json.return_value = responses[url.split('/api/')[1]]
            return response
        session.get.side_effect = get
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            frame = htem.download_htem(temp / 'raw.parquet', temp / 'cache', delay=0, session=session)
            self.assertEqual(len(frame), 1)
            self.assertTrue(frame[['x_mm', 'y_mm']].isna().all(axis=None))
            for path in (temp / 'cache').glob('*.json'):
                payload = path.read_text()
                for forbidden in ['xyz_mm', 'xrd_intensity', 'opt_wavelength', 'owner_email']:
                    self.assertNotIn(forbidden, payload)
            report = json.loads((temp / 'raw.provenance.json').read_text())
            self.assertEqual(report['rows_missing_substrate_coordinates'], 1)
            self.assertEqual(report['fallback_libraries_without_substrate_coordinates'], [1])

    def test_unavailable_library_keeps_its_recipe_and_reports_incomplete_coverage(self):
        records = {1: example_record(), 2: example_record(2, 2)}
        session = Mock()
        def get(url, **kwargs):
            endpoint = url.split('/api/')[1]
            response = Mock()
            if endpoint == 'sample_library':
                value = [{'id': 1}, {'id': 2}]
            elif endpoint in ['sample_library/1', 'sample_library/2']:
                lib = int(endpoint[-1])
                value = {**records[lib], 'id': lib, 'sample_ids': [lib]}
            elif endpoint == 'sample/2':
                raise RuntimeError('source unavailable')
            else:
                value = [{'sample_library_id': 1, 'position': [1], 'x_mm': [4], 'y_mm': [2],
                          **{key: [records[1][key]] for key in htem.LABEL_FIELDS}}] if endpoint.endswith('=1,2') else []
            response.json.return_value = value
            return response
        session.get.side_effect = get
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            frame = htem.download_htem(temp / 'raw.parquet', temp / 'cache', delay=0, session=session)
            self.assertEqual(len(frame), 1)
            self.assertEqual(len(pd.read_parquet(temp / 'htem_libraries.parquet')), 2)
            report = json.loads((temp / 'raw.provenance.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual(report['sample_count_gaps'][0]['library_id'], 2)

    def test_api_failure_does_not_create_partial_final_output(self):
        session = Mock()
        session.get.side_effect = RuntimeError("offline")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "raw.parquet"
            with self.assertRaisesRegex(RuntimeError, "offline"):
                htem.download_htem(output, Path(temp) / "cache", [1], session=session, delay=0)
            self.assertFalse(output.exists())


class HTEMNeuralTests(unittest.TestCase):
    def test_best_validation_epoch_is_restored(self):
        import torch
        rng = np.random.default_rng(0)
        x = rng.normal(size=(30, 5)).astype("float32")
        y = rng.normal(size=(30, 3)).astype("float32")
        model, history, note = htem.fit_neural_head(x, y, x, y, hidden=(8,), epochs=4, patience=2)
        mse = np.mean((htem.neural_predict(model, x) - y) ** 2)
        self.assertAlmostEqual(mse, note["best_validation_MSE"], places=6)
        self.assertEqual(note["best_epoch"], int(history.loc[history.validation_loss.idxmin(), "epoch"]))
        self.assertLess(history.learning_rate.iloc[-1], history.learning_rate.iloc[0])
        self.assertFalse(model.training)


class HTEMSpatialTests(unittest.TestCase):
    def test_coordinates_change_features_without_using_outputs_or_ids(self):
        row = example_record()
        second = {**row, 'x_mm': row['x_mm'] + 4, 'position': 123, 'thickness': 8, 'id': 'another'}
        first_features, second_features = htem.engineer_observation(row), htem.engineer_observation(second)
        self.assertNotEqual(first_features['x_mm'], second_features['x_mm'])
        self.assertNotIn('position', second_features)
        self.assertNotIn('id', second_features)
        self.assertNotIn('thickness', second_features)

    def test_missing_xy_is_not_filled_from_instrument_xyz(self):
        row = {**example_record(), 'x_mm': None, 'xyz_mm': [4, 2, 9]}
        with self.assertRaisesRegex(ValueError, 'No eligible'):
            htem.cleanse(pd.DataFrame([row]))

    def test_same_recipe_on_different_grids_is_grouped_together(self):
        first, second = example_record(1, 1), example_record(1, 2)
        second.update(sample_library_id=2, x_mm=40.0)
        clean, _ = htem.cleanse(pd.DataFrame([first, second]))
        groups = htem.leakage_groups(clean)
        self.assertEqual(groups[0], groups[1])

    def test_map_inference_rejects_mixed_depositions_or_measured_grid_fields(self):
        recipe = {key: example_record()[key] for key in htem.RECIPE_FIELDS}
        grid = pd.DataFrame({'position': [1, 2], 'x_mm': [4, 8], 'y_mm': [2, 2]})
        metadata, encoded = htem.inference_points(recipe, grid)
        self.assertEqual(metadata.position.tolist(), [1, 2])
        self.assertEqual(encoded.x_mm.tolist(), [4, 8])
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            htem.inference_points(recipe, grid.assign(sample_library_id=[1, 2]))
        with self.assertRaisesRegex(ValueError, 'not measured'):
            htem.inference_points(recipe, grid.assign(thickness=[1, 2]))
        with self.assertRaisesRegex(ValueError, 'unique'):
            htem.inference_points(recipe, grid.assign(position=[1, 1]))
        with self.assertRaisesRegex(ValueError, 'same XY'):
            htem.inference_points(recipe, grid.assign(x_mm=[4, 4]))

    def test_plot_never_pools_multiple_libraries_or_duplicate_points(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        table = pd.DataFrame({'library_id': [1, 2], 'position': [1, 1], 'x_mm': [4, 4], 'y_mm': [2, 2],
                              'predicted_thickness_um': [0.5, 1.0], 'predicted_conductivity_S_cm': [5, 10]})
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            htem.plot_substrate_maps(table)
        duplicate = table.assign(library_id=1)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            htem.plot_substrate_maps(duplicate)
        # Explicitly selected single-point grids remain legitimate, no 44-point reshape.
        fig = htem.plot_substrate_maps(table, library_id=1)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
