"""Frozen reference features, honest embeddings, and validation-only selection."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "03_htem_conductivity"
sys.path.insert(0, str(MODULE / 'lib'))
import htem_chemistry as chemistry
import htem_features as frozen
import htem_feature_study as study
import htem_workshop as htem


class FrozenFeatureTests(unittest.TestCase):
    def test_selection_is_complete_and_validation_only(self):
        spec = frozen.specification()
        self.assertFalse(spec["test_used_for_selection"])
        trials = pd.read_csv(MODULE / "data/htem_feature_validation_trials.csv")
        ranking = study.choose_winner(trials)
        self.assertEqual(set(trials.candidate), set(study.CANDIDATES))
        self.assertEqual(ranking.iloc[0].candidate, spec["selected_candidate"])
        self.assertEqual(set(trials.estimator), {"ridge", "forest"})
        self.assertFalse(any("test" in col for col in trials))
        self.assertEqual(study.sha256(MODULE / "data/htem_feature_validation_trials.csv"), spec["trials_sha256"])

    def test_schema_keeps_physical_features_and_rejects_labels_ids(self):
        selected = frozen.specification()["numeric_features"]
        self.assertEqual(htem.NUMERIC_FEATURES, selected)
        self.assertTrue({"x_mm", "y_mm", "local_temperature_C"}.issubset(selected))
        self.assertTrue(set(htem.TARGETS).isdisjoint(selected))
        self.assertTrue({"id", "sample_library_id", "position", "bandgap_eV", "thickness_um", "conductivity_S_cm"}.isdisjoint(selected))
        self.assertEqual(len(selected), len(set(selected)))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            frozen.selected_numeric([])

    def test_published_embeddings_and_real_crystal_coverage(self):
        reference = json.loads((MODULE / "data/htem_feature_candidates.json").read_text(encoding="utf-8"))
        provenance = reference["provenance"]
        self.assertIn("stoichiometric mean", provenance["composition_embedding"]["method"])
        crystal = provenance["crystal_embedding"]
        self.assertGreater(crystal["reference_structures"], 0)
        self.assertEqual(crystal["dimension"], 256)
        self.assertEqual(crystal["archive_sha256"], "d80f5e067c0b0b0f569b8158ecfeeeb68472bebe2ceb27ff787dcfcc68948e20")
        self.assertTrue(crystal["graph"]["line_graph"])
        self.assertEqual(len(crystal["failures"]), 0)
        for values in reference["compositions"].values():
            self.assertEqual(len(values["composition_embedding"]), 16)
            self.assertAlmostEqual(sum(values["curated"][f"valence_fraction_{o}"] for o in "spdf"), 1.)
            if values["crystal_embedding"]:
                self.assertEqual(len(values["crystal_embedding"]["mean"]), 256)

    def test_crystal_pca_ignores_holdout_reference_vectors(self):
        rng = np.random.default_rng(19)
        formulas = ["ZnO", "CuO", "Fe2O3", "TiO2"]
        reference = {"compositions": {chemistry.composition_key(f): {
            "curated": {"a": float(i)}, "composition_embedding": [float(i)] * 16,
            "crystal_embedding": {"mean": rng.normal(size=256).tolist(), "mean_coordinate_std": .1}}
            for i, f in enumerate(formulas)}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reference.json"
            path.write_text(json.dumps(reference), encoding="utf-8")
            with patch.object(study, "REFERENCE", path):
                first = study.build_source_table(reference, set(formulas[:3]), components=2)
                changed = copy.deepcopy(reference)
                changed["compositions"][chemistry.composition_key(formulas[3])]["crystal_embedding"]["mean"] = [1e8] * 256
                second = study.build_source_table(changed, set(formulas[:3]), components=2)
        self.assertEqual(first["projection"], second["projection"])
        self.assertEqual(len(first["projection"]["training_composition_keys"]), 3)
        for formula in formulas[:3]:
            self.assertEqual(first["compositions"][chemistry.composition_key(formula)], second["compositions"][chemistry.composition_key(formula)])

    def test_sources_keep_slots_and_unknown_is_not_absent(self):
        features = frozen.recipe_features(["ZnO", "CuO", "__NONE__"])
        swapped = frozen.recipe_features(["CuO", "ZnO", "__NONE__"])
        for field in frozen.source_table()["blocks"]["composition"]:
            self.assertEqual(features["target_1_" + field], swapped["target_2_" + field])
            self.assertEqual(features["target_3_" + field], 0.)
        unknown = frozen.recipe_features(["Unsupported alloy", "__NONE__", "__NONE__"])
        self.assertEqual(unknown["target_1_composition_embedding_known"], 0.)
        self.assertTrue(np.isnan(unknown["target_1_megnet_mean_00"]))
        self.assertEqual(unknown["target_2_megnet_mean_00"], 0.)

    def test_fingerprints_detect_specification_and_table_changes(self):
        before = frozen.fingerprint()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "spec.json"
            path.write_bytes(frozen.SPECIFICATION.read_bytes() + b" ")
            with patch.object(frozen, "SPECIFICATION", path):
                self.assertNotEqual(before, frozen.fingerprint())
        self.assertEqual(before, frozen.fingerprint())

    def test_encoder_fits_training_only(self):
        training = pd.DataFrame({"setting": [1., 3., np.nan], "gas": ["Ar", "Ar", "Ar"]})
        validation = pd.DataFrame({"setting": [10000.], "gas": ["new"]})
        encoder = study.make_encoder(["setting"], ["gas"]).fit(training)
        stats = encoder.named_transformers_["numeric"][0].statistics_.copy()
        encoder.transform(validation)
        np.testing.assert_array_equal(stats, [2.])
        self.assertEqual(encoder.named_transformers_["categorical"].categories_[0].tolist(), ["Ar"])

    def test_all_selectors_required_before_choosing(self):
        incomplete = pd.DataFrame([{"candidate": "existing_dos", "estimator": "ridge",
                                   "validation_score": 1., "encoded_features": 10}])
        with self.assertRaisesRegex(ValueError, "same two selectors"):
            study.choose_winner(incomplete)

    def test_selection_loader_never_reads_test_artifacts(self):
        original = np.load
        loaded = []
        def guarded(path, *args, **kwargs):
            self.assertNotEqual(Path(path).name, "test.npz")
            loaded.append(Path(path).name)
            return original(path, *args, **kwargs)
        # CI must not require the instructor's ignored historical archive.
        with tempfile.TemporaryDirectory() as temp:
            prepared = Path(temp) / "htem_prepared"
            prepared.mkdir()
            (prepared / "manifest.json").write_text("{}", encoding="utf-8")
            pd.DataFrame({"id": ["a", "b", "hidden"],
                          "split": ["train", "validation", "test"],
                          "leakage_group": [1, 2, 3]}).to_parquet(prepared / "clean_process_table.parquet", index=False)
            np.savez_compressed(prepared / "train.npz", sample_id=np.array(["a"]))
            np.savez_compressed(prepared / "validation.npz", sample_id=np.array(["b"]))
            with patch.object(study.np, "load", side_effect=guarded):
                clean, _, _ = study.load_selection_data(temp)
        self.assertEqual(loaded, ["train.npz", "validation.npz"])
        self.assertEqual(set(clean.split), {"train", "validation"})

    def test_shipped_matrix_uses_selected_features_and_preserves_cohort(self):
        data, transforms, manifest = htem.load_prepared(MODULE / "data/frozen_features")
        self.assertEqual(manifest["rows_clean"], 36762)
        self.assertEqual(manifest["libraries"], 859)
        self.assertEqual(manifest["frozen_feature_selection"]["candidate"], frozen.specification()["selected_candidate"])
        self.assertEqual(data["train"]["X"].shape[1], len(manifest["encoded_features"]))
        self.assertTrue(all(np.isfinite(part["X"]).all() for part in data.values()))
        self.assertTrue(all(np.array_equal(part["y_mask"], np.isfinite(part["y"])) for part in data.values()))
        self.assertFalse(any("megnet" in f for f in manifest["encoded_features"]))
        self.assertIn("numeric__target_1_alignn_pc_00", manifest["encoded_features"])

    def test_workshop_contains_worked_selected_feature_code(self):
        notebook = json.loads((MODULE / "01_HTEM_Process_Data_Carpentry.ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(c.get("source", [])) for c in notebook["cells"] if c["cell_type"] == "code")
        self.assertIn("def encode_reference_crystal(", code)
        self.assertIn("def selected_source_features(", code)
        self.assertIn("crystal_pca = PCA(n_components=16", code)
        self.assertIn("encoder.fit(worked_inputs[clean.split.eq('train')])", code)
        self.assertNotIn("study.run_study(", code)

    def test_every_architecture_reuses_frozen_preparation(self):
        names = sorted((MODULE / "examples").glob("02*.ipynb"))
        self.assertEqual(len(names), 5)
        for path in names:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code = "\n".join("".join(c.get("source", [])) for c in notebook["cells"] if c["cell_type"] == "code")
            self.assertIn("exp.load_context(HERE)", code)
            self.assertNotIn("htem.prepare(", code)
            self.assertNotIn("fit_transform(", code)
            self.assertNotIn("run_study(", code)


if __name__ == "__main__":
    unittest.main()
