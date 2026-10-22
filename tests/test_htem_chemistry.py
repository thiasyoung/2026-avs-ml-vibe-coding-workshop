"""Source chemistry is pre-process metadata; no target-film labels or invented gaps."""
import copy
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "03_htem_conductivity"
sys.path.insert(0, str(MODULE))
import htem_chemistry as chemistry
import htem_workshop as htem


class SourceChemistryTests(unittest.TestCase):
    def test_formula_amounts_parentheses_and_equivalent_compositions(self):
        self.assertEqual(dict(chemistry.parse_formula("Fe2O3")), {"Fe": Fraction(2, 5), "O": Fraction(3, 5)})
        self.assertEqual(dict(chemistry.parse_formula("Ca3(PO4)2"))["O"], Fraction(8, 13))
        self.assertEqual(chemistry.composition_key("ZnO"), chemistry.composition_key("O2Zn2"))
        self.assertEqual(chemistry.composition_key("Zn0.5O0.5"), chemistry.composition_key("ZnO"))

    def test_parser_rejects_unsupported_or_ambiguous_names(self):
        for formula in ["", "zno", "Eagle XG", "Zn1-xCoxO", "CuSO4·5H2O", "Fe3+", "Zn(O", "Zn0O", "()"]:
            with self.subTest(formula=formula), self.assertRaises(ValueError):
                chemistry.parse_formula(formula)

    def test_atom_weighted_properties_have_correct_units_and_stoichiometry(self):
        source = chemistry.source_descriptors("ZnO")
        self.assertAlmostEqual(source["mean_electronegativity"], (1.65 + 3.44) / 2)
        self.assertAlmostEqual(source["range_electronegativity"], 3.44 - 1.65)
        self.assertAlmostEqual(source["std_electronegativity"], (3.44 - 1.65) / 2)
        self.assertAlmostEqual(source["mean_atomic_number"], (30 + 8) / 2)
        self.assertEqual(source["fraction_O"], .5)
        self.assertAlmostEqual(chemistry.source_descriptors("Fe2O3")["fraction_O"], .6)
        self.assertGreater(source["mean_first_ionization_eV"], 0)

    def test_bandgap_is_external_reference_median_not_elemental_average(self):
        table = chemistry.reference_data()["bandgaps"][chemistry.composition_key("ZnO")]
        source = chemistry.source_descriptors("Zn2O2")
        self.assertEqual(source["reference_bandgap_eV"], table["median_eV"])
        self.assertEqual(source["reference_bandgap_range_eV"], table["max_eV"] - table["min_eV"])
        self.assertGreater(table["records"], 1)
        self.assertEqual(chemistry.reference_data()["provenance"]["bandgaps"]["external_rows"], 6354)

    def test_missing_and_absent_sources_do_not_get_fabricated_zero_gaps(self):
        unknown = chemistry.source_descriptors("Unspecified alloy")
        absent = chemistry.source_descriptors("__NONE__")
        self.assertEqual(unknown["formula_known"], 0)
        self.assertEqual(unknown["present"], 1)
        self.assertEqual(absent["present"], 0)
        self.assertTrue(np.isnan(unknown["reference_bandgap_eV"]))
        self.assertTrue(np.isnan(absent["reference_bandgap_eV"]))
        self.assertEqual(absent["fraction_O"], 0)

    def test_power_pairing_and_partial_gap_coverage_are_explicit(self):
        features = chemistry.recipe_descriptors(["ZnO", "Unspecified alloy", "__NONE__"], [20, 60, 0])
        self.assertEqual(features["source_formula_power_coverage"], .25)
        self.assertEqual(features["source_bandgap_power_coverage"], .25)
        self.assertTrue(np.isnan(features["power_weighted_reference_bandgap_eV"]))
        self.assertTrue(np.isnan(features["power_weighted_electronegativity"]))
        self.assertEqual(features["target_1_power_fraction"], .25)
        same = chemistry.recipe_descriptors(["ZnO", "Unspecified alloy", "__NONE__"], [20, 0, 0])
        self.assertEqual(same["power_weighted_reference_bandgap_eV"], same["target_1_reference_bandgap_eV"])

    def test_recipe_identity_ignores_derived_properties_but_not_source_formula(self):
        from test_htem_process_emulator import example_record
        recipe = example_record()
        frame = pd.DataFrame([htem.engineer_observation(recipe)] * 3)
        frame["sample_library_id"] = [1, 2, 3]
        frame.loc[1, "target_1_mean_electronegativity"] = -999
        frame.loc[2, "target_1"] = "different source"
        groups = htem.leakage_groups(frame)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])

    def test_preparation_uses_no_source_onehot_and_checks_reference_fingerprint(self):
        # Import the existing process-only fixture; no API/network calls.
        from test_htem_process_emulator import example_record
        raw = pd.DataFrame([example_record(i, i) for i in range(1, 21)])
        original = copy.deepcopy(raw)
        with tempfile.TemporaryDirectory() as output:
            clean, _, manifest = htem.prepare(raw, output, "fixture")
            data, transforms, _ = htem.load_prepared(output)
            self.assertTrue(np.isfinite(data["train"]["X"]).all())
            self.assertFalse(any(f.startswith("categorical__target_") for f in manifest["encoded_features"]))
            self.assertIn("numeric__target_1_mean_electronegativity", manifest["encoded_features"])
            dropped = transforms["preprocessor"].transformers_[-1][2]
            self.assertEqual({htem.FEATURES[index] for index in dropped}, set(htem.SOURCE_FIELDS))
            self.assertIn("source_chemistry_coverage.csv", manifest["files_sha256"])
            with patch.object(chemistry, "reference_fingerprint", return_value={"reference_sha256": "changed"}):
                with self.assertRaisesRegex(ValueError, "Source-chemistry"):
                    htem.load_prepared(output)
        pd.testing.assert_frame_equal(raw, original)

    def test_all_curated_references_have_provenance_and_nonnegative_values(self):
        reference = chemistry.reference_data()
        self.assertEqual(len(reference["elements"]), 118)
        self.assertEqual(reference["provenance"]["catalog"]["unsupported_labels"], [])
        for values in reference["bandgaps"].values():
            self.assertTrue(0 <= values["min_eV"] <= values["median_eV"] <= values["max_eV"])
            self.assertGreater(values["records"], 0)


if __name__ == "__main__":
    unittest.main()
