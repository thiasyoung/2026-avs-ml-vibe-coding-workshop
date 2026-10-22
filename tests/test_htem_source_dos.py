"""Reference DOS is external composition metadata, not a fabricated source phase."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "03_htem_conductivity"
sys.path.insert(0, str(MODULE))
import htem_source_dos as dos
import htem_chemistry as chemistry
import htem_workshop as htem


class SourceDOSTests(unittest.TestCase):
    def test_energy_grid_uses_published_spacing_without_stretching_or_padding(self):
        self.assertEqual(len(dos.ENERGY_EV), 300)
        self.assertEqual(dos.ENERGY_EV[0], -5)
        self.assertAlmostEqual(dos.ENERGY_EV[-1], 9.95)
        np.testing.assert_allclose(np.diff(dos.ENERGY_EV), .05, rtol=1e-12)
        self.assertEqual(dos.BIN_EDGES_EV[-1], dos.ENERGY_EV[-1])

    def test_integrals_have_known_values_and_ignore_arbitrary_dos_amplitude(self):
        uniform = dos.spectral_descriptors(np.ones(300))
        width = dos.ENERGY_EV[-1] - dos.ENERGY_EV[0]
        self.assertAlmostEqual(uniform['centroid_eV'], 2.475)
        self.assertAlmostEqual(uniform['occupied_fraction'], 5 / width)
        self.assertAlmostEqual(uniform['near_fermi_fraction'], 1 / width)
        self.assertAlmostEqual(uniform['width_eV'], width / np.sqrt(12), delta=1e-3)
        self.assertAlmostEqual(sum(uniform[f'bin_{i}_fraction'] for i in range(7)), 1.)
        self.assertEqual(uniform, dos.spectral_descriptors(np.ones(300) * 300))

    def test_invalid_spectra_are_missing_not_invented(self):
        for values in [None, 'na', [], np.ones(299), np.zeros(300),
                       np.full(300, -1), np.full(300, np.nan), np.full(300, np.inf)]:
            with self.subTest(values=str(values)[:30]), self.assertRaises((ValueError, TypeError)):
                dos.spectral_descriptors(values)
        almost = np.ones(300)
        almost[0] = -1e-10
        self.assertTrue(np.isfinite(list(dos.spectral_descriptors(almost).values())).all())

    def test_streaming_reader_handles_tiny_chunks_unicode_numbers_and_nested_records(self):
        payload = [{'jid': 'é', 'a': [12, {'n': 123.456e-3}]}, None, 12345, 1.2345e-7, 'ZnO']
        for chunk in [1, 2, 7, 1024]:
            result = list(dos.iter_json_array(io.StringIO(' \n' + json.dumps(payload, ensure_ascii=False)), chunk))
            self.assertEqual(result, payload)
        self.assertEqual(list(dos.iter_json_array(io.StringIO('[]'), 1)), [])
        for payload in ['{}', '[{}', '[{},]', '[{}] garbage', '[{} {}]']:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                list(dos.iter_json_array(io.StringIO(payload), 2))

    def test_all_matching_reference_structures_are_kept_without_phase_selection(self):
        key = chemistry.composition_key('ZnO')
        a = {'jid': 'JVASP-fixture-1', 'atoms': {'elements': ['Zn', 'O'], 'coords': [[0, 0, 0]]},
             'edos_up': np.ones(300).tolist()}
        b = {'jid': 'JVASP-fixture-2', 'atoms': {'elements': ['O', 'Zn', 'O', 'Zn']},
             'edos_up': np.linspace(.01, 1., 300).tolist()}
        other = {'jid': 'not-a-match', 'atoms': {'elements': ['Cu']}, 'edos_up': 'na'}
        missing = {'jid': 'JVASP-fixture-3', 'atoms': {'elements': ['Zn', 'O']}, 'edos_up': 'na'}
        result, coverage = dos.curate_records(iter([a, b, other, missing]), {key})
        self.assertEqual(set(result), {key})
        self.assertEqual(result[key]['reference_ids'], ['JVASP-fixture-1', 'JVASP-fixture-2'])
        features = result[key]['features']
        self.assertEqual(features['reference_dos_records'], 2)
        self.assertGreater(features['reference_dos_mean_bin_std'], 0.)
        self.assertAlmostEqual(features['reference_dos_centroid_eV'], np.mean([
            dos.spectral_descriptors(row['edos_up'])['centroid_eV'] for row in [a, b]]))
        self.assertEqual(coverage['rejected'], {'invalid_matched_dos': 1})
        changed = copy.deepcopy(a)
        changed['atoms']['coords'] = [[500, 800, -200]]
        self.assertEqual(dos.curate_records([a], {key})[0], dos.curate_records([changed], {key})[0])
        with self.assertRaisesRegex(ValueError, 'unique'):
            dos.curate_records([a, a], {key})

    def test_missing_and_absent_formulas_do_not_get_fake_zero_dos(self):
        for formula in ['Unspecified alloy', '__NONE__']:
            values = chemistry.source_descriptors(formula)
            self.assertEqual(values['reference_dos_known'], 0)
            self.assertEqual(values['reference_dos_records'], 0)
            self.assertTrue(np.isnan(values['reference_dos_centroid_eV']))
        self.assertEqual(chemistry.source_descriptors('ZnO'), chemistry.source_descriptors('Zn2O2'))

    def test_power_coverage_and_source_slot_pairing_are_preserved(self):
        values = chemistry.recipe_descriptors(['ZnO', 'Unspecified alloy', '__NONE__'], [20, 60, 0])
        self.assertEqual(values['source_dos_power_coverage'], .25)
        self.assertTrue(np.isnan(values['power_weighted_reference_dos_centroid_eV']))
        single = chemistry.recipe_descriptors(['ZnO', 'Unspecified alloy', '__NONE__'], [20, 0, 0])
        self.assertEqual(single['source_dos_power_coverage'], 1.)
        self.assertEqual(single['power_weighted_reference_dos_centroid_eV'], single['target_1_reference_dos_centroid_eV'])
        swapped = chemistry.recipe_descriptors(['Cu2O', 'ZnO', '__NONE__'], [60, 20, 0])
        original = chemistry.recipe_descriptors(['ZnO', 'Cu2O', '__NONE__'], [20, 60, 0])
        self.assertAlmostEqual(swapped['power_weighted_reference_dos_centroid_eV'], original['power_weighted_reference_dos_centroid_eV'])
        self.assertEqual(original['target_1_reference_dos_centroid_eV'], swapped['target_2_reference_dos_centroid_eV'])

    def test_reference_provenance_counts_and_fraction_constraints(self):
        reference = chemistry.dos_reference_data()
        self.assertEqual(reference['provenance']['license'], 'CC BY 4.0')
        self.assertEqual(reference['provenance']['archive_md5'], dos.ARCHIVE_MD5)
        self.assertEqual(reference['provenance']['external_records'], 55659)
        self.assertEqual(len(reference['provenance']['archive_sha256']), 64)
        for record in reference['compositions'].values():
            f = record['features']
            self.assertEqual(set(f), set(dos.DOS_PROPERTIES))
            self.assertTrue(np.isfinite(list(f.values())).all())
            self.assertEqual(f['reference_dos_records'], len(record['reference_ids']))
            self.assertAlmostEqual(sum(f[f'reference_dos_bin_{i}_fraction'] for i in range(7)), 1.)
            self.assertTrue(0 <= f['reference_dos_near_fermi_fraction'] <= 1)

    def test_preparation_obeys_selected_dos_policy_and_rejects_changed_reference(self):
        from test_htem_process_emulator import example_record
        raw = pd.DataFrame([example_record(i, i) for i in range(1, 21)])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'prepared'
            _, _, manifest = htem.prepare(raw, output, 'fixture')
            feature = 'numeric__target_1_reference_dos_centroid_eV'
            self.assertEqual(feature in manifest['encoded_features'],
                             'dos' in manifest['frozen_feature_selection']['blocks'])
            data, _, _ = htem.load_prepared(output)
            self.assertTrue(np.isfinite(data['train']['X']).all())
            changed = Path(temp) / 'changed.json'
            changed.write_bytes(dos.REFERENCE.read_bytes() + b' ')
            with patch.object(dos, 'REFERENCE', changed):
                with self.assertRaisesRegex(ValueError, 'Source-chemistry'):
                    htem.load_prepared(output)


if __name__ == '__main__':
    unittest.main()
