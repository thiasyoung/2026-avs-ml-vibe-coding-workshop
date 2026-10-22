# Source-material reference data

## Additional frozen-feature study references

`htem_feature_candidates.json` contains instructor-curated summaries and embeddings;
`htem_frozen_source_features.json` contains their training-projected offline lookup.
The selected columns are recorded in `htem_feature_selection.json`. Complete upstream
copyright/redistribution notices are retained verbatim in
[`htem_feature_reference_licenses.json`](htem_feature_reference_licenses.json).

- **Magpie-style descriptors:** selected elemental tables from
  [matminer commit bd1386e](https://github.com/hackingmaterials/matminer/tree/bd1386e2412b3367b8e11388d02092524a6dafcb),
  Regents of the University of California/LBNL, BSD-style license. Fraction-weighted
  means/ranges/stds, valence fractions and stoichiometry norms are derived locally.
  Ward et al., [doi:10.1038/npjcompumats.2016.28](https://doi.org/10.1038/npjcompumats.2016.28).
- **Composition vectors:** published 16-dimensional element vectors from
  [MEGNet commit f370576](https://github.com/materialsvirtuallab/megnet/tree/f3705760266f24b62c4810dd4abacfc25f4f64ae),
  Regents of the University of California, BSD-style license. Stoichiometric means
  of frozen formation-energy-trained element vectors; not a full Roost/AtomSets model.
  Chen et al., [doi:10.1021/acs.chemmater.9b01294](https://doi.org/10.1021/acs.chemmater.9b01294).
- **Atomic-graph vectors:** official pretrained ALIGNN formation-energy checkpoint,
  [Figshare file 31458679](https://figshare.com/ndownloader/files/31458679), with NIST
  ALIGNN software terms retained in the notice JSON. Mean atom readout before the scalar
  head, extracted from the CC BY 4.0 JARVIS reference structures attributed below.
  All matching reference vectors are averaged; PCA is fitted only on unique active
  training compositions. Reference-coordinate variation is not calibrated uncertainty.
  Choudhary & DeCost, [doi:10.1038/s41524-021-00650-1](https://doi.org/10.1038/s41524-021-00650-1).

Source URLs, immutable revisions, hashes, checkpoint, graph parameters, package
versions and coverage/failures are recorded in the candidate JSON. These are
independent reference features, not identified HTEM source phases or measured film
information. No full crystal coordinates, model weights or spectra are redistributed
in these compact derived tables. Study scores are workshop-generated results.

`htem_chemistry_reference.json` contains a selected-property extraction from
[pymatgen v2025.1.9](https://github.com/materialsproject/pymatgen/tree/v2025.1.9)
and a composition-matched subset/summary of [Expt Gap Data v1](https://doi.org/10.6084/m9.figshare.7289687.v1),
distributed by matminer. The JSON records source URLs, download checksums, and aggregation rules.
Neither source is the HTEM deposited-film dataset. Derived summaries are not phase-resolved measurements.

Atomic properties: Copyright (c) 2011-2012 MIT & The Regents of the University of California,
through Lawrence Berkeley National Laboratory; pymatgen contributors.
[Original license](https://github.com/materialsproject/pymatgen/blob/v2025.1.9/LICENSE).

Band-gap dataset: Ya Zhuo, Aria Mansouri Tehrani, and Jakoah Brgoch,
*Predicting the Band Gaps of Inorganic Solids by Machine Learning*,
J. Phys. Chem. Lett. 9, 1668–1673 (2018), [doi:10.1021/acs.jpclett.8b00124](https://doi.org/10.1021/acs.jpclett.8b00124).
The Figshare distribution lists the MIT license. Reference records with the same composition can
differ in phase, preparation, temperature, or method. Median/range summarize the supplied records,
not a certified value for an HTEM source target. Missing source references remain missing.

## Additional source-DOS reference summaries (CC BY 4.0)

`htem_source_dos_reference.json` is a derived, composition-matched summary of
[JARVIS electronic/phonon DOS v3](https://doi.org/10.6084/m9.figshare.14745327.v3),
Figshare file 29216859 (`edos-up_pdos-elast_interp-8-18-2021.json.zip`). The dataset
is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Attribution: Prathik R. Kaundinya, Kamal Choudhary, and Surya R. Kalidindi,
*Prediction of the Electron Density of States for Crystalline Compounds with Atomistic
Line Graph Neural Networks (ALIGNN)*, JOM 74, 1395–1405 (2022),
[doi:10.1007/s11837-022-05199-y](https://doi.org/10.1007/s11837-022-05199-y), and the
[NIST JARVIS contributors](https://pages.nist.gov/jarvis/databases/).

Changes: retain only composition-matched `edos_up` records; unit-area normalize each
300-sample spin-up spectrum; calculate fixed energy-window fractions and moments;
average across reference structures and record their variability and identifiers.
No phonon spectra, crystal coordinates, or absolute DOS values are redistributed in
the compact summaries. Download checksum, source catalog checksum, selection rules,
coverage, and processing details are included in the JSON. These reference phases
are not verified HTEM source phases or measured deposited-film structures/DOS.
The MIT notice below applies to the original atomic/band-gap references, not this
separately attributed CC BY 4.0 source.

## MIT permission notice for the original references

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
