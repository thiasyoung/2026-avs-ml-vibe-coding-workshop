"""Mechanically build the participant-facing HTEM spatial notebook sequence."""
from pathlib import Path
import sys
import textwrap
import nbformat as nbf

MODULE = Path(__file__).resolve().parents[1] / "03_htem_conductivity"
SELECTED = set(sys.argv[1:])  # Optional relative notebook paths; omitted rebuilds the full sequence/examples.


def md(value):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


def save(name, cells):
    if SELECTED and name not in SELECTED:
        return
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python (workshop .venv)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    })
    nbf.write(notebook, MODULE / name)


setup = code('''
    from pathlib import Path
    import sys, json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    HERE = Path.cwd()
    if not (HERE / 'htem_workshop.py').exists():
        HERE = HERE / '03_htem_conductivity'
    if not (HERE / 'htem_workshop.py').exists():
        raise FileNotFoundError('Open this notebook from the workshop root or HTEM module folder.')
    sys.path.insert(0, str(HERE))
    import htem_workshop as htem
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
    SEED = 42
''')

save('00_HTEM_Get_Data_NREL.ipynb', [
    md('''
    # 00 · Download compact HTEM process records and substrate grids

    **Collect as much public data as the API exposes**, not just libraries with optical/electrical data.
    Retain deposition recipes, per-library positions, substrate `x_mm`/`y_mm`, local `absolute_temp_c`, thickness, conductivity,
    and optional direct bandgap. No spectra, measured composition, contacts, or full instrument records
    are retained. Missing properties remain missing here; cohort selection belongs in notebook 01.

    The working API is `https://htem-api.nlr.gov/api`. Its **scalar batch endpoint**
    `/sample_library/prop?ids=...` supplies aligned position/coordinate/property arrays. These are
    substrate-map coordinates, **not** the sample endpoint's `xyz_mm` stage/focus coordinates.
    The sample-level temperature can vary across a library; it must not be replaced by the library's
    initial temperature. Old compact scalar caches automatically refresh once to retrieve this field.
    The key is **(sample_library_id, position)**; position 1 in one deposition is not another observation's ID.

    Sources: [HTEM paper](https://doi.org/10.1038/sdata.2018.53),
    [current HTEM site and API documentation](https://htem.nlr.gov/),
    [official historical examples](https://github.com/NatLabRockies/htem-api-examples).
    ''') , setup,
    md('''
    ## Acquire all public libraries, with compact resumable checkpoints

    Default: **live acquisition with no library cap or measured-property filter**. Cached compact
    responses are reused. Set `REFRESH=True` to refresh them, or choose explicit `snapshot` mode for
    offline use. Network/source problems are reported, not disguised as a fresh successful full crawl.

    Most collection uses one recipe request per library plus batches of scalar maps. Rare libraries
    unsupported by the scalar endpoint fall back to individual sample responses; spectra in those
    responses are discarded **in memory before any cache is written**. If the substrate XY map is
    unavailable it stays missing—we never substitute stage coordinates or guess a 4×11 grid.
    ''') ,
    code('''
    import shutil
    ACQUISITION_MODE = 'live_api'  # explicit 'snapshot' for offline use
    LIBRARY_IDS = None            # None = EVERY public library; no cap
    REFRESH = False               # reuse compact per-library checkpoints
    WORKERS = 4                  # bounded recipe-download concurrency
    ACQUISITION = HERE / 'outputs/htem_acquisition'
    ACQUISITION.mkdir(parents=True, exist_ok=True)
    RAW_FILE = ACQUISITION / 'htem_process_raw.parquet'
    SNAPSHOT = HERE / 'data/htem_spatial_snapshot.parquet'
    LIBRARIES = ACQUISITION / 'htem_libraries.parquet'

    if ACQUISITION_MODE == 'live_api':
        raw = htem.download_htem(RAW_FILE, ACQUISITION / 'compact_http_cache_v2',
                                 library_ids=LIBRARY_IDS, workers=WORKERS, refresh=REFRESH)
        provenance = json.loads(RAW_FILE.with_suffix('.provenance.json').read_text(encoding='utf-8'))
        # Distribute the compact result, never raw HTTP payloads or spectra.
        shutil.copy2(RAW_FILE, SNAPSHOT)
        shutil.copy2(LIBRARIES, HERE / 'data/htem_spatial_libraries.parquet')
        htem.write_json(SNAPSHOT.with_suffix('.provenance.json'), provenance)
    elif ACQUISITION_MODE == 'snapshot':
        provenance = json.loads(SNAPSHOT.with_suffix('.provenance.json').read_text(encoding='utf-8'))
        assert htem.sha256(SNAPSHOT) == provenance['sha256'], 'Snapshot checksum mismatch.'
        shutil.copy2(SNAPSHOT, RAW_FILE)
        shutil.copy2(HERE / 'data/htem_spatial_libraries.parquet', LIBRARIES)
        raw = pd.read_parquet(RAW_FILE)
        provenance = {**provenance, 'mode': 'explicit_snapshot_reuse_NOT_a_new_download'}
        htem.write_json(RAW_FILE.with_suffix('.provenance.json'), provenance)
    else:
        raise ValueError("Choose 'live_api' or 'snapshot'.")
    assert raw.columns.tolist() == htem.RAW_FIELDS
    assert not raw.duplicated(['sample_library_id', 'position']).any()
    print(f"Mode: {provenance['mode']}; complete requested sample coverage: {provenance['complete']}")
    print(f"Public libraries: {provenance['public_libraries']:,}; recipes: {provenance['recipe_libraries']:,}")
    print(f"Retained points: {len(raw):,}; compact Parquet: {RAW_FILE.stat().st_size / 1e6:.2f} MB")
    print('Rows without substrate coordinates:', provenance['rows_missing_substrate_coordinates'])
    display(raw[['sample_library_id', 'position', 'x_mm', 'y_mm', 'absolute_temp_c', *htem.LABEL_FIELDS]].count().rename('available_values').to_frame())
    display(pd.DataFrame(provenance['failures']))
    display(pd.DataFrame(provenance['sample_count_gaps']))
    ''') ,
    md('''
    ## Inspect identity and geometry before cleansing

    All process recipes—including ones with no usable targets—are retained in `htem_libraries.parquet`.
    The point table retains scalar labels and coordinates even when other labels are missing.
    `id` is a human-readable **library:position composite key**, not the API's numeric sample ID.
    A geometry cannot be recovered by sorting those API IDs or by assuming all grids have 44 points.
    ''') ,
    code('''
    display(raw[['id', 'sample_library_id', 'position', 'x_mm', 'y_mm',
                 'deposition_compounds', 'deposition_power', 'thickness', 'fpm_conductivity']].head(8))
    display(raw.groupby('sample_library_id').size().value_counts().rename_axis('points_per_library').rename('libraries').to_frame())
    print('Only allowlisted process fields, identities, substrate XY, and scalar property labels are saved.')
    print('Missing XY is NOT filled from xyz_mm. Missing targets are NOT imputed during acquisition.')
    ''') ,
])

save('01_HTEM_Process_Data_Carpentry.ipynb', [
    md('''
    # 01 · Completed data carpentry for a spatial process emulator

    **Learning problem:** (deposition recipe, requested substrate x, requested substrate y)
    → **local thickness and conductivity**. Include the planned/calibrated temperature at each location,
    not only the library's initial temperature. Each deposition has a set of distinct sampling locations.
    We keep one observation per **library + position**, never average a whole substrate into one row,
    and never confuse position indices across depositions.

    Recipe settings and a requested grid are available before deposition. Measured spectra, composition,
    optical summaries, electrical measurements, quality, operator identity, IDs, and focus/stage `xyz_mm`
    are not features. Thickness and conductivity are targets only. Bandgap stays in the compact raw
    dataset but is not required for this two-output mapping exercise, increasing usable coverage.
    ''') , setup,
    md('''
    ## 1. Examine coverage, units, missingness, and grid integrity

    Substrate coordinates come from the API's aligned scalar-map arrays, not an inferred reshape or
    post-measurement stage coordinates. Preserve the API's substrate frame and grid orientation; do not
    fit a coordinate transformation from measured property values. Unknown XY is excluded from spatial
    training, not imputed. A new-process prediction must supply a known requested grid in this same frame.
    ''') ,
    code('''
    RAW_FILE = HERE / 'outputs/htem_acquisition/htem_process_raw.parquet'
    PREPARED = HERE / 'outputs/htem_prepared'
    provenance = json.loads(RAW_FILE.with_suffix('.provenance.json').read_text(encoding='utf-8'))
    assert htem.sha256(RAW_FILE) == provenance['sha256']
    raw = pd.read_parquet(RAW_FILE)
    print(f"{len(raw):,} acquired points, {raw.sample_library_id.nunique():,} libraries; mode={provenance['mode']}")
    print('Units: substrate x/y = mm; thickness = um; conductivity = S/cm; optional bandgap = eV.')
    display(raw[['x_mm', 'y_mm', *htem.RECIPE_FIELDS, *htem.LABEL_FIELDS]].isna().sum().rename('missing_rows').to_frame())
    display(raw[['thickness', 'fpm_conductivity', 'x_mm', 'y_mm']].describe().T)
    assert not raw.duplicated(['sample_library_id', 'position']).any()
    ''') ,
    md('''
    ## 2. Clean and encode without conflating positions

    The supplied implementation keeps a position with **either** usable thickness or conductivity.
    Thickness outside 0.001–10 µm and nonpositive conductivity become missing labels, not guessed values;
    the other valid label can still train its output. Positions with neither label are excluded.
    Additional exclusion rules: invalid/missing XY, ambiguous within-library XY, unknown/invalid duration or initial
    temperature, or inconsistent powered-target settings. No labels are guessed, clipped, or imputed.
    Extreme conductivity is reviewed, not removed simply for being unusual. This is a powered-target
    process cohort, not a claim to model every deposition method in the raw inventory.

    Preserve source/power and gas/flow slot pairing. Normalize gas/substrate case, not chemical formula
    case. Missing active-target power is not zero. Optional missing settings use train-only imputation
    with indicators; **spatial coordinates are never imputed**. Substrate XY enters the model; position
    and library IDs remain metadata only. Extra pulse/energy/cycle/distance settings are retained in the
    raw download, not forced into this sputtering-focused reference feature set.

    `absolute_temp_c` becomes `local_temperature_C`, retained independently at every planned grid node,
    including nodes without labels. Missing/invalid local temperature remains unknown with a train-fitted
    imputation indicator. It is never silently filled from initial temperature. For a new recipe, supply
    the planned/calibrated profile in the same substrate frame; the field is not a complete thermal history.

    ### Source chemistry instead of formula-name one-hot columns

    Each target formula supplies atom-fraction-weighted atomic number/mass, Pauling electronegativity
    (mean/range/std), atomic radius, first ionization energy, composition entropy, and O/N/S/Se/Te
    fractions. Source/power pairing is preserved. We also include independent experimental **reference**
    band-gap median/range where the formula is covered. These are **not** measured HTEM film gaps or
    certified values for a specific target phase. A formula alone does not identify polymorph, doping,
    defects, or measurement temperature. Missing references stay missing with training-only imputation
    and explicit known/unknown indicators; no gap is invented or averaged from elemental gaps.

    Power-weighted source descriptors are engineered heuristics, **not deposited-film composition**
    or a physical mixture band gap. Formula names stay as metadata for recipe identity/material strata,
    but are not one-hot encoded. Chamber, substrate, and gas still use train-fitted categories.
    Atomic data: [pymatgen v2025.1.9](https://github.com/materialsproject/pymatgen/tree/v2025.1.9).
    Band-gap data: [Zhuo et al. (2018)](https://doi.org/10.1021/acs.jpclett.8b00124),
    [Expt Gap Data v1](https://doi.org/10.6084/m9.figshare.7289687.v1).
    ### Electronic DOS references: richer chemistry, without inventing a source phase

    Independent [JARVIS electronic DOS](https://doi.org/10.6084/m9.figshare.14745327.v3) supplies
    **reference spin-up DOS shapes**, matched by exact normalized source composition. We retain all
    matching structures, not an arbitrarily selected polymorph. Each spectrum uses the published 300
    samples at 0.05 eV spacing (-5 inclusive, 10 exclusive relative to Fermi); integrals stay within the
    sampled -5..9.95 eV support. Unit-area normalization makes these shape features, not absolute DOS.

    Per source: spectral centroid/width, occupied-window and near-Fermi (+/-0.5 eV) fractions,
    seven coarse energy-bin fractions, reference count, near-Fermi range and mean bin spread across
    reference structures. The last two are reference variability, **not calibrated uncertainty**.
    Power-weighted centroid/near-Fermi summaries require full active-source coverage; otherwise they
    stay missing. Coverage indicators and train-only imputation are explicit. They are not a physical
    mixture DOS, deposited composition, carrier density, or predicted film conductivity.

    A source formula does not identify its actual atomic coordinates, defects or phase. The instructor
    also tested **reference-crystal** embeddings: real JARVIS lattices/species/coordinates passed through
    the official frozen ALIGNN formation-energy model, pooled before its scalar prediction head.
    All matching reference vectors are averaged and compressed by PCA fit only on unique active
    **training** compositions. This is an honest reference-structure proxy, not identification of an
    actual HTEM target or film phase. The existing substrate-grid GNN is a different, spatial graph.
    No measured HTEM film spectra/structure enter this reference lookup.
    Reference: [Kaundinya et al., DOS/ALIGNN (2022)](https://doi.org/10.1007/s11837-022-05199-y).
    The compact, checksummed reference summaries ship with the repository; no new packages/API keys
    are needed for preparation or training. To reproduce the external download and extraction, run
    `uv run --locked python 03_htem_conductivity/htem_source_dos.py` from the repository root.
    The 155 MB raw reference archive stays in ignored outputs, not the distributed dataset.

    ### Instructor-selected frozen features: your experiment changes the model

    An instructor-only comparison in `scratch/HTEM_Feature_Selection_Study.ipynb` tested curated
    Magpie-style descriptors, pretrained composition vectors, reference-crystal ALIGNN vectors,
    and DOS combinations. It used fixed Ridge/Random Forest selectors and the existing grouped
    training/validation split. **No test artifact or score was used to select features.**
    The lowest mean relative validation RMSE determines `data/htem_feature_selection.json`.
    Improvements on one holdout are not proof of universal superiority; the selection record
    includes per-selector results and an exploratory whole-group bootstrap interval.

    Composition vectors are stoichiometric means of published **16-D MEGNet element vectors**,
    not a full Roost/AtomSets network. Curated additions include periodic/valence/radius statistics,
    stoichiometry norms and elemental melting-temperature statistics, not a guessed compound melting
    point. [MEGNet](https://doi.org/10.1021/acs.chemmater.9b01294),
    [ALIGNN](https://doi.org/10.1038/s41524-021-00650-1),
    [Magpie](https://doi.org/10.1038/npjcompumats.2016.28).

    This notebook loads the **selected** offline source columns and fits imputation/scaling on training
    data only. Candidate blocks not selected are not silently added. All five architecture notebooks
    read the exact same saved matrices and transforms. No DGL, ALIGNN, external model download,
    feature selection, or additional API key is needed during the workshop.
    ''') ,
    code('''
    FROZEN_SPLIT = HERE / 'data/frozen_evaluation/paired_split.csv'
    clean, audit, manifest = htem.prepare(raw, PREPARED, htem.sha256(RAW_FILE), seed=SEED,
                                         frozen_split=FROZEN_SPLIT)
    print('Frozen instructor-selected feature set:', manifest['frozen_feature_selection']['candidate'])
    print('Selected optional blocks:', manifest['frozen_feature_selection']['blocks'])
    print(f"Retained {len(clean):,} points from {manifest['libraries']} libraries.")
    display(audit.loc[audit.excluded, 'reasons'].str.split(';').explode().value_counts().rename('excluded_rows').to_frame())
    display(pd.DataFrame({'feature': htem.MODEL_FEATURES,
                          'encoding': ['train-impute optional settings + train-scale; XY always known'] * len(htem.NUMERIC_FEATURES)
                                      + ['train-fitted one-hot'] * len(htem.CATEGORICAL_FEATURES)}))
    source_coverage = pd.read_csv(PREPARED / 'source_chemistry_coverage.csv')
    display(source_coverage)
    print(f"Formula descriptors: {source_coverage.formula_known.sum()}/{len(source_coverage)} source labels; "
          f"independent band-gap reference: {source_coverage.reference_bandgap_eV.notna().sum()}/{len(source_coverage)}.")
    print(f"Independent DOS reference: {source_coverage.reference_dos_known.sum()}/{len(source_coverage)} source labels. "
          'Reference phases are not verified source phases; missing coverage stays explicit.')
    assert not any(name.startswith('categorical__target_') for name in manifest['encoded_features'])
    ''') ,
    md('''
    ## 2a. Worked feature workflow: the selected representation only

    The instructor comparison selected **curated composition descriptors + frozen reference-crystal
    ALIGNN embeddings + reference DOS**. MEGNet composition embeddings were tested but are **not**
    used here. The combined selector score improved about 2.6% over the earlier DOS enrichment;
    the exploratory validation-group interval includes no improvement, so this is not a claim of
    statistically established superiority or an optimal representation for every model.

    **Expensive extraction, already performed:** download the pinned official ALIGNN formation-energy
    checkpoint, build periodic atomic/bond-angle graphs from real JARVIS reference structures,
    extract the mean atom representation before the prediction head, and average all exact-composition
    references. The function below shows that actual graph-encoding operation. It is defined here but
    not rerun during class: its optional ALIGNN/DGL environment and pinned model loader are in
    `scratch/build_htem_feature_candidates.py` (see the module README for the exact commands).
    The instructor smoke-tests this same function against a cached real structure/vector.

    DOS shape reduction/aggregation is implemented in the linked, runnable `htem_source_dos.py`;
    the supplied compact summaries avoid shipping full spectra. Neither reference pathway uses
    measured HTEM film information. Unknown references remain missing, not guessed.
    '''),
    code('''
    def encode_reference_crystal(frozen_model, atoms_dictionary):
        """Optional instructor extraction: real periodic graph -> frozen 256-D vector."""
        import torch
        from alignn.graphs import Graph
        from jarvis.core.atoms import Atoms
        atoms = Atoms.from_dict(atoms_dictionary)
        graph, line_graph = Graph.atom_dgl_multigraph(
            atoms, cutoff=8.0, max_neighbors=12, atom_features='cgcnn', compute_line_graph=True)
        captured = []
        hook = frozen_model.fc.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().cpu().numpy().copy()))
        frozen_model.eval()
        frozen_model.requires_grad_(False)
        try:
            with torch.inference_mode():
                frozen_model([graph, line_graph, torch.tensor(atoms.lattice_mat, dtype=torch.float32)])
        finally:
            hook.remove()
        return captured[-1].reshape(-1)  # official mean atom readout before the scalar head
    '''),
    md('''
    **Run this part in class:** rebuild the selected elemental statistics, fit the crystal PCA using
    training-source compositions only, assemble source-slot features, then fit training-only median
    imputation and StandardScaler. Transform validation/test without fitting on them. The assertions
    below check that the resulting matrices exactly reproduce the instructor's frozen preparation.
    No feature comparison or new representation choice happens during this step.
    '''),
    code('''
    from sklearn.decomposition import PCA
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    import htem_chemistry as chemistry

    reference = json.loads((HERE / 'data/htem_feature_candidates.json').read_text(encoding='utf-8'))
    elemental = json.loads((HERE / 'data/htem_feature_element_reference.json').read_text(encoding='utf-8'))
    frozen_source = htem.frozen_features.source_table()
    training_recipes = clean[clean.split.eq('train')].drop_duplicates('sample_library_id')
    training_keys = sorted({chemistry.composition_key(row[f'target_{i}'])
        for row in training_recipes.to_dict('records') for i in range(1, 4)
        if row[f'target_{i}_power_W'] > 0
        and reference['compositions'][chemistry.composition_key(row[f'target_{i}'])]['crystal_embedding']})
    training_vectors = np.array([reference['compositions'][key]['crystal_embedding']['mean'] for key in training_keys])
    crystal_pca = PCA(n_components=16, svd_solver='full').fit(training_vectors)
    assert training_keys == frozen_source['projection']['training_composition_keys']
    np.testing.assert_array_equal(crystal_pca.components_, frozen_source['projection']['components'])

    def selected_source_features(formula):
        fields = frozen_source['blocks']['curated'] + frozen_source['blocks']['crystal']
        if formula == '__NONE__':
            return dict.fromkeys(fields, 0.0)  # absent source, not an unknown material
        parsed = chemistry.parse_formula(formula)
        weights = np.array([float(fraction) for element, fraction in parsed])
        values = {'curated_known': 1.0}
        for prop in elemental['property_order']:
            v = np.array([elemental['elements'][element][prop] for element, fraction in parsed], dtype=float)
            values['magpie_mean_' + prop] = float(weights @ v)
            if prop not in ['IsDBlock', 'NsValence', 'NpValence', 'NdValence', 'NfValence', 'NValence']:
                values['magpie_range_' + prop] = float(np.ptp(v))
                values['magpie_std_' + prop] = float(np.sqrt(weights @ (v - weights @ v) ** 2))
        for orbital in 'spdf':
            values['valence_fraction_' + orbital] = values[f'magpie_mean_N{orbital}Valence'] / values['magpie_mean_NValence']
        values['stoichiometry_L2'] = float(np.linalg.norm(weights))
        values['stoichiometry_L3'] = float(np.linalg.norm(weights, ord=3))
        crystal = reference['compositions'].get(chemistry.composition_key(formula), {}).get('crystal_embedding')
        values['crystal_embedding_known'] = float(crystal is not None)
        values['crystal_reference_mean_coordinate_std'] = crystal['mean_coordinate_std'] if crystal else np.nan
        vector = crystal_pca.transform(np.array(crystal['mean']).reshape(1, -1))[0] if crystal else np.full(16, np.nan)
        values.update({f'alignn_pc_{i:02d}': float(value) for i, value in enumerate(vector)})
        return values

    worked_inputs = clean[htem.FEATURES].copy()
    for slot in range(1, 4):
        lookup = {formula: selected_source_features(formula) for formula in clean[f'target_{slot}'].unique()}
        rebuilt = pd.DataFrame([lookup[formula] for formula in clean[f'target_{slot}']], index=clean.index)
        for feature in rebuilt:
            worked_inputs[f'target_{slot}_{feature}'] = rebuilt[feature].to_numpy()
    # Existing recipe/atomic/reference-gap/DOS features remain unchanged; replace
    # only the freshly demonstrated curated descriptors and ALIGNN coordinates.
    encoder = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median', add_indicator=True,
             keep_empty_features=True), StandardScaler()), htem.NUMERIC_FEATURES),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), htem.CATEGORICAL_FEATURES)])
    encoder.fit(worked_inputs[clean.split.eq('train')])
    for split in htem.SPLITS:
        reproduced = encoder.transform(worked_inputs[clean.split.eq(split)]).astype('float32')
        np.testing.assert_array_equal(reproduced, data[split]['X'] if 'data' in globals()
                                      else np.load(PREPARED / f'{split}.npz', allow_pickle=False)['X'])
    print(f'Rebuilt {len(training_keys)} training-source reference embeddings -> 16 crystal coordinates per source.')
    print('Curated statistics + reference crystal PCA + existing DOS/recipe inputs exactly reproduce all three frozen matrices.')
    print('All architecture notebooks consume those matrices unchanged; your task is to experiment with the model.')
    '''),
    md('''
    ## 3. Balance material classes while keeping whole depositions together

    All grid locations of a library stay in one split. Libraries with identical recipe vectors also
    stay together, **even when sampled on different grids**. Otherwise the model could memorize one
    deposition and appear to predict an unseen nearby position.

    Stratify by the **combination of active target materials** (positive-power sources), independent
    of their ordering. This is a recipe category, not measured film composition. Source slots and
    source/power pairing remain unchanged in the actual model inputs. Aim for 60/20/20 independent
    groups **within each material class**; do not duplicate maps or force equally frequent classes.
    Every class with at least three independent groups appears in train, validation, and test.
    A one-group class stays in training; a two-group class gets training plus one holdout. Those
    classes cannot be represented in all three splits without leakage. Integer quotas and unequal
    grid sizes mean overall group/point fractions will differ from 60/20/20.

    The balance table reports independent groups, libraries, and points separately. The splitter
    uses categories and group sizes, never property values or model performance. Recipe identity uses
    original process settings/formulas, not derived chemical descriptors: adding descriptors leaves
    the established material-balanced assignments unchanged. Rerun all models after preparation changes.
    Test remains for final reporting after validation selection.

    **Expanding without moving the old holdouts:** `data/frozen_evaluation/paired_split.csv` freezes the
    original observations and all original library assignments. New connected groups fill material-class
    split deficits without moving any original library. New positions on old libraries inherit their split.
    Metrics on the expanded holdouts and on the **original paired test points** are reported separately;
    they are not interchangeable. Archived old runs are retained locally for before/after inspection.

    Both targets use log10 followed by independent training-only standardization. Input scaling,
    imputation, categorical vocabularies, and target transforms use training data only. IDs, position,
    and original XY are retained beside arrays so every prediction returns to the correct map point.
    ''') ,
    code('''
    data, transforms, manifest = htem.load_prepared(PREPARED)
    display(clean.groupby('split').agg(points=('id', 'size'), libraries=('sample_library_id', 'nunique'),
                                      recipe_groups=('leakage_group', 'nunique')).reindex(['train', 'validation', 'test']))
    balance = pd.read_csv(PREPARED / 'material_split_balance.csv')
    display(balance[['material_class', 'total_groups', 'train_groups', 'validation_groups', 'test_groups', 'coverage_note']])
    summary = manifest['material_stratification']
    print(f"{summary['classes_present_in_all_splits']}/{summary['material_classes']} material classes occur in every split.")
    print(f"Rare exceptions: {summary['singleton_classes_train_only']} singleton classes; "
          f"{summary['two_group_classes_train_plus_one_holdout']} two-group classes. See the saved balance CSV.")
    assert balance.loc[balance.all_splits_feasible, 'all_splits_present'].all()
    assert balance.train_groups.gt(0).all()
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        a, b = clean[clean.split.eq(left)], clean[clean.split.eq(right)]
        assert set(a.sample_library_id).isdisjoint(b.sample_library_id)
        assert set(a.leakage_group).isdisjoint(b.leakage_group)
    display(pd.DataFrame(manifest['label_counts']).T.rename_axis('split'))
    print(f"Finite uniform inputs: {data['train']['X'].shape[1]} features → 2 targets with explicit observed-label masks.")
    print('Original paired evaluation counts:', manifest['frozen_evaluation']['counts'])
    print('Frozen full raw substrate grids saved separately, BEFORE target filtering.')
    ''') ,
    md('''
    ## 4. Examine anomalies using training-only references

    Statistical flags are prompts for inspection, not automatic deletion. Missing-data selection and
    sparse chemistry still limit generalization. Notebook 02 changes models, not this prepared cohort.
    ''') ,
    code('''
    training = clean[clean.split.eq('train')]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
    flags = []
    for ax, target in zip(axes, htem.TARGETS):
        values = training[target].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        unusual = ~values.between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
        flags.append({'target': target, 'IQR_review_flags': int(unusual.sum()), 'action': 'retain; not proven wrong'})
        ax.hist(values, bins=30, color='#236b8e')
        ax.set(xlabel=target, ylabel='Training points')
    fig.tight_layout()
    fig.savefig(PREPARED / 'training_target_distributions.png', bbox_inches='tight')
    plt.show()
    display(pd.DataFrame(flags))
    display(pd.DataFrame([{'feature': c, 'categories': training[c].nunique(),
                           'categories_in_only_one_library': int((training.groupby(c).sample_library_id.nunique() == 1).sum())}
                          for c in htem.CATEGORICAL_FEATURES]))
    ''') ,
])

# Model examples and comparison are maintained in one separate builder.
from build_htem_algorithm_notebooks import build as build_algorithms
build_algorithms(SELECTED)
print('Built selected spatial HTEM notebooks.')
