"""Shared IR examination and file contract; the strategy code is in the notebooks."""
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from rdkit import Chem

from ir_carpentry import validate_frozen_evaluation, native_arrays

LABELS = ['Vinylic', 'Arene', 'Alcohol', 'Ether', 'Amine', 'Nitrile',
          'Aldehyde', 'Ketone', 'CarboxylAcid', 'Ester', 'Amide', 'AcidChloride']
SEED = 42
# Intentionally the exact operational label definitions used by IR 00.
SMARTS = {
    'Vinylic': '[$([CX3]=[CX3])]', 'Arene': 'c', 'Alcohol': '[#6][OX2H]',
    'Ether': '[OD2]([#6])[#6]', 'Amine': '[NX3;H2,H1;!$(NC=O)]', 'Nitrile': '[NX1]#[CX2]',
    'Aldehyde': '[CX3H1](=O)[#6]', 'Ketone': '[#6][CX3](=O)[#6]',
    'CarboxylAcid': '[OX2H][CX3]=[OX1]', 'Ester': '[#6][CX3](=O)[OX2H0][#6]',
    'Amide': '[NX3][CX3](=[OX1])[#6]', 'AcidChloride': '[CX3](=[OX1])[F,Cl,Br,I]',
}
PATTERNS = {label: Chem.MolFromSmarts(pattern) for label, pattern in SMARTS.items()}


def examine_native(pool, grid, output):
    """Training-only raw audit before any unit, bounds or anomaly policy."""
    output.mkdir(parents=True, exist_ok=True)
    units = pool.groupby(['xunits', 'yunits'], dropna=False).size().rename('rows').to_frame()
    print('Native training-candidate units (not the held-out reference):')
    display(units)
    units.to_csv(output / 'native_unit_counts.csv')
    missing = pool[['SMILES', 'CAS_NO', *LABELS]].isna().sum()
    display(missing.to_frame('missing_cells').T)
    labels = pd.DataFrame({'positive': pool[LABELS].sum(),
                           'invalid_or_missing': (~pool[LABELS].isin([0, 1])).sum()})
    display(labels)
    canonical = pool.SMILES.map(canonical_smiles)
    print('Invalid SMILES / duplicate canonical molecules:', int(canonical.isna().sum()),
          int(canonical.dropna().duplicated().sum()))
    records = []
    for _, row in pool.iterrows():
        result = dict(NIST_molecule=row.NIST_molecule, native_points=len(row.y_raw))
        try:
            x, y, _ = native_arrays(row, 'convert_transmittance')
            result.update(status='valid_explicit_units', low=x.min(), high=x.max(),
                          signal_min=y.min(), signal_max=y.max(),
                          coverage=np.mean((grid >= x[0]) & (grid <= x[-1])))
        except (ValueError, TypeError) as error:
            result['status'] = str(error)
        records.append(result)
    statistics = pd.DataFrame(records)
    display(statistics.status.value_counts().to_frame('rows'))
    display(statistics[['native_points', 'low', 'high', 'signal_min', 'signal_max', 'coverage']].quantile([0, .01, .5, .99, 1]))
    statistics.to_csv(output / 'native_spectrum_audit.csv', index=False)
    labels.to_csv(output / 'category_audit.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    statistics.native_points.plot.hist(ax=axes[0], bins=35)
    axes[0].set(title='Unequal native lengths', xlabel='Recorded points')
    statistics.coverage.dropna().plot.hist(ax=axes[1], bins=25)
    axes[1].set(title='Coverage after explicit unit conversion', xlabel='Fraction of fixed grid measured')
    pool[LABELS].sum().sort_values().plot.barh(ax=axes[2])
    axes[2].set(title='Training category counts', xlabel='Positive records')
    fig.tight_layout(); fig.savefig(output / 'examination.png', dpi=140); plt.show()
    return statistics


def cleanse_training(train, mode):
    """One extra, auditable training-only hypothesis per consolidated example."""
    train, audit = train.copy(), []
    if mode == 'canonical_deduplication':
        keys = train.SMILES.map(canonical_smiles)
        # Quarantine contradictory duplicate labels; otherwise keep one observation.
        grouped = train.assign(_canonical=keys).groupby('_canonical')[LABELS].nunique().max(axis=1)
        contradictory = keys.isin(grouped[grouped > 1].index)
        duplicate = keys.duplicated(keep='first') & ~contradictory
        for index in train.index[contradictory | duplicate]:
            audit.append(dict(NIST_molecule=train.loc[index, 'NIST_molecule'], action='removed_from_training',
                              reason='contradictory_duplicate_labels' if contradictory.loc[index] else 'repeat_canonical_molecule'))
        train = train.loc[~(contradictory | duplicate)].copy()
    elif mode == 'smarts_label_check':
        corrections = 0
        for index, row in train.iterrows():
            molecule = Chem.MolFromSmiles(row.SMILES)
            for label, pattern in PATTERNS.items():
                expected = int(molecule.HasSubstructMatch(pattern))
                if row[label] != expected:
                    train.loc[index, label] = expected
                    corrections += 1
                    audit.append(dict(NIST_molecule=row.NIST_molecule, action='corrected_label', reason=label))
        print('Stored labels corrected to the documented operational SMARTS:', corrections)
        print('This is internal consistency, not independent experimental ground-truth validation.')
    elif mode == 'roughness_quantile':
        roughness = train.IR.map(lambda y: float(np.max(np.abs(np.diff(y)))))
        cutoff = float(roughness.quantile(.995))
        remove = roughness > cutoff
        audit.extend(dict(NIST_molecule=key, action='removed_from_training', reason='training_roughness_above_q995')
                     for key in train.loc[remove, 'NIST_molecule'])
        print(f'Training-only 99.5th-percentile jump cutoff: {cutoff:.5f}; removed {remove.sum()} rows.')
        print('An aggressive hypothesis: sharp real peaks and padding edges can also be flagged.')
        train = train.loc[~remove].copy()
    elif mode == 'capped_rare_oversampling':
        counts = train[LABELS].sum()
        target = int(counts[counts > 0].quantile(.25))
        rng = np.random.default_rng(SEED)
        presentations = np.ones(len(train), dtype=int)
        additions = []
        for label in counts.index[counts < target]:
            needed = max(0, target - int(counts[label]))
            for _ in range(needed):
                available = np.flatnonzero(train[label].eq(1).to_numpy() & (presentations < 3))
                if not len(available):
                    break
                position = int(rng.choice(available))
                presentations[position] += 1
                additions.append(position)
                audit.append(dict(NIST_molecule=train.iloc[position].NIST_molecule,
                                  action='repeated_training_presentation', reason=f'rare_{label}'))
        train = pd.concat([train, train.iloc[additions]], ignore_index=True)
        print(f'Added {len(additions)} presentations, not new molecules; cap three per original observation.')
        print('This changes co-occurring category frequencies and combines with the static model class weights.')
    else:
        raise ValueError(f'Unknown cleansing hypothesis: {mode}')
    return train.reset_index(drop=True), pd.DataFrame(audit, columns=['NIST_molecule', 'action', 'reason'])


def module_directory():
    for parent in [Path.cwd(), *Path.cwd().parents]:
        for candidate in (parent, parent / '02_ir_functional_groups'):
            if (candidate / 'lib/ir_carpentry.py').is_file():
                return candidate.resolve()
    raise FileNotFoundError('Open the workshop repository, IR module, or examples folder.')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_smiles(value):
    molecule = Chem.MolFromSmiles(value) if isinstance(value, str) and value.strip() else None
    return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule is not None else None


def parse_spectrum(grid, signal):
    grid, signal = np.asarray(grid, dtype=float), np.asarray(signal, dtype=float)
    if grid.ndim != 1 or signal.ndim != 1 or len(grid) != len(signal) or len(grid) < 2:
        raise ValueError('invalid_array_shape')
    if not (np.isfinite(grid).all() and np.isfinite(signal).all()):
        raise ValueError('non_finite_array')
    order = np.argsort(grid, kind='stable')
    grid, signal = grid[order], signal[order]
    unique, inverse = np.unique(grid, return_inverse=True)
    if len(unique) < 2:
        raise ValueError('insufficient_unique_coordinates')
    return unique, np.bincount(inverse, weights=signal) / np.bincount(inverse)


def load_inputs(module):
    baseline = module / 'outputs/carpentry'
    manifest_file = baseline / 'data_manifest.json'
    if not manifest_file.exists() or not all((baseline / f'IR_{s}.parquet').exists() for s in ('train', 'validation', 'test')):
        raise FileNotFoundError('Run examples/01_IR_Vibe_Coding_Data_Carpentry.ipynb first to prepare the reference baseline.')
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest['labels'] != LABELS:
        raise ValueError('Baseline labels differ from the examples; review the label contract.')
    declared = Path(manifest['source_file'])
    source_path = declared if declared.is_absolute() else module / declared
    tables = {name: pd.read_parquet(baseline / f'IR_{name}.parquet') for name in ('train', 'validation', 'test')}
    validate_frozen_evaluation(module, baseline, manifest['wavenumber'])
    # Most examples audit the baseline prepared view; native unit/bounds variants
    # deliberately return to the checkpoint rather than repairing prefiltered rows.
    source = pd.concat(tables.values(), ignore_index=True).drop_duplicates('NIST_molecule')
    print(f'Native source: {source_path.relative_to(module)}; prepared reference view: {len(source):,} spectra')
    print('Baseline rows:', {name: len(frame) for name, frame in tables.items()})
    print('Frozen test SHA-256:', sha256_file(baseline / 'IR_test.parquet'))
    return source, source_path, tables, manifest, np.asarray(manifest['wavenumber'], dtype=float)


def spectrum_statistics(frame):
    records = []
    for index, row in frame.iterrows():
        r = dict(row=index, NIST_molecule=row.NIST_molecule, finite=False, increasing=False,
                 points=0, grid_min=np.nan, grid_max=np.nan, minimum=np.nan,
                 maximum=np.nan, span=np.nan, roughness=np.nan)
        try:
            x, y = np.asarray(row.wavenumber, dtype=float), np.asarray(row.IR, dtype=float)
            r['points'] = y.size
            if x.ndim == y.ndim == 1 and len(x) == len(y) and len(x) > 1 and np.isfinite(x).all() and np.isfinite(y).all():
                span = np.ptp(y)
                r.update(finite=True, increasing=bool((np.diff(x) > 0).all()), grid_min=x.min(), grid_max=x.max(),
                         minimum=y.min(), maximum=y.max(), span=span,
                         roughness=np.max(np.abs(np.diff(y))) / max(span, 1e-12))
        except (TypeError, ValueError):
            pass
        records.append(r)
    return pd.DataFrame(records)


def examine(source, training, module, output):
    output.mkdir(parents=True, exist_ok=True)
    required = ['NIST_molecule', 'CAS_NO', 'SMILES', 'IR', 'wavenumber', *LABELS]
    print('Missing table cells (array contents checked separately):')
    display(source[required].isna().sum().to_frame('missing_cells').T)
    label_audit = pd.DataFrame({
        'observed_values': [str(source[label].value_counts(dropna=False).to_dict()) for label in LABELS],
        'invalid_or_missing': [(~source[label].isin([0, 1])).sum() for label in LABELS],
        'training_positives': training[LABELS].sum()}, index=LABELS)
    display(label_audit)
    canonical = source.SMILES.map(canonical_smiles)
    print('Invalid SMILES:', int(canonical.isna().sum()))
    print('Duplicate NIST / CAS / canonical SMILES:', source.NIST_molecule.duplicated().sum(),
          source.CAS_NO.duplicated().sum(), canonical.dropna().duplicated().sum())
    statistics = spectrum_statistics(source)
    print('NaN/Inf or malformed arrays:', int((~statistics.finite).sum()))
    print('Non-increasing grids:', int((statistics.finite & ~statistics.increasing).sum()))
    print('Array lengths:', statistics.points.value_counts().to_dict())
    display(statistics[['grid_min', 'grid_max', 'minimum', 'maximum', 'span', 'roughness']].quantile([0, .01, .5, .99, 1]))
    raw_file = module / 'outputs/nist_acquisition/ir_raw_checkpoint.parquet'
    if raw_file.is_file():
        units = pd.read_parquet(raw_file, columns=['xunits', 'yunits'])
        unit_counts = units.fillna('<missing>').groupby(['xunits', 'yunits']).size().rename('rows').reset_index()
        print('Native acquisition unit categories, before IR 01 carpentry:')
        display(unit_counts)
        unit_counts.to_csv(output / 'raw_unit_counts.csv', index=False)
    else:
        print('No raw checkpoint: original units cannot be verified from normalized arrays.')
    print('IR is normalized signal, not absolute absorbance. Never infer unit conversion from amplitude alone.')
    print('Statistical tails and unusual category combinations require review; rare chemistry is not necessarily corrupt.')
    targets = training[LABELS].to_numpy(dtype=int)
    cooccurrence = pd.DataFrame(targets.T @ targets, index=LABELS, columns=LABELS)
    train_stats = spectrum_statistics(training)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    training[LABELS].sum().sort_values().plot.barh(ax=axes[0], color='#4477AA')
    axes[0].set(title='Training label counts', xlabel='Positive examples')
    axes[1].hist(train_stats.roughness.dropna(), bins=45, color='#4477AA')
    axes[1].set(title='Training adjacent-point jumps / span', xlabel='Roughness score', ylabel='Spectra')
    image = axes[2].imshow(cooccurrence, aspect='auto', cmap='Blues')
    axes[2].set_xticks(range(len(LABELS)), LABELS, rotation=90, fontsize=7)
    axes[2].set_yticks(range(len(LABELS)), LABELS, fontsize=7)
    axes[2].set_title('Training category co-occurrence')
    fig.colorbar(image, ax=axes[2], shrink=.7)
    fig.tight_layout(); fig.savefig(output / 'examination.png', dpi=140, bbox_inches='tight'); plt.show()
    label_audit.to_csv(output / 'category_audit.csv')
    statistics.to_csv(output / 'spectrum_audit.csv', index=False)
    cooccurrence.to_csv(output / 'training_label_cooccurrence.csv')
    return train_stats


def guard_training_identities(tables, audit):
    train = tables['train'].copy()
    canonical = train.SMILES.map(canonical_smiles)
    heldout = pd.concat([tables['validation'], tables['test']], ignore_index=True)
    overlaps = canonical.isin(set(heldout.SMILES.map(canonical_smiles).dropna()))
    invalid = canonical.isna()
    audit.extend({'NIST_molecule': train.loc[i, 'NIST_molecule'],
                  'reason': 'invalid_smiles' if invalid.loc[i] else 'canonical_molecule_in_holdout',
                  'action': 'removed_from_training'} for i in train.index[invalid | overlaps])
    print(f'Common identity guard: removed {int((invalid | overlaps).sum())} training rows; holdouts unchanged.')
    return train.loc[~(invalid | overlaps)].copy().reset_index(drop=True)


def validate_tables(tables, grid):
    for name, frame in tables.items():
        if frame.empty:
            raise ValueError(f'{name} is empty.')
        arrays = np.vstack(frame.IR.map(lambda x: np.asarray(x, dtype=np.float32)))
        targets = frame[LABELS].to_numpy()
        if arrays.shape != (len(frame), len(grid)) or not np.isfinite(arrays).all():
            raise ValueError(f'{name}: invalid IR arrays.')
        if not frame.wavenumber.map(lambda x: np.array_equal(np.asarray(x, dtype=np.float32), grid.astype(np.float32))).all():
            raise ValueError(f'{name}: grid mismatch.')
        if not np.isin(targets, [0, 1]).all() or (targets.sum(axis=0) == 0).any():
            raise ValueError(f'{name}: invalid categories or absent positives.')
    for key in ('NIST_molecule', 'CAS_NO'):
        sets = [set(frame[key]) for frame in tables.values()]
        if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
            raise ValueError(f'{key} leaks between splits.')
    train_keys = set(tables['train'].SMILES.map(canonical_smiles))
    holdout_keys = set(pd.concat([tables['validation'], tables['test']]).SMILES.map(canonical_smiles))
    if None in train_keys or train_keys & holdout_keys:
        raise ValueError('Invalid training SMILES or canonical molecular leakage.')


def save_example(tables, grid, output, module, source_path, audit, strategy, choices):
    validate_tables(tables, grid)
    baseline = module / 'outputs/carpentry'
    validate_frozen_evaluation(module, baseline, grid)
    tables['train'].to_parquet(output / 'IR_train.parquet', index=False)
    hashes = {}
    for name in ('validation', 'test'):
        original, destination = baseline / f'IR_{name}.parquet', output / f'IR_{name}.parquet'
        pd.testing.assert_frame_equal(tables[name], pd.read_parquet(original))
        shutil.copyfile(original, destination)
        hashes[name] = sha256_file(original)
        assert sha256_file(destination) == hashes[name]
    manifest = json.loads((baseline / 'data_manifest.json').read_text(encoding='utf-8'))
    baseline_rows = manifest['rows'].copy()
    manifest.update(example=strategy, source_file=str(source_path.relative_to(module)),
                    frozen_holdout_sha256=hashes, baseline_directory='outputs/carpentry',
                    carpentry={**manifest['carpentry'], 'example': strategy, **choices},
                    baseline_rows=baseline_rows,
                    rows={'input': baseline_rows['input'], 'baseline_train': baseline_rows['train'],
                          **{n: len(f) for n, f in tables.items()}})
    (output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    pd.DataFrame(audit, columns=['NIST_molecule', 'reason', 'action']).to_csv(output / 'cleaning_audit.csv', index=False)
    summary = pd.DataFrame({name: frame[LABELS].sum() for name, frame in tables.items()})
    summary.to_csv(output / 'label_counts.csv')
    validate_tables({n: pd.read_parquet(output / f'IR_{n}.parquet') for n in tables}, grid)
    validate_frozen_evaluation(module, output, grid)
    print('PASS: reload, finite arrays, binary labels, positive classes, no training/holdout molecule overlap.')
    print('PASS: validation/test Parquets are byte-identical to the frozen reference.')
    print('Rows:', {name: len(frame) for name, frame in tables.items()})
    print('Actions:', pd.Series([a['reason'] for a in audit], dtype='object').value_counts().to_dict())
    print('Saved:', output.relative_to(module))
    return summary
