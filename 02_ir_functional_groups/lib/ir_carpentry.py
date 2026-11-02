"""Native-spectrum carpentry and an immutable evaluation contract for IR 01/02.

No downloads or file writes occur on import. All scientific choices are explicit
arguments; notebook 00 only acquires native records.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from rdkit import Chem

LABELS = ['Vinylic', 'Arene', 'Alcohol', 'Ether', 'Amine', 'Nitrile',
          'Aldehyde', 'Ketone', 'CarboxylAcid', 'Ester', 'Amide', 'AcidChloride']
# Explicit workshop bounds, retained from the existing benchmark, not fitted to test data.
DEFAULT_GRID = np.linspace(600.88402, 3846.0, 500)
OUTPUT_COLUMNS = ['NIST_molecule', 'CAS_NO', 'Name', 'molform', 'SMILES',
                  'wavenumber', 'IR', *LABELS]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_smiles(value):
    molecule = Chem.MolFromSmiles(value) if isinstance(value, str) and value.strip() else None
    return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule is not None else None


def load_native(module):
    path = Path(module) / 'outputs/nist_acquisition/ir_raw_checkpoint.parquet'
    if not path.exists():
        raise FileNotFoundError('Run IR 00 to acquire native spectra, or restore ir_raw_checkpoint.parquet. '
                                'Already normalized rehearsal data cannot demonstrate native-unit/bounds carpentry.')
    source = pd.read_parquet(path)
    required = {'NIST_molecule', 'CAS_NO', 'SMILES', 'x_raw', 'y_raw', 'xunits', 'yunits', *LABELS}
    if not required.issubset(source.columns):
        raise ValueError(f'Native checkpoint missing columns: {sorted(required - set(source.columns))}')
    return source, path


def native_arrays(row, signal_policy='absorbance_only'):
    """Convert explicit native units; reject ambiguous units and invalid values."""
    if signal_policy not in {'absorbance_only', 'convert_transmittance', 'as_recorded'}:
        raise ValueError(f'Unknown signal policy: {signal_policy}')
    x, y = np.asarray(row.x_raw, dtype=float), np.asarray(row.y_raw, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError('invalid_array_shape')
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('non_finite_array')
    xu, yu = str(row.xunits).strip().upper(), str(row.yunits).strip().upper()
    actions = []
    if signal_policy == 'as_recorded':
        # Deliberately naive control: assumes native coordinates/signals are comparable.
        # No clipping, inversion or artificial corruption; simply uses recorded arrays.
        actions.append('as_recorded_no_unit_handling')
    elif xu in {'MICROMETERS', 'MICROMETRES', 'UM'}:
        if np.any(x <= 0):
            raise ValueError('nonpositive_wavelength')
        x = 1e4 / x
        actions.append('micrometers_to_wavenumber')
    elif xu not in {'1/CM', 'CM-1', 'CM^-1'}:
        raise ValueError('unsupported_coordinate_units')
    if signal_policy == 'as_recorded':
        pass
    elif yu in {'TRANSMITTANCE', '% TRANSMITTANCE', 'PERCENT TRANSMITTANCE'}:
        if signal_policy == 'absorbance_only':
            raise ValueError('transmittance_excluded_by_policy')
        # Plain TRANSMITTANCE means fraction; never guess percent from amplitude.
        fraction = y / 100 if yu != 'TRANSMITTANCE' else y
        if np.any((fraction <= 0) | (fraction > 1)):
            raise ValueError('transmittance_outside_explicit_unit_range')
        y = -np.log10(fraction)
        actions.append('transmittance_to_absorbance')
    elif yu != 'ABSORBANCE':
        raise ValueError('unsupported_signal_units')
    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    unique, inverse = np.unique(x, return_inverse=True)
    if len(unique) < 2:
        raise ValueError('insufficient_unique_coordinates')
    return unique, np.bincount(inverse, weights=y) / np.bincount(inverse), actions


def bin_averages(x, y, centers):
    """Exact bin integrals of the piecewise-linear measured curve, within coverage."""
    edges = np.r_[centers[0], (centers[:-1] + centers[1:]) / 2, centers[-1]]
    edges = np.clip(edges, x[0], x[-1])
    knots = np.unique(np.r_[x, edges])
    heights = np.interp(knots, x, y)
    integral = np.r_[0., np.cumsum(np.diff(knots) * (heights[:-1] + heights[1:]) / 2)]
    widths = np.diff(edges)
    return np.divide(np.diff(np.interp(edges, knots, integral)), widths,
                     out=np.zeros(len(centers)), where=widths > 0)


def regrid(x, y, grid, bounds_policy='require_full_coverage', minimum_coverage=0.8,
           resampling='linear'):
    """Crop/interpolate within coverage, optionally pad normalized absorption with zero.

    Padding is a declared missing-region assumption, NOT a measured observation.
    Normalize measured values only, then pad, so padding cannot change the scale.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError('invalid_target_grid')
    if bounds_policy not in {'require_full_coverage', 'zero_pad'}:
        raise ValueError(f'Unknown bounds policy: {bounds_policy}')
    if resampling not in {'linear', 'bin_average', 'bin_average_then_upsample'}:
        raise ValueError(f'Unknown resampling policy: {resampling}')
    if not 0 < minimum_coverage <= 1:
        raise ValueError('minimum_coverage must be in (0, 1]')
    covered = (grid >= x[0]) & (grid <= x[-1])
    if not covered.all() and bounds_policy == 'require_full_coverage':
        raise ValueError('incomplete_measured_coverage')
    if covered.sum() < 2 or covered.mean() < minimum_coverage:
        raise ValueError('insufficient_measured_coverage')
    measured = np.interp(grid[covered], x, y)
    source_points = np.count_nonzero((x >= grid[covered][0]) & (x <= grid[covered][-1]))
    if resampling == 'bin_average' and source_points > covered.sum():
        # Average the piecewise-linear native curve over each target bin when
        # downsampling, instead of sampling just one point. Sparse data still
        # use interpolation; upsampling creates no new experimental resolution.
        measured = bin_averages(x, y, grid)[covered]
    elif resampling == 'bin_average_then_upsample':
        # Deliberately test a lower-resolution representation; restore dimensions,
        # not experimental information. Intermediate bounds are measured, so no
        # boundary extrapolation is hidden in the upsampling step.
        coarse = np.linspace(grid[covered][0], grid[covered][-1], max(2, len(grid) // 2))
        coarse_signal = bin_averages(x, y, coarse) if source_points > len(coarse) else np.interp(coarse, x, y)
        measured = np.interp(grid[covered], coarse, coarse_signal)
    span = np.ptp(measured)
    if span <= 1e-12:
        raise ValueError('flat_spectrum')
    values = np.zeros(len(grid), dtype=float)
    values[covered] = (measured - measured.min()) / span
    return values, int((~covered).sum())


def prepare_training(source, grid, signal_policy='absorbance_only',
                     bounds_policy='require_full_coverage', minimum_coverage=0.8,
                     resampling='linear'):
    rows, audit = [], []
    # Repeated observations are a separate exercise; only exact acquisition IDs deduplicate here.
    for _, row in source.drop_duplicates('NIST_molecule').iterrows():
        try:
            if pd.isna(row.CAS_NO) or canonical_smiles(row.SMILES) is None:
                raise ValueError('invalid_identity')
            if not row[LABELS].isin([0, 1]).all():
                raise ValueError('invalid_category')
            x, y, actions = native_arrays(row, signal_policy)
            values, padded = regrid(x, y, grid, bounds_policy, minimum_coverage, resampling)
            covered = (np.asarray(grid) >= x[0]) & (np.asarray(grid) <= x[-1])
            inside = np.count_nonzero((x >= np.asarray(grid)[covered][0]) & (x <= np.asarray(grid)[covered][-1]))
            direction = 'downsampled' if inside > covered.sum() else ('upsampled' if inside < covered.sum() else 'equal_count')
            method = 'bin_average' if resampling == 'bin_average' and direction == 'downsampled' else 'linear'
            resampling_action = f'{direction}_{method}'
            if resampling == 'bin_average_then_upsample':
                first = 'downsampled_bin_average' if inside > max(2, len(grid) // 2) else 'interpolated'
                resampling_action = f'{first}_to_{max(2, len(grid)//2)}_then_upsampled_linear'
            record = {key: value for key, value in row.items() if key not in {'x_raw', 'y_raw'}}
            record.update(wavenumber=np.asarray(grid).tolist(), IR=values.tolist(),
                          padded_points=padded, measured_fraction=1 - padded / len(grid),
                          source_points=int(inside), resampling_action=resampling_action)
            rows.append(record)
            audit.append(dict(NIST_molecule=row.NIST_molecule, action='kept',
                              reason=';'.join(actions + (['zero_padded'] if padded else ['cropped_and_interpolated'])),
                              padded_points=padded, resampling_action=resampling_action))
        except (TypeError, ValueError) as error:
            audit.append(dict(NIST_molecule=row.NIST_molecule, action='rejected', reason=str(error), padded_points=0))
    if not rows:
        raise ValueError('No spectra survived the selected carpentry policies.')
    return pd.DataFrame(rows).reset_index(drop=True), pd.DataFrame(audit)


def training_pool(source, holdouts):
    """Exclude every native observation of any held-out molecule BEFORE carpentry."""
    held = pd.concat(list(holdouts.values()), ignore_index=True)
    canonical = source.SMILES.map(canonical_smiles)
    excluded = canonical.isin(set(held.SMILES.map(canonical_smiles).dropna()))
    for identifier in ('NIST_molecule', 'CAS_NO'):
        excluded |= source[identifier].isin(set(held[identifier].dropna()))
    return source.loc[~excluded].copy(), int(excluded.sum())


def ensure_frozen_evaluation(module, source):
    """Migrate existing heldouts byte-for-byte, otherwise create ONCE by molecule.

    Reruns never repartition or change holdout spectra. Existing legacy benchmarks
    remain legacy benchmarks; no new spectra are silently added to their test sets.
    """
    module = Path(module)
    directory = module / 'data/frozen_evaluation'
    contract_file = directory / 'evaluation_contract.json'
    if not contract_file.exists():
        baseline = module / 'outputs/carpentry'
        directory.mkdir(parents=True, exist_ok=True)
        if all((baseline / f'IR_{s}.parquet').exists() for s in ('validation', 'test')):
            origin = 'Existing workshop holdouts preserved byte-for-byte'
            for name in ('validation', 'test'):
                shutil.copyfile(baseline / f'IR_{name}.parquet', directory / f'IR_{name}.parquet')
            grid = np.asarray(pd.read_parquet(directory / 'IR_test.parquet').wavenumber.iloc[0], dtype=float)
        else:
            # One explicit reference preparation defines the task, not a participant's variant.
            grid = DEFAULT_GRID.copy()
            eligible, _ = prepare_training(source, grid)
            eligible['_canonical'] = eligible.SMILES.map(canonical_smiles)
            groups = eligible.groupby('_canonical', sort=True)[LABELS].max().reset_index()
            if (groups[LABELS].sum() < 3).any():
                raise ValueError('Not enough distinct positive molecules: acquire a larger native dataset in IR 00.')
            split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.30, random_state=42)
            _, held_indices = next(split.split(groups, groups[LABELS].to_numpy()))
            held = groups.iloc[held_indices].reset_index(drop=True)
            split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.5, random_state=42)
            vi, ti = next(split.split(held, held[LABELS].to_numpy()))
            for name, indices in (('validation', vi), ('test', ti)):
                frame = eligible.loc[eligible._canonical.isin(held.iloc[indices]._canonical)].copy()
                frame[OUTPUT_COLUMNS].to_parquet(directory / f'IR_{name}.parquet', index=False)
            origin = 'One-time canonical-molecule-grouped 70/15/15 reference, seed 42'
        contract = dict(version=1, origin=origin, labels=LABELS, wavenumber=grid.tolist(),
                        reference_signal='per-spectrum min-max normalized absorbance',
                        reference_bounds='require_full_coverage',
                        sha256={s: sha256_file(directory / f'IR_{s}.parquet') for s in ('validation', 'test')})
        contract_file.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    contract = json.loads(contract_file.read_text(encoding='utf-8'))
    validate_frozen_evaluation(module, directory, contract['wavenumber'])
    tables = {s: pd.read_parquet(directory / f'IR_{s}.parquet') for s in ('validation', 'test')}
    return tables, np.asarray(contract['wavenumber'], dtype=float), contract


def validate_frozen_evaluation(module, dataset_directory, grid):
    directory = Path(module) / 'data/frozen_evaluation'
    contract_file = directory / 'evaluation_contract.json'
    if not contract_file.is_file():
        raise FileNotFoundError('Run updated IR 01 first to establish the frozen evaluation contract.')
    contract = json.loads(contract_file.read_text(encoding='utf-8'))
    if contract['labels'] != LABELS or not np.array_equal(np.asarray(grid, dtype=np.float32),
                                                       np.asarray(contract['wavenumber'], dtype=np.float32)):
        raise ValueError('Dataset changes the frozen evaluation grid or labels. Use the reference task unchanged.')
    for name, expected in contract['sha256'].items():
        for path in (directory / f'IR_{name}.parquet', Path(dataset_directory) / f'IR_{name}.parquet'):
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f'{name} differs from the frozen evaluation reference: {path}. Rerun IR 01; never resplit per attempt.')
    return contract


def save_carpentry(module, output, train, source_file, choices, audit, input_rows):
    module, output = Path(module), Path(output)
    directory = module / 'data/frozen_evaluation'
    contract = json.loads((directory / 'evaluation_contract.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    # Common prepared training schema for every strategy. Audit metadata are
    # retained, but native arrays/units and anomaly diagnostics are not features.
    columns = OUTPUT_COLUMNS + ['padded_points', 'measured_fraction', 'source_points', 'resampling_action']
    train[columns].to_parquet(output / 'IR_train.parquet', index=False)
    for name in ('validation', 'test'):
        shutil.copyfile(directory / f'IR_{name}.parquet', output / f'IR_{name}.parquet')
    rows = {'input': input_rows, 'train': len(train), **{s: len(pd.read_parquet(output / f'IR_{s}.parquet')) for s in ('validation', 'test')}}
    manifest = dict(source_file=Path(source_file).relative_to(module).as_posix(), seed=42, labels=LABELS,
                    feature_count=len(contract['wavenumber']), wavenumber=contract['wavenumber'],
                    carpentry=choices, rows=rows, frozen_holdout_sha256=contract['sha256'],
                    evaluation_contract='data/frozen_evaluation/evaluation_contract.json',
                    raw_source_sha256=sha256_file(source_file))
    (output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    audit.to_csv(output / 'cleaning_audit.csv', index=False)
    validate_frozen_evaluation(module, output, contract['wavenumber'])
    return manifest
