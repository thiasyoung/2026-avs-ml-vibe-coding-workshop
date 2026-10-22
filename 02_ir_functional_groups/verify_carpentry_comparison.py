"""Verify real completed IR batches; no training or downloads.

Run after run_carpentry_comparison.py --prepare, from any directory.
"""
import ast
import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

MODULE = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE))
sys.path.insert(0, str(MODULE / 'examples'))
from ir_carpentry import LABELS, validate_frozen_evaluation, sha256_file
from ir_example_support import validate_tables
from strategies import STRATEGIES


def check_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    code = [c for c in notebook.cells if c.cell_type == 'code' and c.source.strip()]
    assert code and all(c.execution_count is not None for c in code), f'Unexecuted code: {path}'
    assert not [o for c in code for o in c.outputs if o.output_type == 'error'], f'Failed cells: {path}'
    return notebook


def main():
    torch.set_num_threads(4)
    options = {'01 naive baseline': 'outputs/carpentry', **{s['name']: s['directory'] for s in STRATEGIES}}
    comparison = pd.read_csv(MODULE / 'outputs/carpentry_comparison/comparison.csv').set_index('Method')
    assert list(comparison.index) == list(options), 'Comparison does not contain exactly the five active options.'
    assert comparison.Status.eq('ready').all(), comparison
    trainer = check_notebook(MODULE / '02_IR_Optimized_Training.ipynb')
    source = '\n'.join(c.source for c in trainer.cells if c.cell_type == 'code')
    definition = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'SpectralMLP')
    namespace = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(MODULE), 'exec'), namespace)
    architecture = namespace['SpectralMLP']
    reference_config = reference_schema = None
    for name, directory in options.items():
        data = MODULE / directory
        manifest = json.loads((data / 'data_manifest.json').read_text(encoding='utf-8'))
        grid = np.asarray(manifest['wavenumber'])
        validate_frozen_evaluation(MODULE, data, grid)
        tables = {s: pd.read_parquet(data / f'IR_{s}.parquet') for s in ('train', 'validation', 'test')}
        validate_tables(tables, grid)
        schema = tables['train'].columns.tolist()
        reference_schema = schema if reference_schema is None else reference_schema
        assert schema == reference_schema, 'Training schemas differ.'
        run = MODULE / 'outputs/optimized_training' / comparison.loc[name, 'Run']
        doc = json.loads((run / 'run_documentation.json').read_text(encoding='utf-8'))
        assert doc['data_directory'] == directory
        assert doc['data_files_sha256'] == {s: sha256_file(data / f'IR_{s}.parquet') for s in tables}
        assert manifest['raw_source_sha256'] == sha256_file(MODULE / manifest['source_file'])
        reference_config = doc['training'] if reference_config is None else reference_config
        assert doc['training'] == reference_config, 'Model training settings differ.'
        h = pd.read_csv(run / 'training_history.csv')
        results = doc['results']
        assert len(h) == results['epochs_run']
        np.testing.assert_array_equal(h.epoch, np.arange(1, len(h) + 1))
        assert np.isfinite(h[['train loss', 'validation loss', 'learning rate']]).all().all()
        assert h['learning rate'].between(1e-7 - 1e-12, 1e-5 + 1e-12).all()
        for end in range(20, len(h) + 1):
            losses = h['validation loss'].iloc[end-20:end].to_numpy()
            coordinates = np.arange(20)
            keep = np.ones(20, dtype=bool)
            keep[np.argsort(losses, kind='stable')[-5:]] = False
            raw = np.polyfit(coordinates, losses, 1)[0]
            trimmed = np.polyfit(coordinates[keep], losses[keep], 1)[0]
            np.testing.assert_allclose([raw, trimmed], h[['raw validation slope', 'trimmed validation slope']].iloc[end-1], atol=1e-12)
        eligible = h['validation loss'] > h['train loss']
        selection = h.loc[eligible] if eligible.any() else h
        chosen = selection.loc[selection['validation loss'].idxmin()]
        assert int(chosen.epoch) == results['best_epoch']
        trigger = (h['raw validation slope'] > 0) & (h['trimmed validation slope'] > 0) & eligible
        if results['stop_reason'] == 'robust_validation_slope':
            assert trigger.any() and int(h.loc[trigger, 'epoch'].iloc[0]) == len(h)
        else:
            assert results['stop_reason'] == 'maximum_epochs' and len(h) == doc['training']['max_epochs']
            assert not trigger.any()
        best = torch.load(run / 'best_validation_model.pt', map_location='cpu', weights_only=True)
        bundle = torch.load(run / 'ir_functional_group_classifier.pt', map_location='cpu', weights_only=True)
        assert best['epoch'] == bundle['epoch'] == results['best_epoch']
        for key, tensor in best['state_dict'].items():
            assert torch.equal(tensor, bundle['state_dict'][key]), 'Final weights differ from selected checkpoint.'
        model = architecture(500, 12)
        model.load_state_dict(bundle['state_dict']); model.eval()
        features = np.vstack(tables['test'].IR).astype(np.float32)
        features = ((features - np.asarray(bundle['scaler_mean'])) / np.asarray(bundle['scaler_scale'])).astype(np.float32)
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(torch.from_numpy(features))).numpy()
        saved = pd.read_parquet(run / 'test_predictions.parquet')
        np.testing.assert_array_equal(saved.NIST_molecule, tables['test'].NIST_molecule)
        np.testing.assert_allclose(probabilities, saved[[f'{s}_probability' for s in LABELS]], atol=1e-6)
        predictions = probabilities >= np.asarray(bundle['thresholds'])
        actual = tables['test'][LABELS].to_numpy()
        macro = f1_score(actual, predictions, average='macro', zero_division=0)
        np.testing.assert_allclose(macro, results['test_macro_f1'])
        np.testing.assert_allclose(macro, comparison.loc[name, 'Test macro F1'])
        assert (run / 'model_summary.png').stat().st_size > 1000
        check_notebook(MODULE / f'outputs/carpentry_comparison/02_executed_{data.name}.ipynb')
        print(f"PASS {name}: rows={len(tables['train'])}, selected/run epochs={results['best_epoch']}/{len(h)}, test macro F1={macro:.4f}")
    check_notebook(MODULE / '00_IR_Get_Data_NISTChemPy_v2.ipynb')
    check_notebook(MODULE / '01_IR_Vibe_Coding_Data_Carpentry.ipynb')
    for s in STRATEGIES:
        check_notebook(MODULE / 'examples' / s['notebook'])
    check_notebook(MODULE / '03_IR_Compare_Carpentry.ipynb')
    assert (MODULE / 'outputs/carpentry_comparison/comparison.png').stat().st_size > 1000
    print('PASS all five genuine fits, selected weights, predictions, uniform inputs, fixed holdouts and populated notebooks.')


if __name__ == '__main__':
    main()
