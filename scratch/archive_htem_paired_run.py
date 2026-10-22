"""One-time recoverable archive and immutable comparison anchors before expansion."""
from pathlib import Path
import json
import shutil
import pandas as pd

module = Path(__file__).resolve().parents[1] / '03_htem_conductivity'
archive = module / 'archive/paired_labels_2026_10_21'
anchors = module / 'data/frozen_evaluation/paired_split.csv'
if not anchors.exists():
    if archive.exists():
        raise RuntimeError('Archive already exists without anchors; inspect before overwriting anything.')
    archive.mkdir(parents=True)
    for name in ['htem_prepared', 'htem_architectures']:
        shutil.copytree(module / 'outputs' / name, archive / name)
    for path in [*module.glob('*.ipynb'), *module.glob('*.py'), *module.glob('README.md'),
                 *(module / 'examples').glob('*.ipynb')]:
        destination = archive / ('examples' if path.parent.name == 'examples' else 'source')
        destination.mkdir(exist_ok=True)
        shutil.copy2(path, destination / path.name)
    clean = pd.read_parquet(archive / 'htem_prepared/clean_process_table.parquet')
    anchors.parent.mkdir(parents=True, exist_ok=True)
    columns = ['id', 'sample_library_id', 'position', 'x_mm', 'y_mm', 'split',
               'thickness_um', 'conductivity_S_cm']
    clean[columns].to_csv(anchors, index=False)
    print('Archived paired-label outputs and froze', len(clean), 'comparison observations.')
else:
    print('Reusing immutable anchors:', anchors)

baseline = anchors.with_name('paired_test_metrics.csv')
if not baseline.exists():
    records = []
    for folder in (archive / 'htem_architectures').iterdir():
        if not (folder / 'metrics.csv').exists():
            continue
        metrics = pd.read_csv(folder / 'metrics.csv').query("split == 'test'").set_index('target')
        records.append({'experiment_id': folder.name,
                        'previous_thickness_MAE_um': metrics.loc['thickness_um', 'MAE'],
                        'previous_conductivity_log10_MAE': metrics.loc['log10_conductivity', 'MAE']})
    pd.DataFrame(records).to_csv(baseline, index=False)
    print('Saved original paired-test reference metrics without model weights.')
