"""Run the static IR 02 trainer on the naive baseline and four carpentry outputs.

uv run --locked python 02_ir_functional_groups/run_carpentry_comparison.py
Creates ordinary timestamped IR 02 runs and populates IR 02 with its baseline run.
"""
from pathlib import Path
import sys
import copy
import argparse

import nbformat
from nbclient import NotebookClient

MODULE = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE / 'examples'))
from run_examples import EnvironmentKernelManager, execute, main as prepare_examples
from strategies import STRATEGIES


def main(prepare=False):
    if prepare:
        for name in ('00_IR_Get_Data_NISTChemPy_v2.ipynb', '01_IR_Vibe_Coding_Data_Carpentry.ipynb'):
            print(f'Executing preparation: {name}', flush=True)
            path = MODULE / name
            notebook = nbformat.read(path, as_version=4)
            execute(notebook, MODULE)
            nbformat.write(notebook, path)
        prepare_examples()
    # Explicit active list: archived datasets/results must never enter a new batch.
    datasets = [MODULE / 'outputs/carpentry', *[MODULE / s['directory'] for s in STRATEGIES]]
    for dataset in datasets:
        if not (dataset / 'data_manifest.json').is_file():
            raise FileNotFoundError(f'Prepare main IR 01 and all four examples first: missing {dataset}')
    trainer = nbformat.read(MODULE / '02_IR_Optimized_Training.ipynb', as_version=4)
    # Execute every code cell, including the upload control, from the actual trainer.
    pipeline = [cell.source for cell in trainer.cells if cell.cell_type == 'code']
    block_size = len(pipeline)
    cells = [nbformat.v4.new_code_cell('import torch\ntorch.set_num_threads(4)\n%matplotlib inline')]
    for dataset in datasets:
        relative = dataset.relative_to(MODULE).as_posix()
        loader = pipeline[0].replace("CARPENTRY_DIR = Path('outputs/carpentry')", f"CARPENTRY_DIR = Path('{relative}')", 1)
        if loader == pipeline[0] and relative != 'outputs/carpentry':
            raise ValueError('IR 02 dataset-selector format changed; update this runner.')
        cells.extend(nbformat.v4.new_code_cell(source) for source in [loader, *pipeline[1:]])
    notebook = nbformat.v4.new_notebook(cells=cells)
    client = NotebookClient(notebook, timeout=3600, kernel_name='python3',
                            kernel_manager_class=EnvironmentKernelManager,
                            resources={'metadata': {'path': str(MODULE)}})

    def finished(cell, cell_index, **kwargs):
        if cell_index and cell_index % block_size == 0:
            name = datasets[(cell_index - 1) // block_size].name
            print(f'Completed training/evaluation: {name}', flush=True)
            # Save each fully executed trainer as an additional inspectable artifact.
            start = cell_index - block_size + 1
            populated = copy.deepcopy(trainer)
            for original, executed in zip((c for c in populated.cells if c.cell_type == 'code'), notebook.cells[start:cell_index+1]):
                original.source = executed.source
                original.outputs = copy.deepcopy(executed.outputs)
                original.execution_count = executed.execution_count
            nbformat.write(populated, destination.parent / f'02_executed_{name}.ipynb')

    def starting(cell, cell_index, **kwargs):
        if cell_index and (cell_index - 1) % block_size == 0:
            print(f'Starting full static training: {datasets[(cell_index - 1) // block_size].name}', flush=True)

    client.on_cell_executed = finished
    client.on_cell_start = starting
    destination = MODULE / 'outputs/carpentry_comparison/training_executed.ipynb'
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.execute()
    finally:
        # Retain diagnostic cells even when a real training error interrupts a run.
        nbformat.write(notebook, destination)
    # Copy the genuinely executed baseline cells back to IR 02, keeping its source
    # and Markdown intact. Other datasets retain their own timestamped artifacts.
    baseline_cells = notebook.cells[1:1 + block_size]
    for original, executed in zip((c for c in trainer.cells if c.cell_type == 'code'), baseline_cells):
        original.outputs = copy.deepcopy(executed.outputs)
        original.execution_count = executed.execution_count
    if 'widgets' in notebook.metadata:
        trainer.metadata['widgets'] = copy.deepcopy(notebook.metadata['widgets'])
    nbformat.write(trainer, MODULE / '02_IR_Optimized_Training.ipynb')
    print('All five full static IR 02 runs completed; assembling IR 03.', flush=True)
    path = MODULE / '03_IR_Compare_Carpentry.ipynb'
    comparison = nbformat.read(path, as_version=4)
    # TRAIN_ALL stays False here to avoid recursively launching this same runner.
    execute(comparison, MODULE)
    nbformat.write(comparison, path)
    print('IR 03 executed and saved with the current result table and figure.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='First execute IR 00, baseline 01 and all four worked examples.')
    main(prepare=parser.parse_args().prepare)
