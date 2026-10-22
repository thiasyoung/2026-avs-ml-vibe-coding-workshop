"""Execute all IR carpentry examples and verify the unchanged trainer can load them.

Run from the repository: uv run --locked python 02_ir_functional_groups/examples/run_examples.py
Only examples are saved in place; IR 02 is smoke-tested in memory without training.
"""
from pathlib import Path
import json
import sys

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

from ir_example_support import sha256_file
from strategies import STRATEGIES

EXAMPLES = Path(__file__).resolve().parent
MODULE = EXAMPLES.parent


class EnvironmentKernelManager(KernelManager):
    def format_kernel_cmd(self, extra_arguments=None):
        command = super().format_kernel_cmd(extra_arguments)
        command[0] = sys.executable
        return command


def execute(notebook, working_directory):
    NotebookClient(notebook, timeout=600, kernel_name='python3',
                   kernel_manager_class=EnvironmentKernelManager,
                   resources={'metadata': {'path': str(working_directory)}}).execute()


def main():
    hashes = {name: sha256_file(MODULE / f'outputs/carpentry/IR_{name}.parquet')
              for name in ('validation', 'test')}
    smoke_cells = []
    for strategy in STRATEGIES:
        path = EXAMPLES / strategy['notebook']
        print(f'Executing {path.name}', flush=True)
        notebook = nbformat.read(path, as_version=4)
        execute(notebook, EXAMPLES)
        nbformat.write(notebook, path)
        errors = [out for cell in notebook.cells for out in cell.get('outputs', []) if out.output_type == 'error']
        assert not errors, errors
        prefix = path.name[:3]
        output = MODULE / strategy['directory']
        manifest = json.loads((output / 'data_manifest.json').read_text(encoding='utf-8'))
        assert manifest['frozen_holdout_sha256'] == hashes
        for name, digest in hashes.items():
            assert sha256_file(output / f'IR_{name}.parquet') == digest
            assert sha256_file(MODULE / f'outputs/carpentry/IR_{name}.parquet') == digest

        # Exercise the real trainer's entire data loader/scaler on each saved variant.
        trainer = nbformat.read(MODULE / '02_IR_Optimized_Training.ipynb', as_version=4)
        loader = next(cell.source for cell in trainer.cells if cell.cell_type == 'code')
        relative = output.relative_to(MODULE).as_posix()
        loader = loader.replace("CARPENTRY_DIR = Path('outputs/carpentry')", f"CARPENTRY_DIR = Path('{relative}')", 1)
        smoke_cells.append(nbformat.v4.new_code_cell(loader + f"\nprint('PASS IR 02 loader/scaler: {prefix}')"))
        print(f'PASS {prefix}: executed and holdout hashes identical.', flush=True)
        print('  rows:', manifest['rows'], flush=True)
    # Reuse one test kernel for all four real loader/scaler calls; importing PyTorch
    # afresh for each variant is especially slow on Windows/OneDrive installations.
    smoke = nbformat.v4.new_notebook(cells=smoke_cells)
    execute(smoke, MODULE)
    for cell in smoke.cells:
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                for line in output.text.splitlines():
                    if line.startswith('PASS IR 02'):
                        print(line, flush=True)
    print('All four examples executed and saved; no model training was needed for handoff validation.', flush=True)


if __name__ == '__main__':
    main()
