"""Build independent HTEM algorithms plus a short, provenance-checked comparison."""
from pathlib import Path
import sys
import textwrap

import nbformat as nbf

MODULE = Path(__file__).resolve().parents[1] / "03_htem_conductivity"


def md(value):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


BOOTSTRAP = '''
from pathlib import Path
import sys, json
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import joblib
HERE = None
for parent in [Path.cwd(), *Path.cwd().parents]:
    for candidate in [parent, parent / '03_htem_conductivity']:
        if (candidate / 'lib/htem_experiments.py').exists():
            HERE = candidate
            break
    if HERE is not None: break
if HERE is None:
    raise FileNotFoundError('Open from the workshop, HTEM module, or its examples folder.')
sys.path.insert(0, str(HERE / 'lib'))
import htem_workshop as htem
import htem_experiments as exp
ctx = exp.load_context(HERE)  # Validate and reuse notebook 01; NEVER redo preparation/splitting.
print('Shared frozen feature set:', ctx.manifest['frozen_feature_selection']['candidate'])
SEED = ctx.manifest['seed']
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
'''


def model_cells(experiment_id, title, explanation, prompt, fit_code, neural=False, graph=False):
    cells = [md(f'''
    # {title}

    Predict local **thickness and conductivity maps** from a process recipe and planned substrate XY.
    {explanation}

    **Prediction unit:** one feature row per location (recipe + physical X/Y + local temperature),
    yielding thickness and conductivity at that location. Ridge/KRR/MLP/Random Forest predict independently,
    whether queried alone or in a batch. Save point predictions with their identity/XY first; only then
    pivot them into a grid for visualization, without averaging, interpolation, or smoothing. The GNN
    also predicts per location, but is explicitly neighbor-aware and requires the full planned grid.

    **Fixed comparison contract:** run notebook 01 once, then all five algorithm notebooks without
    rerunning data preparation. Reuse the same material-stratified whole-deposition/identical-recipe
    splits, features, train-only transforms, two log targets, and eligible labels. No spectra, measured
    composition/properties, IDs, or label-availability flags enter the inputs. Whole-grid identity stays
    intact. Models save separate outputs; no algorithm can overwrite another's run.

    **Shared chemistry inputs:** source-formula descriptors (electronegativity, atomic mass/size,
    ionization energy, elemental fractions, and independent reference band-gap statistics) replace
    source-name one-hot columns. Recipe source/power slots remain paired. Reference gaps are not
    measured film properties or certified target-phase values; missing coverage is explicit.
    Power-weighted descriptors are not inferred deposited-film stoichiometry. See notebook 01 for
    references, coverage, and training-only imputation. No architecture adds its own features.

    **Frozen instructor-selected features:** notebook 01 uses the offline specification selected by
    an instructor-only grouped training/validation comparison. Candidate additions include curated
    composition descriptors, published MEGNet composition vectors, real frozen ALIGNN reference-crystal
    vectors and compact reference DOS. Only selected blocks enter the matrices. Reference crystals/DOS
    are not verified source phases or measured film information. No architecture downloads an encoder,
    selects features, fits PCA, or changes preprocessing. All five models read the same checksummed
    matrices. Your experiment changes the ML algorithm/architecture, not the data preparation.

    **Expanded observations:** a position needs at least one observed property, not both. Ridge/KRR/forests
    fit each output on its observed labels; neural models use a per-property masked MSE with equal
    property weighting. Missing labels remain NaN, never fabricated. Each node also has a local
    temperature from the source `absolute_temp_c` profile. Inference requires a planned/calibrated
    temperature map; missing temperature is explicitly imputed, not assumed equal to initial temperature.
    Original split assignments are frozen; new groups are material-balanced around them. The summary
    also reports the original paired test points separately for an honest before/after comparison.

    Workshop cycle: **baseline → hypothesis → one model change → validation → final reporting**.
    The default 02 stays linear Ridge regardless of which model ranks first in notebook 03.
    '''), code(BOOTSTRAP + f'''
EXPERIMENT_ID = {experiment_id!r}
RESULTS = HERE / 'outputs/htem_architectures' / EXPERIMENT_ID
RESULTS.mkdir(parents=True, exist_ok=True)
X_train, y_train = ctx.data['train']['X'], ctx.data['train']['y']
X_val, y_val = ctx.data['validation']['X'], ctx.data['validation']['y']
print('Same prepared data:', ctx.fingerprint['prepared_manifest_sha256'][:12])
display(pd.DataFrame([{{'split': s, 'labeled_points': len(ctx.data[s]['X']),
                       'depositions': len(np.unique(ctx.data[s]['library_id'])),
                       'thickness_labels': int(ctx.data[s]['y_mask'][:, 0].sum()),
                       'conductivity_labels': int(ctx.data[s]['y_mask'][:, 1].sum())}} for s in htem.SPLITS]))
print(f'{{X_train.shape[1]}} encoded inputs → 2 standardized log targets.')
'''), md(f'''
    ## 1. Fit the model; use validation, never test, for choices

    **Hypothesis prompt:** “{prompt} Keep the supplied data contract unchanged. Explain each change,
    preserve reproducibility, and compare validation error before final test reporting.”
    ''')]
    if graph:
        cells += [md('''
        Each deposition is an independent four-neighbor grid graph. Node inputs are recipe + XY;
        edges encode signed dx, signed dy, and distance (fixed 10 mm scale). Two residual message
        layers aggregate learned neighbor messages, then predict two values **per location**, not a
        pooled substrate average. Full planned grids include unlabeled nodes; only eligible labels
        contribute to loss. The mask is not an input. No edges cross depositions in a batch.

        This edge-aware mean-neighbor network is inspired by
        [GraphSAGE](https://arxiv.org/abs/1706.02216), not a paper reproduction. Geometry provides an
        inductive bias, not proof that sampling sites interact physically or a guarantee of improvement.
        '''), code('''
        import htem_graph as gnn
        graphs = exp.ensure_graphs(ctx)
        display(ctx.graph_report.groupby('split')[['nodes', 'labeled_nodes', 'undirected_edges']].sum())
        sample = graphs['train'][0]
        fig = gnn.plot_grid_graph(sample.grid, f'Training deposition {sample.library_id}: planned connectivity')
        fig.savefig(RESULTS / 'substrate_connectivity.png', bbox_inches='tight')
        plt.show()
        ''')]
    cells.append(code(fit_code))
    if neural:
        cells += [md('''
        ## 2. Inspect convergence and the restored checkpoint

        AdamW, gradient clipping, cosine LR decay, and validation patience are configured above.
        We restore **minimum validation MSE**; the epoch cap is a budget, not proof of convergence.
        Change one architecture/optimization setting at a time. Plotting is separate from fitting.
        '''), code('''
        print(training_note)
        fig = exp.plot_history(history, training_note)
        fig.savefig(RESULTS / 'training_progress.png', bbox_inches='tight')
        plt.show()
        ''')]
    elif experiment_id == 'random_forest':
        cells.append(md('''
        Random Forest averages bootstrapped decision trees and can learn nonlinear recipe-by-position
        interactions. Tree predictions are piecewise rather than inherently smooth gradients; improvement
        is not guaranteed. The same StandardScaler/log-target transforms are retained for a fair comparison,
        although trees do not require scaled inputs. Each property uses only its observed training labels.
        There are no neural epochs, learning rate, or early-stopping loss curves. This is one fixed
        configuration, not a hyperparameter search. Use the grouped validation set for subsequent changes;
        ordinary row-level out-of-bag scores would not be a whole-deposition holdout.
        See [scikit-learn RandomForestRegressor](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
        '''))
    else:
        cells.append(md('''
        This model fits a regularized regression objective, not epoch-based neural training. KRR solves
        the full linear system to checked residual tolerance. There is no invented
        loss-vs-epoch curve or early stopping. Any candidate selection uses validation only; the selected
        model remains fitted on training, not training plus validation. Search choices are recorded.
        '''))
    cells += [md('''
    ## Evaluate the same train, validation, and test points

    **Analysis prompt:** “Report each property's error and the training/validation gap. Invert the
    existing transforms, preserve library/position/XY, and show parity plots for all three splits.
    Do not tune using test results. Be honest about weak conductivity predictions.”

    Conductivity log error is in decades: 1 decade means a factor of 10. Raw conductivity metrics are
    also saved, but extreme values dominate them. These observational models are not causal simulators.
    '''), code('''
    RESULTS, metrics, run = exp.evaluate_and_save(
        ctx, model, EXPERIMENT_ID, configuration, fit_seconds,
        history=history, training_note=training_note, trials=trials)
    print(f"Validation score (mean standardized RMSE): {run['validation_score']:.4f}")
    print('Standardized target MSE:', {s: round(v, 4) for s, v in run['standardized_MSE'].items()})
    display(metrics[metrics.target.ne('conductivity_S_cm')].round(4))
    predictions = pd.read_csv(RESULTS / 'predictions.csv', dtype={'sample_id': str})
    fig = exp.plot_parity(predictions)
    fig.savefig(RESULTS / 'parity_all_splits.png', bbox_inches='tight')
    plt.show()
    '''), md('''
    ## Compare maps for the same held-out deposition

    All algorithm notebooks select the same test library by eligible point count, not favorable error.
    Predict its **full planned grid**; missing measured labels remain blank. No interpolation, fixed-size
    reshape, or averaging across positions/depositions is allowed. Color limits match actual/predicted.
    '''), code('''
    full_map = pd.read_csv(RESULTS / 'heldout_map.csv')
    MAP_LIBRARY_ID = int(run['map_library_id'])
    fig = htem.plot_substrate_maps(full_map, MAP_LIBRARY_ID)
    fig.suptitle(f"{run['name']}: held-out deposition {MAP_LIBRARY_ID}")
    fig.tight_layout()
    fig.savefig(RESULTS / 'heldout_maps.png', bbox_inches='tight')
    plt.show()
    '''), md('''
    ## Predict a recipe on a requested grid using the saved model

    Edit `recipe` below, or supply planned `position/x_mm/y_mm` coordinates in the HTEM substrate frame.
    Do not pass measured outputs or stage/focus `xyz_mm`. Use a **full planned grid** for graph inference;
    changing nodes changes neighborhoods. For grids missing entire coordinate levels, specify known
    `spacing_mm=(dx, dy)` rather than guessing nominal spacing. Load only trusted local joblib artifacts.
    '''), code('''
    fitted = joblib.load(RESULTS / 'model.joblib')
    def predict_substrate(recipe, grid, spacing_mm=None):
        return exp.predict_substrate(fitted, recipe, grid, spacing_mm)

    reference_grid = full_map[['position', 'x_mm', 'y_mm', 'local_temperature_C']].copy()
    reference_recipe = exp.example_recipe(ctx, MAP_LIBRARY_ID)
    reference_map = predict_substrate(reference_recipe, reference_grid)
    # Validate the unchanged reference BEFORE trying a modified recipe or grid.
    np.testing.assert_allclose(reference_map[['predicted_thickness_um', 'predicted_conductivity_S_cm']],
                               full_map[['predicted_thickness_um', 'predicted_conductivity_S_cm']], rtol=5e-5, atol=1e-7)
    recipe = dict(reference_recipe)
    grid = reference_grid.copy()  # EDIT planned grid here, after the reference check.
    # EDIT a planned temperature profile, for example: grid['local_temperature_C'] = 350
    # Changing initial temperature alone does NOT redefine the specified local profile.
    predicted_map = predict_substrate(recipe, grid)
    display(pd.Series(recipe, name='RECIPE SETTINGS').to_frame())
    display(predicted_map.head())
    fig = htem.plot_substrate_maps(predicted_map)
    fig.savefig(RESULTS / 'requested_recipe_maps.png', bbox_inches='tight')
    plt.show()
    predicted_map.to_csv(RESULTS / 'requested_recipe_map.csv', index=False)
    '''), md('''
    Record your hypothesis, configuration, validation outcome, and next experiment. Run all five
    architecture notebooks before `03_HTEM_Compare_Architectures.ipynb`. The comparison rejects stale
    preparation, different labeled rows/targets, missing runs, and changed result files. Better scores
    on one fixed holdout do not establish statistical superiority; do not repeatedly optimize on test.
    ''')]
    return cells


def build(selected=None):
    selected = set(selected or [])

    def save(name, cells):
        if selected and name not in selected:
            return
        path = MODULE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        notebook = nbf.v4.new_notebook(cells=cells, metadata={
            'kernelspec': {'display_name': 'Python (workshop .venv)', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'},
        })
        nbf.write(notebook, path)
        print('Built', name)

    save('examples/02_HTEM_Process_Emulator_Vibe_Coding.ipynb', model_cells(
        'linear_ridge', '02 · Linear regression: default process emulator',
        'The default is **linear Ridge regression**, `Ridge(alpha=10)`: a linear model with L2 regularization, not kernel ridge and not unregularized OLS.',
        'Establish a simple regularized linear baseline. Expose the L2 penalty, explain the assumptions, and analyze where a nonlinear or graph model might help.',
        '''
        from sklearn.linear_model import Ridge
        from threadpoolctl import threadpool_limits
        ALPHA = 10.0  # Workshop default stays linear; do not auto-replace it with a comparison winner.
        configuration = {'alpha': ALPHA, 'algorithm': 'linear Ridge regression'}
        started = perf_counter()
        with threadpool_limits(limits=4):
            model = exp.fit_observed_regressors(Ridge(alpha=ALPHA), X_train, y_train)
        fit_seconds = perf_counter() - started
        history = trials = None
        training_note = {'solver': 'linear Ridge; no epochs or early stopping'}
        '''))

    save('examples/02a_HTEM_Kernel_Ridge.ipynb', model_cells(
        'kernel_ridge', '02a · Nonlinear RBF kernel ridge regression',
        'Use **exact RBF kernel ridge per observed output**, not Nyström features or subsampling. The full kernel grows quadratically with each property’s training labels. A float64 conjugate-gradient solve checks relative residual ≤1.1e-8; it is a numerical solution to the full KRR system, not a low-rank approximation. See [SciPy CG](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html). Nine alpha/gamma candidates use validation only.',
        'The recipe-to-property relationship may be nonlinear. Try an RBF kernel, select regularization alpha and bandwidth gamma using the fixed validation set, and compare fairly against linear Ridge without adding features.',
        '''
        ALPHAS = [0.1, 1.0, 10.0]
        GAMMAS = [factor / X_train.shape[1] for factor in [0.1, 1.0, 10.0]]
        print(f'Exact kernel: {len(X_train):,} × {len(X_train):,} entries; nine training/validation candidates.')
        started = perf_counter()
        model, trials = exp.fit_kernel_ridge(ctx, ALPHAS, GAMMAS)
        fit_seconds = perf_counter() - started  # Includes ALL search candidates, not just the winner.
        configuration = {'kernel': 'rbf', 'alpha': float(model.alpha), 'gamma': float(model.gamma),
                         'search_alphas': ALPHAS, 'search_gammas': GAMMAS, 'solver_dtype': 'float64'}
        history = None
        training_note = {'selection': 'minimum mean standardized validation RMSE; no test feedback',
                         'solver': 'full exact RBF kernel; checked CG residual; no subsampling or validation refit'}
        display(trials.round(5))
        print('Selected:', configuration['alpha'], configuration['gamma'])
        '''))

    save('examples/02b_HTEM_Feed_Forward_Network.ipynb', model_cells(
        'mlp', '02b · Feed-forward neural process emulator',
        'Predict each location independently with a **64–32 ReLU MLP**. Recipe and absolute XY are inputs; no explicit neighbor connectivity is used.',
        'Try a compact nonlinear MLP. Expose hidden widths, learning rate, weight decay, epoch cap, and validation patience; keep the fixed features, transforms, and split unchanged.',
        '''
        configuration = {'hidden': (64, 32), 'lr': 1e-3, 'weight_decay': 1e-3, 'epochs': 300, 'patience': 40}
        started = perf_counter()
        model, history, training_note = htem.fit_neural_head(X_train, y_train, X_val, y_val, seed=SEED, **configuration)
        fit_seconds = perf_counter() - started
        trials = None
        ''', neural=True))

    save('examples/02c_HTEM_Grid_GNN.ipynb', model_cells(
        'grid_gnn', '02c · Grid-connectivity graph neural process emulator',
        'Use a **two-layer, width-64 residual message-passing GNN** over each deposition grid. It trains on precisely the same eligible labeled locations as the point models.',
        'Local spatial patterns may benefit from neighboring recipe/position representations. Use independent full-grid graphs, geometry-only edges, a masked node loss, and validation-minimum checkpoints; never feed measured neighbor properties into messages.',
        '''
        configuration = {'hidden_dim': 64, 'message_layers': 2, 'dropout': 0.05, 'lr': 5e-4,
                         'weight_decay': 1e-3, 'epochs': 300, 'patience': 40, 'batch_size': 16}
        started = perf_counter()
        model, history, training_note = gnn.fit_graph_model(graphs['train'], graphs['validation'], seed=SEED, **configuration)
        fit_seconds = perf_counter() - started
        trials = None
        ''', neural=True, graph=True))

    save('examples/02d_HTEM_Random_Forest.ipynb', model_cells(
        'random_forest', '02d · Random Forest process emulator',
        'Use a **Random Forest regressor per observed property**, with 300 trees, bounded tree depth, and a fixed seed. Each point is predicted from its recipe, physical X/Y, and local temperature, independently of the other queried points.',
        'Try tree ensembles to learn nonlinear process-by-position interactions. Keep the prepared data and grouped splits unchanged; expose tree count, maximum depth, minimum leaf size, and feature sampling. Compare validation error and spatial maps without tuning on test.',
        '''
        from sklearn.ensemble import RandomForestRegressor
        configuration = {'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 3,
                         'max_features': 0.7, 'bootstrap': True, 'oob_score': False,
                         'random_state': SEED, 'n_jobs': 4}
        print('Fitting one Random Forest per property on its observed training labels...')
        started = perf_counter()
        model = exp.fit_observed_regressors(RandomForestRegressor(**configuration), X_train, y_train)
        fit_seconds = perf_counter() - started
        history = trials = None
        training_note = {'selection': 'one fixed configuration; validation reporting only; no test tuning',
                         'training': 'separate forests on observed labels; no epochs or early stopping'}
        print(f'Fitted {len(model.estimators_)} forests in {fit_seconds:.1f} seconds.')
        '''))

    save('examples/03_HTEM_Compare_Architectures.ipynb', [md('''
    # 03 · Compare HTEM algorithms on identical prepared data

    Run `02` and `examples/02a`, `02b`, `02c`, `02d` first. This short notebook **does not train or alter data**.
    It checks preparation/preprocessing hashes, exact labeled-point identities/targets, result checksums,
    and the held-out map example before assembling results. Missing/stale runs produce a clear error.
    Rank by **validation**, not test. The default workshop 02 remains linear Ridge regardless of rank.
    '''), code(BOOTSTRAP + '''
comparison = exp.collect_comparison(ctx)
COMPARISON = HERE / 'outputs/htem_architectures/comparison'
COMPARISON.mkdir(parents=True, exist_ok=True)
comparison.to_csv(COMPARISON / 'architecture_summary.csv', index=False)
display(comparison[['algorithm', 'validation_score', 'train_scaled_MSE', 'validation_scaled_MSE',
                    'test_scaled_MSE', 'test_thickness_MAE_um', 'test_conductivity_log10_MAE',
                    'test_thickness_n', 'test_conductivity_n', 'fit_search_seconds']].round(4))
previous = pd.read_csv(HERE / 'data/frozen_evaluation/paired_test_metrics.csv')
original = comparison[['experiment_id', 'algorithm', 'original_test_thickness_MAE_um',
                       'original_test_conductivity_log10_MAE']].merge(previous, on='experiment_id', how='left', validate='one_to_one')
display(original.drop(columns='experiment_id').round(4))
original.to_csv(COMPARISON / 'original_test_before_after.csv', index=False)
print('Before/after uses exactly the original 2,108 paired test points. Expanded-test scores above use different counts per property.')
print('Missing previous scores mean no archived pre-expansion run exists for that algorithm, not zero error.')
print('Same data/splits/preprocessing verified:', ctx.fingerprint['prepared_manifest_sha256'][:12])
print('Lowest validation score:', comparison.iloc[0].algorithm, '— default 02 is still linear Ridge.')
'''), md('''
    ## Compare errors, not just a single winner

    Lower validation score, thickness MAE, and conductivity log-MAE are better. Standardized MSE is
    shown for all three splits in the table; physical errors are not interchangeable between properties.
    Conductivity log-MAE is in decades (1 = factor 10). Fit/search seconds include all nine KRR trials,
    versus one supplied NN architecture each; timings are descriptive, not equal search-budget claims.
    '''), code('''
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
for ax, column, title in zip(axes,
        ['validation_score', 'test_thickness_MAE_um', 'test_conductivity_log10_MAE'],
        ['Validation: mean standardized RMSE', 'Test: thickness MAE (um)', 'Test: conductivity log-MAE (decades)']):
    ax.barh(comparison.algorithm, comparison[column], color='#236b8e')
    ax.set(title=title, xlabel='Error (lower is better)')
    ax.invert_yaxis()
fig.tight_layout()
fig.savefig(COMPARISON / 'architecture_errors.png', bbox_inches='tight')
plt.show()
'''), md('''
    ## Test-set parity: thickness and conductivity for every algorithm

    Each row uses the same untouched test locations. The dashed line is perfect prediction.
    Thickness is in µm; conductivity is shown as **log10(S/cm)** because it spans many orders of
    magnitude. All rows share the same limits for each property, and all points—including poor
    predictions—remain visible. This is reporting, not another round of model selection.
    '''), code('''
# Test-set parity: read the prediction artifacts already verified above; never refit.
test_predictions = {
    row.experiment_id: pd.read_csv(
        HERE / 'outputs/htem_architectures' / row.experiment_id / 'predictions.csv',
        dtype={'sample_id': str}).query("split == 'test'")
    for row in comparison.itertuples(index=False)
}
properties = [('thickness_um', 'Thickness (µm)'), ('conductivity_S_cm', 'log10 conductivity (S/cm)')]
limits = []
for target, _ in properties:
    values = np.concatenate([table.loc[table['actual_' + target].notna(),
                             ['actual_' + target, 'predicted_' + target]].to_numpy().ravel()
                             for table in test_predictions.values()])
    if not np.isfinite(values).all() or (target == 'conductivity_S_cm' and (values <= 0).any()):
        raise ValueError('Parity plots require finite predictions and positive conductivity; do not silently omit points.')
    if target == 'conductivity_S_cm':
        values = np.log10(values)
    lo, hi = float(values.min()), float(values.max())
    padding = .04 * (hi - lo) if hi > lo else max(abs(lo) * .04, .01)
    limits.append((lo - padding, hi + padding))

fig, axes = plt.subplots(len(comparison), 2, figsize=(10, 3 * len(comparison)),
                         sharex='col', sharey='col', squeeze=False)
for row_index, result in enumerate(comparison.itertuples(index=False)):
    table = test_predictions[result.experiment_id]
    for column, (target, label) in enumerate(properties):
        known = table['actual_' + target].notna()
        actual, predicted = table.loc[known, 'actual_' + target], table.loc[known, 'predicted_' + target]
        if target == 'conductivity_S_cm':
            actual, predicted = np.log10(actual), np.log10(predicted)
        ax = axes[row_index, column]
        ax.scatter(actual, predicted, s=9, alpha=.3, color='#236b8e', rasterized=True)
        ax.plot(limits[column], limits[column], 'k--', lw=1)
        ax.set(xlim=limits[column], ylim=limits[column], title=f'{result.algorithm} · {label}',
               ylabel='Predicted ' + label, xlabel='Actual ' + label if row_index == len(comparison) - 1 else '')
        ax.set_aspect('equal', adjustable='box')
        ax.text(.04, .95, f'n = {len(actual):,}', transform=ax.transAxes, va='top')
fig.suptitle('Identical test set · parity across HTEM algorithms', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .98))
fig.savefig(COMPARISON / 'test_set_parity.png', bbox_inches='tight')
plt.show()
'''), md('''
    ## Interpretation

    Use validation for the next hypothesis; reserve test for reporting. If nonlinear models do not win,
    say so. Sparse material/recipe coverage, missing process history, and observational selection remain
    limitations. One fixed split and seed do not establish a statistically significant improvement.
    All models preserve deposition/position identity and use the same held-out grid example; their
    individual notebooks contain the maps, parity plots, saved models, and recipe inference functions.
    ''')])


if __name__ == '__main__':
    build(sys.argv[1:])
