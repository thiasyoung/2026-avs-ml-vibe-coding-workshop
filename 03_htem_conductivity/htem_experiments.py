"""Shared evaluation/inference for HTEM architecture notebooks; never redo prep."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import clone
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
import torch

import htem_graph as gnn
import htem_workshop as htem

EXPERIMENTS = {
    "linear_ridge": ("Linear Ridge (default)", "02_HTEM_Process_Emulator_Vibe_Coding.ipynb"),
    "kernel_ridge": ("RBF Kernel Ridge", "examples/02a_HTEM_Kernel_Ridge.ipynb"),
    "mlp": ("Feed-forward MLP", "examples/02b_HTEM_Feed_Forward_Network.ipynb"),
    "grid_gnn": ("Grid GNN", "examples/02c_HTEM_Grid_GNN.ipynb"),
    "random_forest": ("Random Forest", "examples/02d_HTEM_Random_Forest.ipynb"),
}
PHYSICAL_TARGETS = ["thickness_um", "conductivity_S_cm"]


class ObservedTargetRegressors:
    """One genuine regressor per property, fit only where that label is observed."""
    def __init__(self, estimators):
        self.estimators_ = estimators
        self.alpha = getattr(estimators[0], "alpha", None)
        self.gamma = getattr(estimators[0], "gamma", None)

    def predict(self, X):
        # Bound prediction memory for the larger exact kernel models.
        blocks = []
        for start in range(0, len(X), 256):
            blocks.append(np.column_stack([m.predict(X[start:start + 256]) for m in self.estimators_]))
        return np.vstack(blocks)


def fit_observed_regressors(estimator, X, y):
    models = []
    for i in range(y.shape[1]):
        known = np.isfinite(y[:, i])
        if not known.any():
            raise ValueError("Each training target needs observed labels.")
        models.append(clone(estimator).fit(X[known], y[known, i]))
    return ObservedTargetRegressors(models)


class ExactRBFRegressor(KernelRidge):
    """Full RBF kernel, CG to checked residual tolerance; no low-rank approximation."""
    def fit_kernel(self, X, y, kernel):
        from scipy.sparse.linalg import LinearOperator, cg
        self.X_fit_ = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        operator = LinearOperator(kernel.shape, matvec=lambda v: kernel @ v + self.alpha * v, dtype=np.float64)
        solution, info = cg(operator, y, rtol=1e-8, atol=0, maxiter=4000)
        residual = np.linalg.norm(operator @ solution - y) / max(np.linalg.norm(y), 1e-30)
        if info != 0 or not np.isfinite(solution).all() or residual > 1.1e-8:
            raise RuntimeError(f"Exact KRR solve did not converge: info={info}, relative residual={residual:g}")
        self.dual_coef_ = solution
        self.solver_relative_residual_ = float(residual)
        return self


@dataclass
class Context:
    module: Path
    prepared: Path
    data: dict
    transforms: dict
    manifest: dict
    clean: pd.DataFrame
    grids: pd.DataFrame
    fingerprint: dict
    graphs: dict | None = None
    graph_report: pd.DataFrame | None = None


def load_context(module):
    module = Path(module).resolve()
    prepared = module / "outputs/htem_prepared"
    if not (prepared / "manifest.json").exists():
        # A fresh GitHub checkout can train directly from the instructor's fixed
        # matrices. Notebook 01 remains available to demonstrate/reproduce prep.
        prepared = module / "data/frozen_features"
    data, transforms, manifest = htem.load_prepared(prepared)
    fingerprint = {
        "prepared_manifest_sha256": htem.sha256(prepared / "manifest.json"),
        "prepared_files_sha256": manifest["files_sha256"],
        "features": manifest["recipe_features"], "encoded_features": manifest["encoded_features"],
        "targets": manifest["targets"], "seed": manifest["seed"],
        "evaluation_code_sha256": htem.sha256(__file__),
        "chemistry": manifest["chemistry"],
    }
    return Context(module, prepared, data, transforms, manifest,
                   pd.read_parquet(prepared / "clean_process_table.parquet"),
                   pd.read_parquet(prepared / "substrate_grids.parquet"), fingerprint)


def ensure_graphs(context):
    if context.graphs is None:
        context.graphs, context.graph_report = gnn.build_graph_splits(context.prepared, context.data, context.transforms)
    return context.graphs


def solver_inputs(model, X):
    # Exact dense KRR amplifies float32 kernel roundoff, including batch-size
    # differences. Promote the SAME prepared values, not the preprocessing.
    kernel = isinstance(model, KernelRidge) or (isinstance(model, ObservedTargetRegressors)
             and isinstance(model.estimators_[0], KernelRidge))
    return np.asarray(X, dtype=np.float64) if kernel else X


def predict_scaled(context, model, split):
    X = context.data[split]["X"]
    if isinstance(model, gnn.SubstrateGNN):
        return gnn.predict_labeled(model, ensure_graphs(context)[split], len(X))
    if isinstance(model, torch.nn.Module):
        return htem.neural_predict(model, X)
    with threadpool_limits(limits=4):
        return model.predict(solver_inputs(model, X))


def validation_score(context, model):
    return float(np.sqrt(htem.observed_mse(predict_scaled(context, model, "validation"),
                                         context.data["validation"]["y"])).mean())


def fit_kernel_ridge(context, alphas, gammas):
    """Exact multi-output RBF KRR; select only on the existing validation split.

    No Nyström approximation, subsampling, new encoding, test-set feedback,
    or refit on validation. Fit/search runtime should include every candidate.
    """
    if not alphas or not gammas or any(v <= 0 or not np.isfinite(v) for v in [*alphas, *gammas]):
        raise ValueError("Use nonempty positive finite alpha/gamma grids.")
    training = context.data["train"]
    X_train = np.asarray(training["X"], dtype=np.float64)
    y_train = np.asarray(training["y"], dtype=np.float64)
    best, best_score, trials = None, float("inf"), []
    with threadpool_limits(limits=4):
        for gamma in gammas:
            candidates = {alpha: [] for alpha in alphas}
            for target_index in range(y_train.shape[1]):
                mask = np.isfinite(y_train[:, target_index])
                X, y = X_train[mask], y_train[mask, target_index]
                gib = len(X) ** 2 * 8 / 1024**3
                if gib > 12:
                    raise MemoryError(f"Exact kernel needs {gib:.1f} GiB; choose a larger-memory runner, not silent subsampling.")
                print(f"Full RBF kernel: target {target_index + 1}, {len(X):,} observed labels, {gib:.2f} GiB", flush=True)
                kernel = rbf_kernel(X, gamma=gamma)
                for alpha in alphas:
                    estimator = ExactRBFRegressor(kernel="rbf", alpha=alpha, gamma=gamma).fit_kernel(X, y, kernel)
                    candidates[alpha].append(estimator)
                del kernel
            for alpha in alphas:
                model = ObservedTargetRegressors(candidates[alpha])
                score = validation_score(context, model)
                trials.append({"alpha": alpha, "gamma": gamma, "validation_score": score})
                print(f"KRR {len(trials)}/{len(alphas) * len(gammas)}: alpha={alpha:g}, gamma={gamma:.5g}, val={score:.4f}", flush=True)
                if score < best_score:
                    best, best_score = model, score
    return best, pd.DataFrame(trials).sort_values("validation_score", kind="stable").reset_index(drop=True)


def point_metadata(context, split):
    data = context.data[split]
    table = pd.DataFrame({"sample_id": data["sample_id"].astype(str), "library_id": data["library_id"],
                          "position": data["position"], "x_mm": data["x_mm"], "y_mm": data["y_mm"], "split": split,
                          "legacy_evaluation": data["legacy_evaluation"]})
    values = htem.inverse_predictions(data["y"], context.transforms["target_scaler"])
    for target in PHYSICAL_TARGETS:
        table["actual_" + target] = values[target]
    return table


def grid_prediction(model, X, grid, spacing_mm=None):
    if isinstance(model, gnn.SubstrateGNN):
        return gnn.predict_grid(model, X, grid, spacing_mm)
    if isinstance(model, torch.nn.Module):
        return htem.neural_predict(model, X)
    with threadpool_limits(limits=4):
        return model.predict(solver_inputs(model, X))


def heldout_map(context, model, library_id=None):
    """Use the same metadata-chosen test deposition and full grid for every model."""
    test = context.clean[context.clean.split.eq("test")]
    if library_id is None:
        anchor = test[test.legacy_evaluation]
        selection = anchor if not anchor.empty else test
        library_id = int(selection.groupby("sample_library_id").size().idxmax())
    rows = test[test.sample_library_id.eq(library_id)]
    if rows.empty:
        raise ValueError("Map example must be a held-out test deposition.")
    grid = context.grids[context.grids.sample_library_id.eq(library_id)][[*htem.GEOMETRY_FIELDS, "local_temperature_C"]].reset_index(drop=True)
    gnn.grid_topology(grid[htem.GEOMETRY_FIELDS])
    frame = htem.grid_feature_frame(rows, grid)
    X = context.transforms["preprocessor"].transform(frame).astype("float32")
    values = htem.inverse_predictions(grid_prediction(model, X, grid), context.transforms["target_scaler"])
    result = grid.assign(library_id=library_id)
    for target in PHYSICAL_TARGETS:
        result["predicted_" + target] = values[target]
    actual = point_metadata(context, "test")
    actual = actual[actual.library_id.eq(library_id)]
    result = result.merge(actual[["position", *["actual_" + t for t in PHYSICAL_TARGETS]]],
                          on="position", how="left", validate="one_to_one", sort=False)
    return result


def make_bundle(context, model, experiment_id, config):
    training = context.clean[context.clean.split.eq("train")]
    ranges = {field: (training[field].min(), training[field].max()) for field in htem.NUMERIC_FEATURES}
    return {"version": 1, "experiment_id": experiment_id, "name": EXPERIMENTS[experiment_id][0],
            "model": model, "transforms": context.transforms, "fingerprint": context.fingerprint,
            "configuration": config, "training_ranges": ranges}


def predict_substrate(bundle, recipe, grid, spacing_mm=None):
    """Recipe + requested grid only; return physical-unit predictions, no labels."""
    saved_chemistry = bundle["fingerprint"].get("chemistry", {})
    if any(saved_chemistry.get(key) != value for key, value in htem.chemistry.reference_fingerprint().items()):
        raise ValueError("Source-chemistry reference changed since fitting; rerun preparation and training.")
    metadata, frame = htem.inference_points(recipe, grid)
    unknown = [frame.iloc[0][f"target_{i}"] for i in range(1, 4)
               if frame.iloc[0][f"target_{i}_present"] and not frame.iloc[0][f"target_{i}_formula_known"]]
    if unknown:
        warnings.warn(f"Unsupported source formulas {unknown}: descriptors missing; do not treat this as reliable chemistry extrapolation.")
    transforms = bundle["transforms"]
    if frame[htem.NUMERIC_FEATURES].isna().any(axis=None):
        warnings.warn("Optional recipe settings missing: using training-only imputation.")
    encoder = transforms["preprocessor"].named_transformers_["categorical"]
    for field, vocabulary in zip(htem.CATEGORICAL_FEATURES, encoder.categories_):
        if frame.iloc[0][field] not in vocabulary:
            warnings.warn(f"Unseen {field}: extrapolative prediction.")
    for field, (lo, hi) in bundle["training_ranges"].items():
        if ((frame[field] < lo) | (frame[field] > hi)).any():
            warnings.warn(f"{field} outside training range.")
    X = transforms["preprocessor"].transform(frame).astype("float32")
    values = htem.inverse_predictions(grid_prediction(bundle["model"], X, grid, spacing_mm), transforms["target_scaler"])
    for target in PHYSICAL_TARGETS:
        metadata["predicted_" + target] = values[target]
    return metadata


def example_recipe(context, library_id):
    """Raw recipe for the editable inference example; no measured fields included."""
    raw_file = context.module / "outputs/htem_acquisition/htem_process_raw.parquet"
    if not raw_file.exists():
        raw_file = context.module / "data/htem_spatial_snapshot.parquet"
        if not raw_file.exists() or htem.sha256(raw_file) != context.manifest["raw_sha256"]:
            raise FileNotFoundError("Run notebook 00, or restore the matching distributed raw snapshot for the inference example.")
    source = pd.read_parquet(raw_file)
    return source[source.sample_library_id.eq(library_id)].iloc[0][htem.RECIPE_FIELDS].to_dict()


def evaluate_and_save(context, model, experiment_id, configuration, fit_seconds, history=None, training_note=None, trials=None):
    """Evaluate identical labeled rows and save an independently reloadable run."""
    if experiment_id not in EXPERIMENTS:
        raise ValueError("Unknown experiment ID.")
    # If preparation was regenerated during training, do not certify stale data.
    if htem.sha256(context.prepared / "manifest.json") != context.fingerprint["prepared_manifest_sha256"]:
        raise ValueError("Preparation changed during training; reload and rerun this notebook.")
    output = context.module / "outputs/htem_architectures" / experiment_id
    output.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave a valid old completion marker.
    completion = output / "experiment.json"
    if completion.exists():
        completion.unlink()
    metrics, tables, scaled_mse, scaled_rmse = [], [], {}, {}
    for split in htem.SPLITS:
        estimate = predict_scaled(context, model, split)
        target = context.data[split]["y"]
        if estimate.shape != target.shape or not np.isfinite(estimate).all():
            raise ValueError("Expected two finite predictions per original labeled location.")
        per_target_mse = htem.observed_mse(estimate, target)
        scaled_mse[split] = float(per_target_mse.mean())
        scaled_rmse[split] = float(np.sqrt(per_target_mse).mean())
        table = point_metadata(context, split)
        physical = htem.inverse_predictions(estimate, context.transforms["target_scaler"])
        for name in PHYSICAL_TARGETS:
            if not np.isfinite(physical[name]).all() or not physical[name].gt(0).all():
                raise ValueError("Invalid inverse-transformed property predictions.")
            table["predicted_" + name] = physical[name]
        tables.append(table)
        for name in [*PHYSICAL_TARGETS, "log10_conductivity"]:
            if name == "log10_conductivity":
                a, b = np.log10(table.actual_conductivity_S_cm), np.log10(table.predicted_conductivity_S_cm)
            else:
                a, b = table["actual_" + name], table["predicted_" + name]
            known = np.isfinite(a)
            a, b = a[known], b[known]
            mse = mean_squared_error(a, b)
            metrics.append(dict(split=split, target=name, MAE=float(mean_absolute_error(a, b)),
                                MSE=float(mse), RMSE=float(np.sqrt(mse)), R2=float(r2_score(a, b)), n=int(known.sum())))
    predictions, metrics = pd.concat(tables, ignore_index=True), pd.DataFrame(metrics)
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    legacy_metrics = []
    for split in htem.SPLITS:
        part = predictions[predictions.split.eq(split) & predictions.legacy_evaluation]
        if part.empty:
            continue
        for name in PHYSICAL_TARGETS:
            a, b = part["actual_" + name], part["predicted_" + name]
            if name == "conductivity_S_cm":
                a, b = np.log10(a), np.log10(b)
            legacy_metrics.append(dict(split=split, target=name, n=len(a), MAE=float(mean_absolute_error(a, b)),
                                       RMSE=float(np.sqrt(mean_squared_error(a, b)))))
    pd.DataFrame(legacy_metrics).to_csv(output / "legacy_metrics.csv", index=False)
    full_map = heldout_map(context, model)
    # Full-grid inference must agree with point-array evaluation at labeled nodes.
    checked = full_map.merge(predictions[predictions.split.eq("test")], on=["library_id", "position"],
                             suffixes=("", "_evaluated"), validate="one_to_one")
    for name in PHYSICAL_TARGETS:
        np.testing.assert_allclose(checked["predicted_" + name], checked["predicted_" + name + "_evaluated"], rtol=5e-5, atol=1e-7)
    full_map.to_csv(output / "heldout_map.csv", index=False)
    joblib.dump(make_bundle(context, model, experiment_id, configuration), output / "model.joblib")
    artifacts = ["metrics.csv", "predictions.csv", "heldout_map.csv", "model.joblib", "legacy_metrics.csv"]
    if history is not None:
        history.to_csv(output / "loss_history.csv", index=False)
        artifacts.append("loss_history.csv")
    if trials is not None:
        trials.to_csv(output / "validation_search.csv", index=False)
        artifacts.append("validation_search.csv")
    if isinstance(model, gnn.SubstrateGNN):
        context.graph_report.to_csv(output / "graph_dataset_summary.csv", index=False)
        artifacts.append("graph_dataset_summary.csv")
    summary = {"schema_version": 1, "experiment_id": experiment_id, "name": EXPERIMENTS[experiment_id][0],
               "fingerprint": context.fingerprint, "configuration": configuration, "fit_search_seconds": float(fit_seconds),
               "validation_score": scaled_rmse["validation"], "standardized_MSE": scaled_mse,
               "training_note": training_note, "map_library_id": int(full_map.library_id.iloc[0]),
               "artifacts_sha256": {name: htem.sha256(output / name) for name in artifacts}}
    htem.write_json(completion, summary)
    return output, metrics, summary


def plot_parity(predictions):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for col, split in enumerate(htem.SPLITS):
        part = predictions[predictions.split.eq(split)]
        for row, target in enumerate(PHYSICAL_TARGETS):
            known = part["actual_" + target].notna()
            a, b = part.loc[known, "actual_" + target], part.loc[known, "predicted_" + target]
            if row:
                a, b = np.log10(a), np.log10(b)
            lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
            ax = axes[row, col]
            ax.scatter(a, b, s=7, alpha=.25, color="#236b8e")
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set(title=split.title(), xlabel="Actual " + ("log10 conductivity" if row else "thickness (um)"), ylabel="Predicted")
    fig.tight_layout()
    return fig


def plot_history(history, note):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(history.epoch, history.train_loss, label="Training")
    axes[0].plot(history.epoch, history.validation_loss, "--", label="Validation")
    axes[0].axvline(note["best_epoch"], color="gray", linestyle=":", label="Restored epoch")
    axes[0].set(xlabel="Epoch", ylabel="Standardized log-target MSE")
    axes[0].legend()
    axes[1].plot(history.epoch, history.learning_rate)
    axes[1].set(xlabel="Epoch", ylabel="Learning rate")
    fig.tight_layout()
    return fig


def collect_comparison(context):
    """Fail on missing/stale/mismatched results, never silently compare other data."""
    rows, reference = [], None
    expected = pd.concat([point_metadata(context, s) for s in htem.SPLITS], ignore_index=True)
    identity = ["sample_id", "library_id", "position", "x_mm", "y_mm", "split",
                "actual_thickness_um", "actual_conductivity_S_cm", "legacy_evaluation"]
    for experiment_id, (name, notebook) in EXPERIMENTS.items():
        output = context.module / "outputs/htem_architectures" / experiment_id
        marker = output / "experiment.json"
        if not marker.exists():
            raise FileNotFoundError(f"Missing complete {name} results. Run {notebook} before notebook 03.")
        summary = json.loads(marker.read_text(encoding="utf-8"))
        if summary.get("schema_version") != 1 or summary.get("experiment_id") != experiment_id:
            raise ValueError(f"Incorrect experiment metadata for {name}; rerun its notebook.")
        if summary["fingerprint"] != context.fingerprint:
            raise ValueError(f"Data/preprocessing/evaluation fingerprint mismatch for {name}; rerun its notebook.")
        for filename, digest in summary["artifacts_sha256"].items():
            if Path(filename).name != filename or not (output / filename).exists() or htem.sha256(output / filename) != digest:
                raise ValueError(f"Missing/changed {name} artifact {filename}; rerun its notebook.")
        predictions = pd.read_csv(output / "predictions.csv", dtype={"sample_id": str})
        try:
            pd.testing.assert_frame_equal(predictions[identity], expected[identity], check_dtype=False, rtol=1e-6, atol=1e-8)
        except AssertionError as exc:
            raise ValueError(f"Different labeled point identities/actual targets for {name}; rerun its notebook.") from exc
        if reference is None:
            reference = summary["map_library_id"]
        if summary["map_library_id"] != reference:
            raise ValueError("Models used different held-out map examples.")
        metric = pd.read_csv(output / "metrics.csv").query("split == 'test'").set_index("target")
        legacy = pd.read_csv(output / "legacy_metrics.csv") if context.manifest["frozen_evaluation"]["enabled"] else None
        rows.append({"experiment_id": experiment_id, "algorithm": name, "validation_score": summary["validation_score"],
                     **{s + "_scaled_MSE": summary["standardized_MSE"][s] for s in htem.SPLITS},
                     "test_thickness_MAE_um": metric.loc["thickness_um", "MAE"],
                     "test_conductivity_log10_MAE": metric.loc["log10_conductivity", "MAE"],
                     "test_conductivity_R2": metric.loc["conductivity_S_cm", "R2"],
                     "test_thickness_n": int(metric.loc["thickness_um", "n"]),
                     "test_conductivity_n": int(metric.loc["conductivity_S_cm", "n"]),
                     **({"original_test_thickness_MAE_um": float(legacy.query("split == 'test' and target == 'thickness_um'").MAE.iloc[0]),
                         "original_test_conductivity_log10_MAE": float(legacy.query("split == 'test' and target == 'conductivity_S_cm'").MAE.iloc[0])} if legacy is not None else {}),
                     "fit_search_seconds": summary["fit_search_seconds"],
                     "configuration": json.dumps(summary["configuration"]), "map_library_id": reference})
    comparison = pd.DataFrame(rows).sort_values("validation_score", kind="stable").reset_index(drop=True)
    comparison.insert(0, "validation_rank", np.arange(1, len(comparison) + 1))
    return comparison
