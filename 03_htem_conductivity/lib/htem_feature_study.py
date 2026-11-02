"""Instructor feature ablation: fixed train/validation only; no test loading.

This is not part of student model training. Selection is exploratory evidence on
one existing grouped holdout, not a proof of universal feature superiority.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

import htem_chemistry as chemistry

MODULE = Path(__file__).resolve().parents[1]
ARCHIVE = MODULE / "archive/pre_feature_study_2026_10_21"
STUDY = MODULE / "outputs/htem_feature_study"
REFERENCE = MODULE / "data/htem_feature_candidates.json"
# Declared before any candidate is scored. Same process settings, source slots,
# target transforms, labels, splits and estimator configurations for every row.
CANDIDATES = {
    "existing_no_dos": [],
    "existing_dos": ["dos"],
    "curated": ["curated"],
    "curated_composition": ["curated", "composition"],
    "curated_crystal": ["curated", "crystal"],
    "curated_both": ["curated", "composition", "crystal"],
    "curated_composition_dos": ["curated", "composition", "dos"],
    "curated_crystal_dos": ["curated", "crystal", "dos"],
    "curated_both_dos": ["curated", "composition", "crystal", "dos"],
}
ESTIMATORS = {
    "ridge": {"alpha": 10.},
    "forest": {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 3,
               "max_features": .7, "random_state": 42, "n_jobs": 4},
}
TARGETS = ["log10_thickness_um", "log10_conductivity_S_cm"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_selection_data(archive=ARCHIVE):
    """Read ONLY original training/validation labels; no test artifact access."""
    archive = Path(archive)
    prepared = archive / "htem_prepared"
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    clean = pd.read_parquet(prepared / "clean_process_table.parquet",
                            filters=[("split", "in", ["train", "validation"])])
    if not set(clean.split).issubset({"train", "validation"}):
        raise ValueError("Test rows are forbidden during feature selection.")
    if set(clean.loc[clean.split.eq("train"), "leakage_group"]) & set(clean.loc[clean.split.eq("validation"), "leakage_group"]):
        raise ValueError("Recipe group leakage in feature study.")
    arrays = {s: dict(np.load(prepared / f"{s}.npz", allow_pickle=False)) for s in ["train", "validation"]}
    for split, values in arrays.items():
        part = clean[clean.split.eq(split)]
        np.testing.assert_array_equal(part.id.astype(str), values["sample_id"])
    return clean, arrays, manifest


def build_source_table(reference, training_formulas, components=16):
    """Fit crystal PCA on unique ACTIVE TRAINING compositions only, not rows.

    Reference embeddings are external; PCA fitting is not allowed to use the
    identities/frequencies of validation/test-only source compositions.
    """
    training_keys = sorted({chemistry.composition_key(f) for f in training_formulas if f != "__NONE__"})
    available = [key for key in training_keys if reference["compositions"].get(key, {}).get("crystal_embedding")]
    if len(available) < 2:
        raise ValueError("Need at least two training compositions with real crystal embeddings.")
    matrix = np.array([reference["compositions"][k]["crystal_embedding"]["mean"] for k in available])
    pca = PCA(n_components=min(components, len(matrix) - 1, matrix.shape[1]), svd_solver="full").fit(matrix)
    result = {}
    curated_fields = sorted(next(iter(reference["compositions"].values()))["curated"])
    blocks = {
        "curated": ["curated_known", *curated_fields],
        "composition": ["composition_embedding_known", *[f"megnet_mean_{i:02d}" for i in range(16)]],
        "crystal": ["crystal_embedding_known", "crystal_reference_mean_coordinate_std",
                    *[f"alignn_pc_{i:02d}" for i in range(pca.n_components_)]],
    }
    for key, value in reference["compositions"].items():
        features = {"curated_known": 1., **value["curated"], "composition_embedding_known": 1.,
                    **{f"megnet_mean_{i:02d}": float(v) for i, v in enumerate(value["composition_embedding"])}}
        crystal = value.get("crystal_embedding")
        features["crystal_embedding_known"] = float(crystal is not None)
        features["crystal_reference_mean_coordinate_std"] = crystal["mean_coordinate_std"] if crystal else None
        projected = pca.transform(np.array(crystal["mean"]).reshape(1, -1))[0] if crystal else [None] * pca.n_components_
        features.update({f"alignn_pc_{i:02d}": float(v) if v is not None else None for i, v in enumerate(projected)})
        result[key] = features
    projection = {"training_composition_keys": available, "training_formulas": sorted(training_formulas),
                  "components": pca.components_.tolist(), "mean": pca.mean_.tolist(),
                  "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                  "n_components": int(pca.n_components_), "fit_policy": "unique active training source compositions only"}
    return {"schema_version": 1, "blocks": blocks, "compositions": result, "projection": projection,
            "reference_sha256": sha256(REFERENCE)}


def source_values(formula, table):
    fields = [f for block in table["blocks"].values() for f in block]
    if formula == "__NONE__":
        return dict.fromkeys(fields, 0.)  # Explicit absent slot, not a measured material.
    try:
        values = table["compositions"].get(chemistry.composition_key(formula))
    except (ValueError, TypeError):
        values = None
    if values is None:
        return {f: (0. if f.endswith("_known") else np.nan) for f in fields}
    return {f: np.nan if values[f] is None else values[f] for f in fields}


def enrich_frame(clean, table):
    additions = []
    for slot in range(1, 4):
        lookup = {f: source_values(f, table) for f in clean[f"target_{slot}"].unique()}
        rows = pd.DataFrame([lookup[f] for f in clean[f"target_{slot}"]], index=clean.index)
        additions.append(rows.add_prefix(f"target_{slot}_"))
    return pd.concat([clean, *additions], axis=1).copy()


def candidate_columns(blocks, manifest, table):
    old = manifest["model_features"]
    categorical = [f for f in old if f in ["chamber", "substrate", "gas_1", "gas_2", "gas_3"]]
    numeric = [f for f in old if f not in categorical and ("dos" in blocks or "dos" not in f)]
    for block in blocks:
        if block != "dos":
            numeric += [f"target_{i}_{f}" for i in range(1, 4) for f in table["blocks"][block]]
    if len(numeric) != len(set(numeric)):
        raise ValueError("Duplicate feature columns.")
    return numeric, categorical


def make_encoder(numeric, categorical):
    return ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                                keep_empty_features=True), StandardScaler()), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


def score_predictions(predictions, y):
    per_target = []
    for i in range(y.shape[1]):
        known = np.isfinite(y[:, i])
        per_target.append(float(np.sqrt(np.mean((predictions[known, i] - y[known, i]) ** 2))))
    return float(np.mean(per_target)), per_target


def choose_winner(results):
    """Equal estimator weight; normalize each score by its existing-DOS baseline."""
    baseline = results[results.candidate.eq("existing_dos")].set_index("estimator").validation_score
    result = results.copy()
    result["relative_score"] = result.apply(lambda row: row.validation_score / baseline.loc[row.estimator], axis=1)
    table = result.groupby("candidate", sort=False).agg(
        mean_relative_validation_score=("relative_score", "mean"),
        mean_validation_score=("validation_score", "mean"), encoded_features=("encoded_features", "first"))
    if any(result.groupby("candidate").estimator.nunique() != len(ESTIMATORS)):
        raise ValueError("Every candidate must finish the same two selectors.")
    return table.sort_values(["mean_relative_validation_score", "encoded_features"], kind="stable").reset_index()


def bootstrap_gain(predictions, arrays, groups, winner, baseline="existing_dos", draws=500):
    """Paired group bootstrap on existing validation; exploratory, selection-biased."""
    y = arrays["validation"]["y"]
    unique, inverse = np.unique(groups, return_inverse=True)
    counts, sums = {}, {}
    for key, pred in predictions.items():
        error = (pred - y) ** 2
        sums[key] = np.stack([np.bincount(inverse, weights=np.nan_to_num(error[:, i]), minlength=len(unique)) for i in range(2)], axis=1)
        counts[key] = np.stack([np.bincount(inverse, weights=np.isfinite(y[:, i]), minlength=len(unique)) for i in range(2)], axis=1)
    rng, gains = np.random.default_rng(42), []
    for _ in range(draws):
        picked = rng.integers(0, len(unique), size=len(unique))
        relative = []
        for estimator in ESTIMATORS:
            scores = []
            for candidate in [winner, baseline]:
                key = (candidate, estimator)
                score = np.sqrt(sums[key][picked].sum(axis=0) / counts[key][picked].sum(axis=0)).mean()
                scores.append(score)
            relative.append(scores[0] / scores[1])
        gains.append(1. - np.mean(relative))
    return {"validation_groups": len(unique), "draws": draws,
            "relative_gain_interval_95": np.quantile(gains, [.025, .975]).tolist(),
            "caveat": "paired validation-group bootstrap after selection; exploratory, not independent confirmation or a guarantee for all architectures"}


def run_study():
    STUDY.mkdir(parents=True, exist_ok=True)
    clean, arrays, manifest = load_selection_data()
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    training = clean[clean.split.eq("train")]
    formulas = {row[f"target_{i}"] for row in training.drop_duplicates("sample_library_id").to_dict("records")
                for i in range(1, 4) if row[f"target_{i}_power_W"] > 0}
    table = build_source_table(reference, formulas)
    json_write(STUDY / "source_feature_table.json", table)
    clean = enrich_frame(clean, table)
    rows, predictions = [], {}
    for candidate, blocks in CANDIDATES.items():
        numeric, categorical = candidate_columns(blocks, manifest, table)
        encoder = make_encoder(numeric, categorical)
        encoder.fit(clean.loc[clean.split.eq("train")])
        X = {s: encoder.transform(clean[clean.split.eq(s)]).astype("float32") for s in ["train", "validation"]}
        if candidate == "existing_dos":
            for split in X:
                np.testing.assert_array_equal(X[split], arrays[split]["X"])
        for estimator in ESTIMATORS:
            started = perf_counter()
            outputs = []
            with threadpool_limits(limits=4):
                for i in range(2):
                    known = np.isfinite(arrays["train"]["y"][:, i])
                    model = Ridge(**ESTIMATORS[estimator]) if estimator == "ridge" else RandomForestRegressor(**ESTIMATORS[estimator])
                    model.fit(X["train"][known], arrays["train"]["y"][known, i])
                    outputs.append(model.predict(X["validation"]))
            prediction = np.column_stack(outputs)
            score, targets = score_predictions(prediction, arrays["validation"]["y"])
            predictions[(candidate, estimator)] = prediction
            row = {"candidate": candidate, "estimator": estimator, "validation_score": score,
                   "thickness_scaled_RMSE": targets[0], "conductivity_scaled_RMSE": targets[1],
                   "encoded_features": X["train"].shape[1], "fit_predict_seconds": perf_counter() - started}
            rows.append(row)
            pd.DataFrame(rows).to_csv(STUDY / "validation_trials.csv", index=False)
            print(f"{candidate:26} {estimator:7} val={score:.6f} inputs={X['train'].shape[1]} ({row['fit_predict_seconds']:.1f}s)", flush=True)
    results = pd.DataFrame(rows)
    ranking = choose_winner(results)
    ranking.to_csv(STUDY / "feature_ranking.csv", index=False)
    winner = ranking.iloc[0].candidate
    gain = bootstrap_gain(predictions, arrays, clean.loc[clean.split.eq("validation"), "leakage_group"].to_numpy(), winner)
    numeric, categorical = candidate_columns(CANDIDATES[winner], manifest, table)
    decision = {
        "schema_version": 1, "selected_candidate": winner, "blocks": CANDIDATES[winner],
        "numeric_features": numeric, "categorical_features": categorical,
        "selection_rule": "minimum mean relative validation RMSE across fixed Ridge and Random Forest; each normalized by its existing-DOS baseline; equal estimator and target weights; tie -> fewer columns",
        "estimators": ESTIMATORS, "candidates": CANDIDATES,
        "reference_sha256": sha256(REFERENCE), "source_feature_table_sha256": sha256(STUDY / "source_feature_table.json"),
        "archived_manifest_sha256": sha256(ARCHIVE / "htem_prepared/manifest.json"),
        "split_manifest_sha256": sha256(ARCHIVE / "htem_prepared/split_manifest.csv"),
        "train_npz_sha256": sha256(ARCHIVE / "htem_prepared/train.npz"),
        "validation_npz_sha256": sha256(ARCHIVE / "htem_prepared/validation.npz"),
        "trials_sha256": sha256(STUDY / "validation_trials.csv"), "ranking_sha256": sha256(STUDY / "feature_ranking.csv"),
        "test_used_for_selection": False, "bootstrap": gain,
        "composition_representation": "fraction-mean of published 16-D MEGNet element vectors, not a full Roost/AtomSets network",
        "crystal_representation": "frozen official ALIGNN formation-energy 256-D pre-head features; mean over JARVIS references; 16 training-only PCA coordinates; not verified HTEM source phase",
        "limitations": ["One historical grouped validation split; no statistically confirmed or universally optimal feature set.",
                        "Selection uses two representative estimators, not every possible student architecture.",
                        "Existing test results were previously viewed, but no test artifact or score is used by this study.",
                        "No source-gun geometry, actual film composition/structure or new physical information is invented."],
    }
    json_write(STUDY / "selection_decision.json", decision)
    np.savez_compressed(STUDY / "validation_predictions.npz", **{f"{a}__{b}": p for (a, b), p in predictions.items()})
    return results, ranking, decision


def deploy_choice():
    """Explicit instructor operation: publish the selected offline specification."""
    import shutil
    decision = json.loads((STUDY / "selection_decision.json").read_text(encoding="utf-8"))
    for filename, field in [("source_feature_table.json", "source_feature_table_sha256"),
                            ("validation_trials.csv", "trials_sha256"), ("feature_ranking.csv", "ranking_sha256")]:
        if sha256(STUDY / filename) != decision[field]:
            raise ValueError(f"Study artifact changed: {filename}")
    if sha256(REFERENCE) != decision["reference_sha256"] or decision["test_used_for_selection"]:
        raise ValueError("Invalid reference or test-contaminated selection.")
    for source, dest in [("source_feature_table.json", "htem_frozen_source_features.json"),
                         ("validation_trials.csv", "htem_feature_validation_trials.csv"),
                         ("feature_ranking.csv", "htem_feature_ranking.csv")]:
        shutil.copy2(STUDY / source, MODULE / "data" / dest)
    json_write(MODULE / "data/htem_feature_selection.json", decision)
    return decision
