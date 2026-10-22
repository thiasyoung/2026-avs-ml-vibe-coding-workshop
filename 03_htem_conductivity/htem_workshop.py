"""HTEM acquisition and instructor-provided process-only data preparation.

No HTEM deposited-film property, spectrum, composition, ID, or post-deposition
quality indicator is eligible as input. Source formulas use independent atomic
and composition-reference data. Notebook 02 owns model experiments.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import htem_chemistry as chemistry
import htem_features as frozen_features
from htem_download import (
    API_BASE, RECIPE_FIELDS, EXTRA_PROCESS_FIELDS, LABEL_FIELDS, RAW_FIELDS, UNITS,
    download_htem, sha256, write_json,
)

PROCESS_NUMERIC_FEATURES = [
    "duration_min", "initial_temperature_C", "base_pressure_mTorr",
    "growth_pressure_mTorr", *[f"target_{i}_power_W" for i in range(1, 4)],
    *[f"gas_{i}_flow_sccm" for i in range(1, 4)], "x_mm", "y_mm", "local_temperature_C",
]
GEOMETRY_FIELDS = ["position", "x_mm", "y_mm"]
SPATIAL_FEATURES = ["x_mm", "y_mm", "local_temperature_C"]
SOURCE_FIELDS = [f"target_{i}" for i in range(1, 4)]
IDENTITY_CATEGORICAL_FEATURES = [
    "chamber", "substrate", *[f"target_{i}" for i in range(1, 4)],
    *[f"gas_{i}" for i in range(1, 4)],
]
RECIPE_IDENTITY_FEATURES = [f for f in PROCESS_NUMERIC_FEATURES if f not in SPATIAL_FEATURES] + IDENTITY_CATEGORICAL_FEATURES
NUMERIC_FEATURES = frozen_features.selected_numeric(
    PROCESS_NUMERIC_FEATURES + chemistry.CHEMISTRY_FEATURES + frozen_features.extra_fields())
CATEGORICAL_FEATURES = [f for f in IDENTITY_CATEGORICAL_FEATURES if f not in SOURCE_FIELDS]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Keep formula strings for recipe identity/strata/audit, but DROP them in the encoder.
FEATURES = MODEL_FEATURES + SOURCE_FIELDS
RECIPE_FEATURES = [field for field in FEATURES if field not in SPATIAL_FEATURES]
TARGETS = ["log10_thickness_um", "log10_conductivity_S_cm"]


def compact_raw(frame):
    """Allowlist export; drop spectra/personal metadata, but do not cleanse labels."""
    missing = set(RAW_FIELDS) - set(frame)
    if missing:
        raise ValueError(f"Missing raw HTEM fields: {sorted(missing)}")
    return frame[RAW_FIELDS].copy().sort_values(["sample_library_id", "position"]).reset_index(drop=True)


def slots(value):
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return [None] * 3
    values = list(value)
    if len(values) > 3:
        raise ValueError("More than three deposition sources/gases: revise the fixed feature schema.")
    return values + [None] * (3 - len(values))


def text(value, missing="__NONE__"):
    if value is None or pd.isna(value) or not str(value).strip():
        return missing
    return str(value).strip()


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def engineer_recipe(row):
    """Recipe settings only. Spatial query coordinates are added separately."""
    substrate = text(row.get("deposition_substrate_material"), "__UNKNOWN__").lower()
    # Only case/whitespace normalization. Do not equate quartz with fused silica,
    # or Eagle 2000 with Eagle XG based on a convenient-looking name.
    result = {
        "chamber": text(row.get("pdac"), "__UNKNOWN__"), "substrate": substrate,
        "duration_min": number(row.get("deposition_sample_time_min")),
        "initial_temperature_C": number(row.get("deposition_initial_temp_c")),
        "base_pressure_mTorr": number(row.get("deposition_base_pressure_mtorr")),
        "growth_pressure_mTorr": number(row.get("deposition_growth_pressure_mtorr")),
    }
    aliases = {"AR": "ARGON", "O2": "OXYGEN", "N2": "NITROGEN"}
    for index, (compound, power) in enumerate(zip(slots(row.get("deposition_compounds")),
                                                slots(row.get("deposition_power"))), 1):
        result[f"target_{index}"] = text(compound)
        power = number(power)
        # Empty source, missing power: absence is zero; unknown active power is not.
        if text(compound) == "__NONE__" and not np.isfinite(power):
            power = 0.0
        result[f"target_{index}_power_W"] = power
    for index, (gas, flow) in enumerate(zip(slots(row.get("deposition_gases")),
                                          slots(row.get("deposition_gas_flow_sccm"))), 1):
        gas = text(gas).upper()
        result[f"gas_{index}"] = aliases.get(gas, gas)
        flow = number(flow)
        if gas == "__NONE__" and not np.isfinite(flow):
            # Entire gas metadata absent is UNKNOWN, not evidence of zero gas.
            flow = np.nan
        result[f"gas_{index}_flow_sccm"] = flow
    result.update(chemistry.recipe_descriptors([result[f"target_{i}"] for i in range(1, 4)],
                                               [result[f"target_{i}_power_W"] for i in range(1, 4)]))
    result.update(frozen_features.recipe_features([result[f"target_{i}"] for i in range(1, 4)]))
    return {field: result[field] for field in RECIPE_FEATURES}


def inference_recipe(recipe):
    """Check core physical settings and reject any non-recipe input field."""
    unexpected = set(recipe) - set(RECIPE_FIELDS + EXTRA_PROCESS_FIELDS)
    if unexpected:
        raise ValueError(f"Only pre-deposition recipe fields are allowed: unexpected {sorted(unexpected)}")
    engineered = engineer_recipe(recipe)
    if not engineered["duration_min"] > 0:
        raise ValueError("Provide a finite positive deposition duration in minutes.")
    temperature = engineered["initial_temperature_C"]
    if not -273.15 < temperature <= 1500:
        raise ValueError("Provide a finite initial temperature in degrees Celsius.")
    active = False
    for i in range(1, 4):
        target, power = engineered[f"target_{i}"], engineered[f"target_{i}_power_W"]
        if not np.isfinite(power) or power < 0 or (power > 0 and target == "__NONE__"):
            raise ValueError("Provide consistent nonnegative power for every specified target.")
        active |= power > 0
        flow = engineered[f"gas_{i}_flow_sccm"]
        if np.isfinite(flow) and flow < 0:
            raise ValueError("Gas flows cannot be negative.")
    if not active:
        raise ValueError("At least one known target must have positive power.")
    for key in ["base_pressure_mTorr", "growth_pressure_mTorr"]:
        if np.isfinite(engineered[key]) and engineered[key] < 0:
            raise ValueError("Pressure cannot be negative.")
    return pd.DataFrame([engineered])[RECIPE_FEATURES]


def engineer_observation(row):
    temperature = number(row.get("absolute_temp_c"))
    if not -273.15 < temperature <= 1500:
        temperature = np.nan
    return {**engineer_recipe(row), "x_mm": number(row.get("x_mm")), "y_mm": number(row.get("y_mm")),
            "local_temperature_C": temperature}


def grid_feature_frame(rows, grid):
    """Broadcast RECIPE only; use each node's planned temperature, never one label row's."""
    frame = pd.concat([rows.iloc[[0]][FEATURES]] * len(grid), ignore_index=True)
    frame[["x_mm", "y_mm"]] = grid[["x_mm", "y_mm"]].to_numpy()
    frame["local_temperature_C"] = grid.get("local_temperature_C", np.nan)
    return frame


def inference_points(recipe, grid):
    """A recipe plus a known requested substrate grid; labels are never accepted."""
    inference_recipe(recipe)
    grid = pd.DataFrame(grid).copy()
    if set(grid) - {"position", "x_mm", "y_mm", "local_temperature_C", "sample_library_id", "id"}:
        raise ValueError("Supply only grid identities and x_mm/y_mm, not measured properties.")
    if not {"position", "x_mm", "y_mm"}.issubset(grid) or grid.empty:
        raise ValueError("Provide a nonempty grid with position, x_mm, and y_mm.")
    if "sample_library_id" in grid and grid.sample_library_id.nunique() != 1:
        raise ValueError("A map must describe exactly one deposition/library.")
    if grid.position.isna().any() or grid.position.duplicated().any():
        raise ValueError("Positions must be unique within the selected grid.")
    if not np.isfinite(grid[["x_mm", "y_mm"]].to_numpy(dtype=float)).all():
        raise ValueError("All requested substrate coordinates must be finite.")
    if grid.duplicated(["x_mm", "y_mm"]).any():
        raise ValueError("Distinct positions map to the same XY point; inspect the grid.")
    if "local_temperature_C" in grid:
        temperature = pd.to_numeric(grid.local_temperature_C, errors="raise")
        if (temperature.notna() & ~temperature.between(-273.15, 1500, inclusive="right")).any():
            raise ValueError("Planned local temperatures must be Celsius in (-273.15, 1500].")
    frame = pd.DataFrame([engineer_observation({**recipe, **point,
                         "absolute_temp_c": point.get("local_temperature_C")})
                         for point in grid.to_dict("records")])
    return grid.reset_index(drop=True), frame[FEATURES]


def cleanse(raw):
    """Deterministic, auditable cohort rules; never repair units by guessing.

    Restrict to powered-target deposition records with known positive duration
    and temperature, at least one usable label, and a valid spatial grid. Do not trim rare
    conductivity values just because they are extreme.
    """
    raw = compact_raw(raw)
    if raw[["id", "sample_library_id"]].isna().any().any():
        raise ValueError("Sample and library IDs must exist for leakage-safe grouping.")
    duplicate = raw.id.duplicated(keep=False)
    # Identical duplicates may be reduced; conflicting observations need review.
    for _, rows in raw[duplicate].groupby("id"):
        hashes = {json.dumps(row, default=lambda x: x.tolist(), sort_keys=True)
                  for row in rows.to_dict("records")}
        if len(hashes) > 1:
            raise ValueError("Conflicting duplicate sample IDs; review source, do not average.")
    raw = raw.drop_duplicates("id").reset_index(drop=True)
    if raw.duplicated(["sample_library_id", "position"]).any():
        raise ValueError("Duplicate library-position observations; do not pool or average them.")
    # Recipe descriptors are constant within a library: compute once, not 82k times.
    recipes = pd.DataFrame([{"sample_library_id": row["sample_library_id"], **engineer_recipe(row)}
                            for row in raw.drop_duplicates("sample_library_id").to_dict("records")])
    x = raw[["sample_library_id", "x_mm", "y_mm"]].merge(recipes, on="sample_library_id", validate="many_to_one")
    x[["x_mm", "y_mm"]] = x[["x_mm", "y_mm"]].apply(pd.to_numeric, errors="coerce")
    temperature = pd.to_numeric(raw.absolute_temp_c, errors="coerce")
    x["local_temperature_C"] = temperature.where(temperature.between(-273.15, 1500, inclusive="right"))
    x = x[FEATURES]
    y = raw[LABEL_FIELDS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    flags = pd.DataFrame(index=raw.index)
    invalid_thickness = ~y.thickness.between(0.001, 10)
    invalid_conductivity = ~(y.fpm_conductivity > 0)
    y.loc[invalid_thickness, "thickness"] = np.nan
    y.loc[invalid_conductivity, "fpm_conductivity"] = np.nan
    flags["no_usable_target"] = y[["thickness", "fpm_conductivity"]].isna().all(axis=1)
    flags["missing_grid_position"] = raw.position.isna() | ~np.isfinite(x[["x_mm", "y_mm"]]).all(axis=1)
    flags["ambiguous_grid_coordinates"] = raw.duplicated(["sample_library_id", "x_mm", "y_mm"], keep=False)
    flags["unknown_or_invalid_duration"] = ~(x.duration_min > 0)
    flags["unknown_or_invalid_temperature"] = ~x.initial_temperature_C.between(-273.15, 1500, inclusive="right")
    powers = x[[f"target_{i}_power_W" for i in range(1, 4)]]
    flags["no_positive_power"] = ~(powers > 0).any(axis=1)
    flags["invalid_target_power_pair"] = False
    for i in range(1, 4):
        p, target = x[f"target_{i}_power_W"], x[f"target_{i}"]
        flags["invalid_target_power_pair"] |= ((target == "__NONE__") & (p > 0)) | (p < 0) | p.isna()
    # Negative optional settings become unknown; never silently turn them into zero.
    optional = ["base_pressure_mTorr", "growth_pressure_mTorr"] + [f"gas_{i}_flow_sccm" for i in range(1, 4)]
    optional_invalid = (x[optional] < 0).any(axis=1)
    x[optional] = x[optional].mask(x[optional] < 0)
    audit = raw[["id", "sample_library_id", "position", "x_mm", "y_mm"]].copy()
    audit["excluded"] = flags.any(axis=1)
    audit["reasons"] = flags.apply(lambda row: ";".join(flags.columns[row]), axis=1)
    audit["negative_optional_setting_set_missing"] = optional_invalid
    audit["thickness_missing_or_invalid"] = invalid_thickness
    audit["conductivity_missing_or_invalid"] = invalid_conductivity
    audit["local_temperature_missing_or_invalid"] = x.local_temperature_C.isna()
    keep = ~audit.excluded
    clean = pd.concat([raw[["id", "sample_library_id", "position"]], x, y.rename(columns={
        "thickness": "thickness_um", "opt_direct_bandgap": "bandgap_eV",
        "fpm_conductivity": "conductivity_S_cm"})], axis=1)[keep].reset_index(drop=True)
    clean["log10_thickness_um"] = np.log10(clean.thickness_um)
    clean["log10_conductivity_S_cm"] = np.log10(clean.conductivity_S_cm)
    if clean.empty:
        raise ValueError("No eligible thickness/conductivity spatial process records.")
    return clean, audit


def leakage_groups(clean):
    """Connected libraries: identical recipes in separate runs also stay together.

    No labels enter grouping. Missing recipe settings use a literal sentinel.
    """
    parent = {int(v): int(v) for v in clean.sample_library_id.unique()}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    seen = {}
    for row in clean.to_dict("records"):
        # Coordinates are not recipe identity. Different grids of one recipe stay together.
        key = tuple(None if pd.isna(row[c]) else row[c] for c in RECIPE_IDENTITY_FEATURES)
        lib = int(row["sample_library_id"])
        if key in seen:
            a, b = find(lib), find(seen[key])
            parent[max(a, b)] = min(a, b)
        else:
            seen[key] = lib
    return clean.sample_library_id.map(lambda v: find(int(v))).to_numpy()


SPLITS = ["train", "validation", "test"]
SPLIT_FRACTIONS = np.array([0.6, 0.2, 0.2])


def material_classes(clean):
    """Strata are sets of active source materials, not measured film composition.

    Sorting is ONLY for split strata. Model inputs retain source/power slot pairing.
    Zero-power sources do not define the deposited material combination.
    """
    def category(row):
        active = sorted({row[f"target_{i}"] for i in range(1, 4)
                         if row[f"target_{i}"] != "__NONE__" and row[f"target_{i}_power_W"] > 0})
        if not active:
            raise ValueError("Material stratification requires a known powered target.")
        return " + ".join(active)
    return clean.apply(category, axis=1).rename("material_class")


def split_groups(clean, seed=42, frozen_split=None):
    """Material-stratified whole-group split; never split a substrate to balance it.

    Each class with >=3 independent groups occurs in all splits. Singletons go
    to train; two-group classes get one train group and one validation/test group.
    Integer quotas approach 60/20/20; group sizes preclude exact point fractions.
    No property values or model scores enter assignment.
    """
    groups = leakage_groups(clean)
    if len(set(groups)) < 10:
        raise ValueError("Need at least ten independent recipe/library groups.")
    table = pd.DataFrame({"group": groups, "material_class": material_classes(clean).to_numpy(),
                          "library": clean.sample_library_id.to_numpy()})
    if table.groupby("group").material_class.nunique().gt(1).any():
        raise ValueError("A connected library/recipe group has multiple material classes; review metadata, do not split it.")
    inventory = table.groupby("group", sort=True).agg(
        material_class=("material_class", "first"), points=("library", "size"), libraries=("library", "nunique"))
    rng = np.random.default_rng(seed)
    if frozen_split is not None:
        anchors = pd.read_csv(frozen_split, dtype={"id": str}) if isinstance(frozen_split, (str, Path)) else frozen_split
        frozen = anchors.groupby("sample_library_id").split.agg(lambda a: a.unique().tolist())
        if any(len(v) != 1 or v[0] not in SPLITS for v in frozen):
            raise ValueError("Frozen libraries must have exactly one valid split.")
        fixed = table[table.library.isin(frozen.index)].copy()
        fixed["split"] = fixed.library.map(lambda lib: frozen.loc[lib][0])
        if fixed.groupby("group").split.nunique().gt(1).any():
            raise ValueError("Frozen assignments conflict within a connected recipe group.")
        assignment = fixed.groupby("group").split.first().to_dict()
        for category, part in inventory.groupby("material_class", sort=True):
            count = len(part)
            counts = np.array([sum(assignment.get(g) == s for g in part.index) for s in SPLITS])
            unknown = [int(g) for g in part.index if g not in assignment]
            rng.shuffle(unknown)
            for group in unknown:
                # Fill absent feasible splits before approaching the 60/20/20 quotas.
                if counts[0] == 0:
                    index = 0
                elif count >= 3 and (counts == 0).any():
                    index = int(np.flatnonzero(counts == 0)[0])
                elif count == 2 and counts[1:].sum() == 0:
                    index = int(rng.choice([1, 2]))
                else:
                    deficit = count * SPLIT_FRACTIONS - counts
                    index = int(rng.choice(np.flatnonzero(np.isclose(deficit, deficit.max()))))
                assignment[group] = SPLITS[index]
                counts[index] += 1
        return np.array([assignment[int(g)] for g in groups]), groups
    assignment, totals = {}, np.zeros(3, dtype=int)
    classes = inventory.material_class.value_counts().rename_axis("material_class").reset_index(name="groups")
    classes = classes.sort_values(["groups", "material_class"], ascending=[False, True])
    for category in classes.material_class:
        part = inventory[inventory.material_class.eq(category)]
        count = len(part)
        if count == 1:
            quotas = np.array([1, 0, 0])
        elif count == 2:
            # Rare holdouts alternate according to current group-count deficits.
            quotas = np.array([1, 0, 0])
            tied = np.flatnonzero(totals[1:] == totals[1:].min()) + 1
            quotas[rng.choice(tied)] = 1
        else:
            quotas = np.ones(3, dtype=int)
            while quotas.sum() < count:
                deficit = count * SPLIT_FRACTIONS - quotas
                quotas[rng.choice(np.flatnonzero(np.isclose(deficit, deficit.max())))] += 1
        # Among random whole-group assignments with the same quotas, prefer
        # representative point/library counts, not convenient outcome values.
        weights = part[["points", "libraries"]].to_numpy(dtype=float)
        desired = quotas[:, None] / count * weights.sum(axis=0)
        best_score, best_order = float("inf"), None
        for _ in range(128 if count > 2 else 1):
            order = rng.permutation(count)
            allocations = np.split(order, np.cumsum(quotas)[:-1])
            observed = np.array([weights[idx].sum(axis=0) for idx in allocations])
            score = np.square((observed - desired) / np.maximum(desired, 1)).sum()
            if score < best_score:
                best_score, best_order = score, allocations
        for name, indices in zip(SPLITS, best_order):
            assignment.update({int(part.index[i]): name for i in indices})
        totals += quotas
    split = np.array([assignment[int(group)] for group in groups])
    if set(split) != set(SPLITS):
        raise ValueError("Material-safe split has an empty holdout; need more independent groups per material class.")
    return split, groups


def material_split_report(clean):
    """Audit representation at independent-group, deposition, and point levels."""
    table = clean.assign(material_class=material_classes(clean))
    rows = []
    for category, part in table.groupby("material_class", sort=True):
        total = part.leakage_group.nunique()
        record = {"material_class": category, "total_groups": int(total),
                  "all_splits_feasible": bool(total >= 3)}
        for name in SPLITS:
            subset = part[part.split.eq(name)]
            record.update({f"{name}_groups": int(subset.leakage_group.nunique()),
                           f"{name}_libraries": int(subset.sample_library_id.nunique()),
                           f"{name}_points": len(subset)})
        record["all_splits_present"] = all(record[f"{name}_groups"] > 0 for name in SPLITS)
        record["coverage_note"] = ("all splits" if total >= 3 else
                                   "train + one holdout; only two independent groups" if total == 2 else
                                   "train only; one independent group")
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["total_groups", "material_class"], ascending=[False, True]).reset_index(drop=True)


def prepare(raw, output, raw_sha256, seed=42, frozen_split=None):
    """Save fixed feature matrices, labels, IDs, audit, and train-only transforms."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    clean, audit = cleanse(raw)
    clean["material_class"] = material_classes(clean)
    clean["split"], clean["leakage_group"] = split_groups(clean, seed, frozen_split)
    anchors = None
    clean["legacy_evaluation"] = False
    if frozen_split is not None:
        anchors = pd.read_csv(frozen_split, dtype={"id": str}) if isinstance(frozen_split, (str, Path)) else frozen_split.copy()
        current = clean.assign(id=clean.id.astype(str)).set_index("id")
        if not set(anchors.id.astype(str)).issubset(current.index):
            raise ValueError("Frozen observations were lost during acquisition/cleansing; inspect instead of changing the holdout.")
        check = current.loc[anchors.id.astype(str)]
        for field in ["split", "sample_library_id", "position", "x_mm", "y_mm", "thickness_um", "conductivity_S_cm"]:
            if field in anchors:
                if field == "split":
                    np.testing.assert_array_equal(check[field], anchors[field])
                else:
                    np.testing.assert_allclose(check[field], anchors[field], rtol=1e-12, atol=1e-12)
        clean["legacy_evaluation"] = clean.id.astype(str).isin(anchors.id.astype(str))
    balance = material_split_report(clean)
    balance.to_csv(output / "material_split_balance.csv", index=False)
    chemistry.coverage_report(clean).to_csv(output / "source_chemistry_coverage.csv", index=False)
    assert balance.loc[balance.all_splits_feasible, "all_splits_present"].all()
    train = clean.split.eq("train")
    preprocessor = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                                keep_empty_features=True), StandardScaler()), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    preprocessor.fit(clean.loc[train, FEATURES])
    target_scaler = StandardScaler().fit(clean.loc[train, TARGETS])
    feature_names = preprocessor.get_feature_names_out().tolist()
    for name in ["train", "validation", "test"]:
        part = clean[clean.split.eq(name)]
        encoded = preprocessor.transform(part[FEATURES]).astype("float32")
        scaled_y = target_scaler.transform(part[TARGETS]).astype("float32")
        mask = np.isfinite(scaled_y)
        if not np.isfinite(encoded).all() or not mask.any(axis=1).all() or not mask.any(axis=0).all():
            raise ValueError("Expected finite inputs, at least one observed target per row, and both targets represented in each split.")
        np.savez_compressed(output / f"{name}.npz", X=encoded, y=scaled_y,
                            y_mask=mask, legacy_evaluation=part.legacy_evaluation.to_numpy(dtype=bool),
                            sample_id=part.id.to_numpy(dtype=str), library_id=part.sample_library_id.to_numpy(),
                            position=part.position.to_numpy(), x_mm=part.x_mm.to_numpy(), y_mm=part.y_mm.to_numpy())
    clean.to_parquet(output / "clean_process_table.parquet", index=False)
    audit.to_csv(output / "cleaning_audit.csv", index=False)
    clean[["id", "sample_library_id", "position", "x_mm", "y_mm", "material_class", "leakage_group", "split"]].to_csv(output / "split_manifest.csv", index=False)
    # Catalog uses all raw grid points, before label filtering, not target-dependent extents.
    grids = raw[["id", "sample_library_id", "position", "x_mm", "y_mm", "absolute_temp_c"]].copy()
    grids = grids.rename(columns={"absolute_temp_c": "local_temperature_C"})
    grids.local_temperature_C = pd.to_numeric(grids.local_temperature_C, errors="coerce")
    grids.local_temperature_C = grids.local_temperature_C.where(grids.local_temperature_C.between(-273.15, 1500, inclusive="right"))
    grids.to_parquet(output / "substrate_grids.parquet", index=False)
    joblib.dump({"preprocessor": preprocessor, "target_scaler": target_scaler,
                 "features": FEATURES, "targets": TARGETS}, output / "preprocessing.joblib")
    manifest = {
        "version": 5, "seed": seed, "raw_sha256": raw_sha256,
        "rows_raw": len(raw), "rows_clean": len(clean), "libraries": int(clean.sample_library_id.nunique()),
        "independent_groups": int(clean.leakage_group.nunique()),
        "counts": clean.split.value_counts().to_dict(), "recipe_features": FEATURES,
        "label_counts": {s: clean.loc[clean.split.eq(s), TARGETS].notna().sum().to_dict() for s in SPLITS},
        "frozen_evaluation": {"enabled": anchors is not None,
            "sha256": sha256(frozen_split) if isinstance(frozen_split, (str, Path)) else None,
            "counts": clean[clean.legacy_evaluation].split.value_counts().to_dict(),
            "policy": "All original library assignments fixed; new connected groups allocated by material class. Original paired observations scored separately."},
        "missing_labels": "NaN plus explicit per-target mask; never impute labels; average observed per-target standardized MSE equally",
        "temperature": "local_temperature_C from API absolute_temp_c per position; user supplies planned/calibrated map at inference; no silent initial-temperature substitution",
        "encoded_features": feature_names, "targets": TARGETS, "label_units": UNITS,
        "model_features": MODEL_FEATURES, "formula_metadata_only": SOURCE_FIELDS,
        "frozen_feature_selection": {"candidate": frozen_features.specification()["selected_candidate"],
            "blocks": frozen_features.specification()["blocks"],
            "selection_rule": frozen_features.specification()["selection_rule"],
            "test_used_for_selection": False,
            "policy": "Instructor-selected offline features; shared immutable matrices for all architectures; no selection in participant training."},
        "chemistry": {**chemistry.reference_fingerprint(),
            "source_encoding": "formula descriptors; no one-hot target-material columns",
            "bandgap": "independent experimental composition-reference median/range, not actual HTEM target/film properties",
            "dos": "independent JARVIS spin-up DOS shape summaries; included only if selected in frozen feature specification",
            "composition_embedding": "stoichiometric mean of frozen published 16-D MEGNet element vectors; included only if selected; not a full Roost model",
            "crystal_embedding": "official frozen ALIGNN formation-energy pre-head readout from composition-matched JARVIS reference crystals; mean over references and training-only PCA; included only if selected; not verified source or film structure",
            "power_weighting": "engineering heuristic, not deposited stoichiometry or a physical mixture bandgap"},
        "selection_score": "mean RMSE on two training-standardized log10 targets",
        "spatial_key": ["sample_library_id", "position"],
        "coordinate_source": "HTEM scalar-properties substrate x_mm/y_mm; never sample xyz_mm stage/focus",
        "split_rule": "active-target-material-combination stratified connected library/recipe groups; approximately 60/20/20, not exact point fractions",
        "material_stratification": {
            "category": "sorted unique positive-power target materials; no measured film composition",
            "material_classes": len(balance),
            "classes_with_three_or_more_groups": int(balance.all_splits_feasible.sum()),
            "classes_present_in_all_splits": int(balance.all_splits_present.sum()),
            "singleton_classes_train_only": int(balance.total_groups.eq(1).sum()),
            "two_group_classes_train_plus_one_holdout": int(balance.total_groups.eq(2).sum()),
            "rare_policy": "1 group: train only; 2 groups: train plus one holdout; >=3: every split",
            "training_resampling": "none; preserve observed material frequencies, no duplicated/oversampled maps",
        },
        "limitations": [
            "Historical observational subset, not a causal or universally transferable emulator.",
            "Spatial prediction requires the requested substrate grid in the HTEM x_mm/y_mm frame.",
            "Do not rotate, normalize by observed labels, reshape by row order, or pool maps across libraries.",
            "Initial temperature is not a full temperature trajectory; missing settings remain uncertainty.",
            "At-least-one-label and powered-target selection changes the population; PLD requires separate preparation.",
            "Bandgap is retained as an optional scalar label but not required for this two-output map task.",
            "Thickness unit follows the course source convention (um); no heuristic unit repair.",
            "Rare material classes cannot occupy all three splits without leakage; integer class quotas change total fractions.",
        ],
    }
    manifest["files_sha256"] = {p.name: sha256(p) for p in output.iterdir()
                                 if p.name in {"train.npz", "validation.npz", "test.npz", "split_manifest.csv",
                                               "clean_process_table.parquet", "preprocessing.joblib", "substrate_grids.parquet",
                                               "material_split_balance.csv", "source_chemistry_coverage.csv"}}
    write_json(output / "manifest.json", manifest)
    return clean, audit, manifest


def load_prepared(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 5:
        raise ValueError("Prepared data predates material stratification or source-chemistry encoding. Rerun notebook 01 and all model notebooks.")
    if any(manifest.get("chemistry", {}).get(key) != value for key, value in chemistry.reference_fingerprint().items()):
        raise ValueError("Source-chemistry reference/featurizer changed. Rerun notebook 01 and all model notebooks.")
    if manifest["recipe_features"] != FEATURES or manifest["targets"] != TARGETS:
        raise ValueError("Prepared feature/target schema differs from the process-only contract.")
    for name, expected in manifest["files_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}. Rerun notebook 01 completely.")
    data = {split: dict(np.load(output / f"{split}.npz", allow_pickle=False))
            for split in ["train", "validation", "test"]}
    for part in data.values():
        if not np.array_equal(part["y_mask"], np.isfinite(part["y"])):
            raise ValueError("Observed-label mask disagrees with targets.")
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(data[a]["library_id"]) & set(data[b]["library_id"]):
            raise ValueError("Library leakage detected.")
    return data, joblib.load(output / "preprocessing.joblib"), manifest


def inverse_predictions(scaled, target_scaler):
    transformed = target_scaler.inverse_transform(np.asarray(scaled))
    return pd.DataFrame({
        "thickness_um": np.power(10.0, transformed[:, 0]),
        "conductivity_S_cm": np.power(10.0, transformed[:, 1]),
    })


def plot_substrate_maps(table, library_id=None):
    """Plot exact coordinates with holes masked; never average or interpolate points."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    table = table.copy()
    if "library_id" in table:
        if library_id is not None:
            table = table[table.library_id.eq(library_id)]
        if table.library_id.nunique() != 1:
            raise ValueError("Select exactly one deposition/library for a substrate map.")
        label = f"Library {table.library_id.iloc[0]}"
    else:
        label = "Requested recipe/grid"
    if table.empty or table.position.duplicated().any() or table.duplicated(["x_mm", "y_mm"]).any():
        raise ValueError("Empty or ambiguous grid; do not merge duplicate map positions.")
    if not np.isfinite(table[["x_mm", "y_mm"]].to_numpy()).all():
        raise ValueError("Map coordinates must be finite.")
    has_actual = "actual_thickness_um" in table
    kinds = ["actual", "predicted"] if has_actual else ["predicted"]
    fig, axes = plt.subplots(2, len(kinds), figsize=(5 * len(kinds), 7), squeeze=False)
    for row, target in enumerate(["thickness_um", "conductivity_S_cm"]):
        columns = [kind + "_" + target for kind in kinds]
        values = table[columns].to_numpy(dtype=float)
        finite = values[np.isfinite(values) & ((values > 0) if row else True)]
        if not len(finite):
            raise ValueError(f"No finite map values for {target}.")
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            hi = lo * 1.01 if lo > 0 else lo + 1
        norm = LogNorm(vmin=lo, vmax=hi) if row else Normalize(vmin=lo, vmax=hi)
        for col, kind in enumerate(kinds):
            # pivot(), not pivot_table(): duplicate spatial observations must raise.
            grid = table.pivot(index="y_mm", columns="x_mm", values=kind + "_" + target)
            grid = grid.sort_index().sort_index(axis=1)
            ax = axes[row, col]
            image = ax.pcolormesh(grid.columns.to_numpy(), grid.index.to_numpy(),
                                 np.ma.masked_invalid(grid.to_numpy()), shading="nearest", cmap="viridis", norm=norm)
            ax.set(title=f"{kind.title()} {target}", xlabel="Substrate x (mm)", ylabel="Substrate y (mm)", aspect="equal")
            ax.invert_yaxis()  # explicit image-style substrate frame, consistent for actual/predicted
            fig.colorbar(image, ax=ax, label="S/cm (log scale)" if row else "um", shrink=0.75)
    fig.suptitle(label + " · exact sample coordinates; blank = unobserved")
    fig.tight_layout()
    return fig


def masked_mse(prediction, target):
    """Equal weight per observed property; absent measurements have zero gradient."""
    import torch
    mask = torch.isfinite(target)
    counts = mask.sum(dim=0)
    if not mask.any():
        raise ValueError("No observed targets in this batch.")
    safe = torch.nan_to_num(target)
    per_target = ((prediction - safe).square() * mask).sum(dim=0) / counts.clamp_min(1)
    return per_target[counts > 0].mean()


def observed_mse(prediction, target):
    """Per-target observed MSE in numpy, with explicit nonempty-target checks."""
    mask = np.isfinite(target)
    if not mask.any(axis=0).all():
        raise ValueError("Each target needs observed labels for evaluation.")
    return np.array([np.mean((prediction[mask[:, i], i] - target[mask[:, i], i]) ** 2)
                     for i in range(target.shape[1])])


def fit_neural_head(x, y, x_val, y_val, hidden=(64, 32), lr=1e-3,
                    weight_decay=1e-3, epochs=300, patience=40, seed=42):
    """Workshop-sized CPU network; validation selects checkpoint, never test."""
    import copy
    import torch
    from torch import nn

    if epochs < 1 or patience < 1 or lr <= 0 or any(width < 1 for width in hidden):
        raise ValueError("Use positive learning rate, epochs, patience, and hidden-layer widths.")
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    layers = []
    previous = x.shape[1]
    for width in hidden:
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        previous = width
    layers.append(nn.Linear(previous, y.shape[1]))
    model = nn.Sequential(*layers)
    tensors = [torch.tensor(a, dtype=torch.float32) for a in [x, y, x_val, y_val]]
    xt, yt, xv, yv = tensors
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)
    loss_fn = masked_mse
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, best_weights, waiting = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in torch.randperm(len(xt), generator=generator).split(128):
            optimizer.zero_grad()
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(loss_fn(model(xt), yt))
            val_loss = float(loss_fn(model(xv), yv))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if not np.isfinite(val_loss):
            raise ValueError("Nonfinite neural-network loss; review hyperparameters.")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_weights, waiting = copy.deepcopy(model.state_dict()), 0
        else:
            waiting += 1
        scheduler.step()
        if waiting >= patience:
            break
    model.load_state_dict(best_weights)
    model.eval()
    return model, pd.DataFrame(history), {
        "best_epoch": best_epoch, "epochs_run": len(history),
        "stop_reason": "validation patience" if waiting >= patience else "epoch cap (not proof of convergence)",
        "best_validation_MSE": best_loss,
    }


def neural_predict(model, x):
    import torch
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(np.asarray(x), dtype=torch.float32)).numpy()
