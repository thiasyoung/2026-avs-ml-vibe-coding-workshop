"""Offline, instructor-selected source features shared by every architecture.

No external model, download, PCA fitting or feature selection occurs here. The
selected columns and source vectors are immutable inputs to notebook 01.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from htem_paths import MODULE_DIR, source_sha256

DATA = MODULE_DIR / "data"
SPECIFICATION = DATA / "htem_feature_selection.json"
TABLE = DATA / "htem_frozen_source_features.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def specification():
    value = json.loads(SPECIFICATION.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("test_used_for_selection") is not False:
        raise ValueError("Invalid frozen HTEM feature-selection specification.")
    return value


@lru_cache(maxsize=1)
def source_table():
    value = json.loads(TABLE.read_text(encoding="utf-8"))
    if digest(TABLE) != specification()["source_feature_table_sha256"]:
        raise ValueError("Frozen source-feature table checksum mismatch.")
    return value


def extra_fields():
    return [f"target_{i}_{f}" for i in range(1, 4)
            for fields in source_table()["blocks"].values() for f in fields]


def selected_numeric(available):
    selected = specification()["numeric_features"]
    if len(selected) != len(set(selected)) or set(selected) - set(available):
        raise ValueError("Frozen feature schema contains duplicate or unsupported columns.")
    if not {"x_mm", "y_mm", "local_temperature_C"}.issubset(selected):
        raise ValueError("Frozen features must retain physical XY and local temperature.")
    return list(selected)


def recipe_features(targets):
    # Import at call time: chemistry provides the strict parser, without a
    # circular module-level dependency or any model-specific preprocessing.
    import htem_chemistry as chemistry
    table = source_table()
    fields = [f for fields in table["blocks"].values() for f in fields]
    result = {}
    for index, formula in enumerate(targets, 1):
        if formula == "__NONE__":
            values = dict.fromkeys(fields, 0.)
        else:
            try:
                values = table["compositions"].get(chemistry.composition_key(formula))
            except (ValueError, TypeError):
                values = None
            if values is None:
                values = {f: 0. if f.endswith("_known") else np.nan for f in fields}
        result.update({f"target_{index}_{field}": np.nan if values[field] is None else values[field]
                       for field in fields})
    return result


def fingerprint():
    # Hash the files afresh, even if parsed lookups are cached. A modified
    # reference cannot silently pass model reload/prepared-data checks.
    return {"frozen_feature_spec_sha256": digest(SPECIFICATION),
            "frozen_source_features_sha256": digest(TABLE),
            "frozen_feature_code_sha256": source_sha256(__file__),
            "feature_element_reference_sha256": digest(DATA / "htem_feature_element_reference.json"),
            "feature_candidates_sha256": digest(DATA / "htem_feature_candidates.json")}
