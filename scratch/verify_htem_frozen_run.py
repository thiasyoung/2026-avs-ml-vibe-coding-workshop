"""Real-data smoke checks for frozen matrices, all models and fresh checkouts."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "03_htem_conductivity"
sys.path.insert(0, str(MODULE))
import htem_experiments as exp
import htem_workshop as htem

context = exp.load_context(MODULE)
frozen_data, _, frozen_manifest = htem.load_prepared(MODULE / "data/frozen_features")
assert frozen_manifest == context.manifest
for split in htem.SPLITS:
    for field in context.data[split]:
        np.testing.assert_array_equal(context.data[split][field], frozen_data[split][field])
comparison = exp.collect_comparison(context)
assert len(comparison) == 5
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for key in exp.EXPERIMENTS:
        output = MODULE / "outputs/htem_architectures" / key
        bundle = joblib.load(output / "model.joblib")
        full = pd.read_csv(output / "heldout_map.csv")
        recipe = exp.example_recipe(context, int(full.library_id.iloc[0]))
        grid = full[["position", "x_mm", "y_mm", "local_temperature_C"]]
        predicted = exp.predict_substrate(bundle, recipe, grid)
        fields = ["predicted_thickness_um", "predicted_conductivity_S_cm"]
        np.testing.assert_allclose(predicted[fields], full[fields], rtol=5e-5, atol=1e-7)
        if key != "grid_gnn":
            individual = pd.concat([exp.predict_substrate(bundle, recipe, grid.iloc[[i]]) for i in range(len(grid))], ignore_index=True)
            np.testing.assert_allclose(individual[fields], predicted[fields], rtol=5e-5, atol=1e-7)
            shuffled = exp.predict_substrate(bundle, recipe, grid.sample(frac=1, random_state=42))
            np.testing.assert_allclose(shuffled.sort_values("position")[fields], predicted.sort_values("position")[fields], rtol=5e-5, atol=1e-7)
        print(f"PASS {key}: saved-model/full-grid reload, {len(grid)} locations" + ("; single-point/batch/shuffled invariance" if key != "grid_gnn" else "; neighbor-aware graph"), flush=True)

# No outputs/acquisition directory: exercise what a fresh GitHub checkout uses.
with tempfile.TemporaryDirectory() as temp:
    checkout = Path(temp)
    shutil.copytree(MODULE / "data/frozen_features", checkout / "data/frozen_features")
    shutil.copy2(MODULE / "data/htem_spatial_snapshot.parquet", checkout / "data/htem_spatial_snapshot.parquet")
    fresh = exp.load_context(checkout)
    assert fresh.prepared == checkout / "data/frozen_features"
    assert fresh.fingerprint == context.fingerprint
    exp.example_recipe(fresh, int(context.clean.sample_library_id.iloc[0]))
print("PASS fresh checkout: frozen-matrix fallback and raw-recipe snapshot inference without generated outputs.")
print(comparison[["algorithm", "validation_score", "test_thickness_MAE_um", "test_conductivity_log10_MAE"]].to_string(index=False))
