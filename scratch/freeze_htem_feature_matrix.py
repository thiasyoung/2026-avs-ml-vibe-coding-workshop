"""Publish notebook 01's exact matrices and prove unchanged cohort/labels/splits."""
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "03_htem_conductivity"
sys.path.insert(0, str(MODULE / 'lib'))
import htem_workshop as htem

source = MODULE / "outputs/htem_prepared"
old = MODULE / "archive/pre_feature_study_2026_10_21/htem_prepared"
data, _, manifest = htem.load_prepared(source)
pd.testing.assert_frame_equal(pd.read_csv(source / "split_manifest.csv"), pd.read_csv(old / "split_manifest.csv"))
for split, values in data.items():
    with np.load(old / f"{split}.npz", allow_pickle=False) as previous:
        for field in previous.files:
            if field != "X":
                np.testing.assert_array_equal(values[field], previous[field], err_msg=f"Changed {split}/{field}")

# The final prepared matrix must reproduce the selected STUDY matrix exactly,
# not merely have an equal column count or a matching feature-set name.
import htem_feature_study as study
clean, arrays, original_manifest = study.load_selection_data()
table = json.loads((MODULE / "data/htem_frozen_source_features.json").read_text(encoding="utf-8"))
decision = htem.frozen_features.specification()
frame = study.enrich_frame(clean, table)
numeric, categorical = study.candidate_columns(decision["blocks"], original_manifest, table)
encoder = study.make_encoder(numeric, categorical).fit(frame[frame.split.eq("train")])
for split in ["train", "validation"]:
    expected = encoder.transform(frame[frame.split.eq(split)]).astype("float32")
    np.testing.assert_array_equal(data[split]["X"], expected)

destination = MODULE / "data/frozen_features"
destination.mkdir(parents=True, exist_ok=True)
for name in [*manifest["files_sha256"], "manifest.json"]:
    shutil.copy2(source / name, destination / name)
htem.load_prepared(destination)
print("Published fixed matrices:", destination)
print("Feature set:", manifest["frozen_feature_selection"]["candidate"])
print("Identical to selected study inputs; all prior identities, labels, masks, XY and split assignments preserved.")
print("Total distributed prepared bytes:", sum(p.stat().st_size for p in destination.iterdir() if p.is_file()))
