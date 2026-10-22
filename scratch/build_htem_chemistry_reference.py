"""Print a compact, checksummed source-material reference JSON; no HTEM labels."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

MODULE = Path(__file__).resolve().parents[1] / "03_htem_conductivity"
sys.path.insert(0, str(MODULE))
import htem_chemistry as chemistry


def build():
    # Existing atomic table is a pinned, attributed extraction, not a dependency.
    result = json.loads(chemistry.REFERENCE.read_text(encoding="utf-8"))
    url = "https://ndownloader.figshare.com/files/13464434"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    expected = "2d0980e3533c1ba6ad6e392a88f08cfcf2d311d4b7fe6eb0b0c8e876211dfda3"
    if digest != expected:
        raise ValueError("Reference band-gap download differs from pinned matminer v0.9.3 metadata.")
    payload = json.loads(gzip.decompress(response.content))
    table = pd.DataFrame(payload["data"], columns=payload["columns"])
    # Determine reference coverage from the complete recipe catalog, not outcomes/splits.
    catalog = MODULE / "data/htem_spatial_libraries.parquet"
    labels = sorted({str(label).strip() for row in pd.read_parquet(catalog).deposition_compounds
                     if row is not None for label in row if label is not None and str(label).strip()})
    keys, unsupported = set(), []
    for label in labels:
        try:
            keys.add(chemistry.composition_key(label))
        except ValueError:
            unsupported.append(label)
    matched = {}
    for formula, gap in table[["formula", "gap expt"]].itertuples(index=False, name=None):
        if not isinstance(formula, str) or not np.isfinite(gap) or gap < 0:
            continue
        try:
            key = chemistry.composition_key(formula)
        except ValueError:
            continue
        if key in keys:
            matched.setdefault(key, []).append((formula, float(gap)))
    result["bandgaps"] = {
        key: {"median_eV": float(np.median([g for _, g in rows])),
              "min_eV": min(g for _, g in rows), "max_eV": max(g for _, g in rows),
              "records": len(rows), "formulas": sorted({f for f, _ in rows})}
        for key, rows in sorted(matched.items())}
    result["provenance"]["bandgaps"] = {
        "url": url, "raw_sha256": digest, "external_rows": len(table),
        "dataset_doi": "https://doi.org/10.6084/m9.figshare.7289687.v1",
        "doi": "https://doi.org/10.1021/acs.jpclett.8b00124",
        "authors": "Ya Zhuo, Aria Mansouri Tehrani, Jakoah Brgoch (2018)",
        "license": "MIT (Figshare article 7289687)",
        "metadata": "https://raw.githubusercontent.com/hackingmaterials/matminer/v0.9.3/matminer/datasets/dataset_metadata.json",
        "rule": "Median, min, max of independent experimental references matched by exact normalized atomic composition. No phase/temperature identification; NOT the actual target or film band gap. Duplicates retained as supplied, not independent uncertainty estimates.",
        "unknown_policy": "No match remains missing; never substitute measured HTEM bandgap, average elemental gaps, or assume a metal/semiconductor gap.",
    }
    result["provenance"]["catalog"] = {"sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "source_labels": labels, "unsupported_labels": unsupported,
        "note": "Reference subset covers catalog source formulas; no film outcomes or split assignments read."}
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, allow_nan=False))
