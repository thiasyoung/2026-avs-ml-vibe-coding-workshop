"""Compact public HTEM acquisition using scalar batches and stripped fallbacks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = "https://htem-api.nlr.gov/api"
RECIPE_FIELDS = [
    "pdac", "deposition_sample_time_min", "deposition_initial_temp_c",
    "deposition_base_pressure_mtorr", "deposition_growth_pressure_mtorr",
    "deposition_compounds", "deposition_power", "deposition_gases",
    "deposition_gas_flow_sccm", "deposition_substrate_material",
]
EXTRA_PROCESS_FIELDS = ["deposition_target_pulses", "deposition_rep_rate", "deposition_energy",
                        "deposition_cycles", "deposition_ts_distance"]
LABEL_FIELDS = ["thickness", "opt_direct_bandgap", "fpm_conductivity"]
SCALAR_PROCESS_FIELDS = ["absolute_temp_c"]
SPATIAL_FIELDS = ["position", "x_mm", "y_mm"]
RAW_FIELDS = ["id", "sample_library_id", *SPATIAL_FIELDS, *SCALAR_PROCESS_FIELDS, *RECIPE_FIELDS,
              *EXTRA_PROCESS_FIELDS, *LABEL_FIELDS]
SCALAR_BLOCK_FIELDS = ["sample_library_id", *SPATIAL_FIELDS, *SCALAR_PROCESS_FIELDS, *LABEL_FIELDS]
UNITS = {"thickness": "um", "opt_direct_bandgap": "eV", "fpm_conductivity": "S/cm",
         "x_mm": "mm", "y_mm": "mm"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def http_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "AVS-2026-HTEM-scalar-workshop/2.0"
    return session


def compact_library(value):
    """Strip people/contact information and all non-process payloads before caching."""
    sample_ids = value.get("sample_ids") or []
    return {"sample_library_id": int(value["id"]),
            "expected_sample_count": sum(v is not None for v in sample_ids),
            "null_sample_id_placeholders": sum(v is None for v in sample_ids),
            **{key: value.get(key) for key in RECIPE_FIELDS + EXTRA_PROCESS_FIELDS}}


def compact_block(value):
    return {key: value.get(key) for key in SCALAR_BLOCK_FIELDS}


def expand_scalar_block(library, block):
    """Position arrays are aligned by the API, never by unordered sample_ids."""
    library_id = int(library["sample_library_id"])
    if int(block["sample_library_id"]) != library_id:
        raise ValueError("Scalar-property/library mismatch.")
    positions = block.get("position") or []
    if len(set(positions)) != len(positions) or any(p is None for p in positions):
        raise ValueError("Missing/duplicate positions within a library; cannot align labels safely.")
    for key in SPATIAL_FIELDS + SCALAR_PROCESS_FIELDS + LABEL_FIELDS:
        values = block.get(key)
        if values is not None and len(values) != len(positions):
            raise ValueError(f"Misaligned scalar array: {key} in library {library_id}.")
    records = []
    for index, position in enumerate(positions):
        record = {key: library.get(key) for key in RECIPE_FIELDS + EXTRA_PROCESS_FIELDS}
        # This is an explicit composite observation key, NOT the /api/sample numeric ID.
        record.update(id=f"{library_id}:{position}", sample_library_id=library_id, position=position)
        for key in ["x_mm", "y_mm", *SCALAR_PROCESS_FIELDS, *LABEL_FIELDS]:
            values = block.get(key)
            record[key] = values[index] if values is not None else None
        records.append(record)
    return records


def download_htem(output, cache_dir, library_ids=None, api_base=API_BASE,
                  delay=0.1, session=None, workers=4, refresh=False):
    """All public libraries, including missing labels. Compact caches only.

    Two documented API resources are used: library recipe details, and bulk
    /sample_library/prop?ids=... scalar maps. Exceptional libraries unsupported
    by that endpoint use individual samples, stripped IN MEMORY before caching.
    Instrument xyz_mm is never substituted for missing substrate coordinates.
    Per-library checkpoints make reruns resumable. All failures and
    coverage gaps are recorded; incomplete data are never presented as complete.
    """
    output, cache_dir = Path(output), Path(cache_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if workers < 1 or workers > 8:
        raise ValueError("Use 1–8 workers to avoid excessive public API load.")
    local = threading.local()

    def get_session():
        if session is not None:
            return session
        if not hasattr(local, "session"):
            local.session = http_session()
        return local.session

    def get(endpoint):
        response = get_session().get(f"{api_base.rstrip('/')}/{endpoint}", timeout=(10, 90))
        response.raise_for_status()
        value = response.json()
        if delay:
            time.sleep(delay)
        return value

    # Refresh public discovery every run; only a compact ID inventory is retained.
    listing = get("sample_library")
    if not isinstance(listing, list):
        raise ValueError("Unexpected public library-list schema.")
    public_ids = sorted(set(int(v["id"] if isinstance(v, dict) else v) for v in listing))
    requested = public_ids if library_ids is None else sorted(set(int(v) for v in library_ids))
    write_json(cache_dir / "public_library_ids.json", public_ids)
    missing_requested = set(requested) - set(public_ids)
    if missing_requested:
        raise ValueError(f"Requested libraries not in public inventory: {sorted(missing_requested)}")
    failures, libraries, records, fallback_libraries = [], [], [], []

    def metadata(library_id):
        path = cache_dir / f"library_{library_id}.json"
        if path.exists() and not refresh:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if "null_sample_id_placeholders" in cached or cached["expected_sample_count"] != 1:
                return cached
        value = get(f"sample_library/{library_id}")
        if int(value["id"]) != library_id:
            raise ValueError("Wrong library in API detail response.")
        compact = compact_library(value)
        write_json(path, compact)
        return compact

    with ThreadPoolExecutor(max_workers=1 if session is not None else workers) as pool:
        future_ids = {pool.submit(metadata, library_id): library_id for library_id in requested}
        for count, future in enumerate(as_completed(future_ids), 1):
            library_id = future_ids[future]
            try:
                libraries.append(future.result())
            except Exception as exc:
                failures.append({"library_id": library_id, "stage": "recipe", "error": str(exc)})
            if count == 1 or count % 100 == 0 or count == len(requested):
                print(f"Recipes checked {count}/{len(requested)}", flush=True)
    libraries.sort(key=lambda v: v["sample_library_id"])
    catalog = pd.DataFrame(libraries)
    catalog_path = output.with_name("htem_libraries.parquet")
    if not catalog.empty:
        catalog.to_parquet(catalog_path, index=False)

    scalar_failures = []

    def current_scalar_cache(path):
        # Refresh old compact caches once when a new process field is introduced.
        return path.exists() and all(k in json.loads(path.read_text(encoding="utf-8"))
                                     for k in SCALAR_PROCESS_FIELDS)
    # Batch scalar downloads to avoid one large response or tens of thousands of sample requests.
    for start in range(0, len(libraries), 25):
        batch = libraries[start:start + 25]
        needed = [v for v in batch if v["expected_sample_count"] > 0 and
                  (refresh or not current_scalar_cache(cache_dir / f"properties_{v['sample_library_id']}.json"))]
        if needed:
            ids = [v["sample_library_id"] for v in needed]
            try:
                blocks = get("sample_library/prop?ids=" + ",".join(map(str, ids)))
                if not isinstance(blocks, list):
                    raise ValueError("Unexpected scalar-property batch schema.")
                for block in blocks:
                    lib = int(block["sample_library_id"])
                    if lib not in ids:
                        raise ValueError("Unexpected library in scalar-property batch.")
                    write_json(cache_dir / f"properties_{lib}.json", compact_block(block))
            except Exception as exc:
                scalar_failures.append({"library_ids": ids, "error": str(exc)})
        for library in batch:
            library_id = library["sample_library_id"]
            if not library["expected_sample_count"]:
                continue  # Recipes with [null] sample IDs are retained in the catalog, not fake map rows.
            path = cache_dir / f"properties_{library_id}.json"
            fallback_path = cache_dir / f"fallback_samples_{library_id}.json"
            try:
                fallback_current = (fallback_path.exists() and all(
                    all(k in row for k in SCALAR_PROCESS_FIELDS)
                    for row in json.loads(fallback_path.read_text(encoding="utf-8"))))
                if fallback_current and not refresh:
                    records.extend(json.loads(fallback_path.read_text(encoding="utf-8")))
                    fallback_libraries.append(library_id)
                    continue
                if not path.exists():
                    # Retry a missing batch result individually, still without full spectra.
                    blocks = get(f"sample_library/prop?ids={library_id}")
                    if not isinstance(blocks, list) or len(blocks) != 1:
                        raise ValueError("Missing individual scalar-property block.")
                    write_json(path, compact_block(blocks[0]))
                block = json.loads(path.read_text(encoding="utf-8"))
                records.extend(expand_scalar_block(library, block))
            except Exception as exc:
                try:
                    # Rare legacy libraries have no functioning scalar-map endpoint.
                    # Fetch only these individual responses, keep NO spectra on disk.
                    detail = get(f"sample_library/{library_id}")
                    fallback, fallback_errors = [], []
                    for sample_id in detail["sample_ids"]:
                        if sample_id is None:
                            continue
                        try:
                            sample_path = cache_dir / f"fallback_scalar_{sample_id}.json"
                            if current_scalar_cache(sample_path) and not refresh:
                                fallback.append(json.loads(sample_path.read_text(encoding="utf-8")))
                                continue
                            sample = get(f"sample/{int(sample_id)}")
                            if int(sample["sample_library_id"]) != library_id or int(sample["id"]) != int(sample_id):
                                raise ValueError("Fallback sample identity mismatch.")
                            position = sample.get("position")
                            record = {
                                "id": f"{library_id}:{position}" if position is not None else f"{library_id}:sample_{sample_id}",
                                "sample_library_id": library_id, "position": position,
                                "x_mm": None, "y_mm": None,
                                **{key: library.get(key) for key in RECIPE_FIELDS + EXTRA_PROCESS_FIELDS},
                                **{key: sample.get(key) for key in SCALAR_PROCESS_FIELDS + LABEL_FIELDS},
                            }
                            write_json(sample_path, record)
                            fallback.append(record)
                        except Exception as sample_error:
                            fallback_errors.append({"sample_id": sample_id, "error": str(sample_error)})
                    if not fallback_errors:
                        write_json(fallback_path, fallback)
                    records.extend(fallback)
                    if fallback:
                        fallback_libraries.append(library_id)
                    if fallback_errors:
                        failures.append({"library_id": library_id, "stage": "individual_scalar_fallback",
                                         "recovered_points": len(fallback), "unavailable_samples": fallback_errors})
                except Exception as fallback_error:
                    failures.append({"library_id": library_id, "stage": "scalar_grid_and_fallback",
                                     "error": str(exc), "fallback_error": str(fallback_error)})
        print(f"Scalar maps checked {min(start + 25, len(libraries))}/{len(libraries)}; points {len(records):,}", flush=True)
        write_json(output.with_name("download_progress.json"), {
            "requested_libraries": len(requested), "recipes_retained": len(libraries),
            "scalar_libraries_checked": min(start + 25, len(libraries)), "points": len(records),
            "failures": failures, "running": True,
        })
    frame = pd.DataFrame(records, columns=RAW_FIELDS).sort_values(["sample_library_id", "position"]).reset_index(drop=True)
    if frame.empty or frame.id.duplicated().any():
        raise ValueError("No scalar points or duplicate composite keys; final dataset not replaced.")
    frame.to_parquet(output, index=False)
    returned = set(frame.sample_library_id)
    expected = {v["sample_library_id"]: v["expected_sample_count"] for v in libraries}
    counts = frame.groupby("sample_library_id").size().to_dict()
    gaps = [{"library_id": int(lib), "expected_samples": count, "scalar_positions": int(counts.get(lib, 0))}
            for lib, count in expected.items() if count != counts.get(lib, 0)]
    report = {
        "mode": "live_scalar_api", "api_base": api_base,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_libraries": len(public_ids), "requested_libraries": len(requested),
        "recipe_libraries": len(libraries), "libraries_with_scalar_rows": len(returned),
        "rows": len(frame), "complete": not failures and len(libraries) == len(requested) and not gaps,
        "failures": failures, "sample_count_gaps": gaps, "recovered_batch_errors": scalar_failures,
        "fallback_libraries_without_substrate_coordinates": fallback_libraries,
        "rows_missing_substrate_coordinates": int(frame[["x_mm", "y_mm"]].isna().any(axis=1).sum()),
        "libraries_without_scalar_rows": sorted(set(requested) - returned),
        "libraries_with_no_valid_sample_ids": [v["sample_library_id"] for v in libraries if not v["expected_sample_count"]],
        "sha256": sha256(output), "library_catalog_sha256": sha256(catalog_path),
        "label_units": UNITS, "fields": RAW_FIELDS,
        "temperature_source": "API absolute_temp_c per position; distinct from library initial temperature; no trajectory inferred",
        "coordinate_source": "API scalar-property x_mm/y_mm (substrate grid), NOT sample xyz_mm stage/focus",
        "observation_key": "id is the composite library:position key, not the numeric /api/sample ID",
        "cache_policy": "Allowlisted process/scalar arrays only; fallback sample spectra discarded in memory, never cached",
        "note": "All discovered public libraries requested, no has_opt/has_ele/complete-target filter; cached responses can predate this run.",
    }
    write_json(output.with_suffix(".provenance.json"), report)
    write_json(output.with_name("download_progress.json"), {**report, "running": False})
    print(f"Saved {len(frame):,} points; complete coverage: {report['complete']}; failures: {len(failures)}", flush=True)
    return frame


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parents[1]
    download_htem(here / "outputs/htem_acquisition/htem_process_raw.parquet",
                  here / "outputs/htem_acquisition/compact_http_cache_v2",
                  workers=args.workers, refresh=args.refresh)
