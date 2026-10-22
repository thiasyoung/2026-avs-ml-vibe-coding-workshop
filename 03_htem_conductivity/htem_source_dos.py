"""Curate independent source-composition DOS references, never HTEM film spectra.

The shipped compact reference is used offline. The explicit builder streams the
public JARVIS archive once; no API key, crystal-phase assignment, or ML is needed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests

ARCHIVE_URL = "https://ndownloader.figshare.com/files/29216859"
ARCHIVE_MD5 = "b6005174c1ea87c12958658a533bf28f"
ARCHIVE_BYTES = 154538553
MEMBER = "edos-up_pdos-elast_interp-8-18-2021.json"
# 300 bins at the paper's 0.05 eV spacing: -5 inclusive, 10 exclusive.
# linspace(-5, 10, 300) would silently stretch the grid to 0.050167 eV.
ENERGY_EV = -5. + .05 * np.arange(300)
BIN_EDGES_EV = np.array([-5., -3., -1., 0., 1., 3., 6., ENERGY_EV[-1]])
SHAPE_FIELDS = ["centroid_eV", "width_eV", "occupied_fraction", "near_fermi_fraction",
                *[f"bin_{i}_fraction" for i in range(len(BIN_EDGES_EV) - 1)]]
DOS_PROPERTIES = ["reference_dos_known", "reference_dos_records",
                  *["reference_dos_" + field for field in SHAPE_FIELDS],
                  "reference_dos_near_fermi_range", "reference_dos_mean_bin_std"]
REFERENCE = Path(__file__).with_name("data") / "htem_source_dos_reference.json"


def iter_json_array(stream, chunk_size=1 << 20):
    """Read a strict top-level JSON array without retaining its entire payload."""
    decoder = json.JSONDecoder()
    buffer, index, eof = "", 0, False

    def next_nonspace():
        nonlocal buffer, index, eof
        while True:
            while index < len(buffer) and buffer[index].isspace():
                index += 1
            if index < len(buffer) or eof:
                return buffer[index:index + 1]
            buffer, index = "", 0
            part = stream.read(chunk_size)
            eof = not part
            buffer += part

    if next_nonspace() != "[":
        raise ValueError("Expected a JSON array.")
    index += 1
    if next_nonspace() == "]":
        index += 1
    else:
        while True:
            while True:
                try:
                    record, end = decoder.raw_decode(buffer, index)
                    if not eof and (end == len(buffer) or buffer[end] not in ",] \t\r\n"):
                        buffer, index = buffer[index:], 0
                        part = stream.read(chunk_size)
                        eof = not part
                        buffer += part
                        continue
                    index = end
                    break
                except json.JSONDecodeError as error:
                    if eof:
                        raise ValueError("Truncated/invalid JSON record.") from error
                    buffer, index = buffer[index:], 0
                    part = stream.read(chunk_size)
                    eof = not part
                    buffer += part
            yield record
            delimiter = next_nonspace()
            if delimiter == "]":
                index += 1
                break
            if delimiter != ",":
                raise ValueError("Expected ',' or ']' after JSON record.")
            index += 1
            next_nonspace()
    if next_nonspace():
        raise ValueError("Unexpected content after JSON array.")


def composition_from_atoms(atoms):
    """Exact reduced composition; atom coordinates are NOT an HTEM phase match."""
    elements = atoms.get("elements", [])
    if not elements or any(not isinstance(e, str) or not e for e in elements):
        raise ValueError("Missing atom species.")
    counts = Counter(elements)
    return ";".join(f"{element}:{Fraction(count, len(elements))}"
                    for element, count in sorted(counts.items()))


def spectral_descriptors(values):
    """Unit-area spin-up DOS shape on the fixed -5..9.95 eV sampled support.

    Fractions integrate a piecewise-linear spectrum with exact window endpoints.
    Tiny interpolation roundoff negatives (>= -1e-8) are clipped; larger ones are
    rejected, not repaired. These are not absolute DOS, carrier density, or gaps.
    """
    density = np.asarray(values, dtype=float)
    if density.shape != ENERGY_EV.shape or not np.isfinite(density).all() or density.min() < -1e-8:
        raise ValueError("DOS requires 300 finite, nonnegative bins.")
    density = np.maximum(density, 0.)
    area = float(np.trapezoid(density, ENERGY_EV))
    if area <= 0:
        raise ValueError("Zero-integral DOS is not a known reference.")
    density = density / area

    def integral(lo, hi):
        x = np.r_[lo, ENERGY_EV[(ENERGY_EV > lo) & (ENERGY_EV < hi)], hi]
        return float(np.trapezoid(np.interp(x, ENERGY_EV, density), x))

    mean = float(np.trapezoid(ENERGY_EV * density, ENERGY_EV))
    width = float(np.sqrt(np.trapezoid((ENERGY_EV - mean) ** 2 * density, ENERGY_EV)))
    return dict(centroid_eV=mean, width_eV=width, occupied_fraction=integral(-5., 0.),
                near_fermi_fraction=integral(-.5, .5),
                **{f"bin_{i}_fraction": integral(lo, hi)
                   for i, (lo, hi) in enumerate(zip(BIN_EDGES_EV[:-1], BIN_EDGES_EV[1:]))})


def curate_records(records, allowed_compositions):
    """Aggregate ALL matching reference structures, without choosing a phase."""
    matched, seen, rejected, counts = defaultdict(list), set(), Counter(), Counter()
    for record in records:
        counts["external_records"] += 1
        try:
            key = composition_from_atoms(record.get("atoms", {}))
        except (ValueError, TypeError):
            rejected["missing_composition"] += 1
            continue
        if key not in allowed_compositions:
            continue
        jid = record.get("jid")
        if not isinstance(jid, str) or not jid or jid in seen:
            raise ValueError("Matching DOS records need unique JARVIS identifiers.")
        seen.add(jid)
        counts["composition_matched_records"] += 1
        try:
            features = spectral_descriptors(record.get("edos_up"))
        except (ValueError, TypeError):
            rejected["invalid_matched_dos"] += 1
            continue
        matched[key].append((jid, features))
    output = {}
    for key, rows in sorted(matched.items()):
        table = pd.DataFrame([r[1] for r in rows])
        # Arithmetic means preserve the fact that the seven bin fractions sum to 1.
        values = {"reference_dos_" + f: float(table[f].mean()) for f in SHAPE_FIELDS}
        values.update(reference_dos_known=1., reference_dos_records=len(rows),
                      reference_dos_near_fermi_range=float(np.ptp(table.near_fermi_fraction)),
                      reference_dos_mean_bin_std=float(table[[f"bin_{i}_fraction" for i in range(7)]].std(ddof=0).mean()))
        output[key] = {"features": values, "reference_ids": sorted(jid for jid, _ in rows)}
    return output, {**counts, "rejected": dict(rejected)}


def fetch_archive(cache):
    """Pinned public archive, streamed to an ignored cache; reject corrupt data."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "jarvis_edos_29216859.zip"
    if not path.exists():
        temporary = path.with_suffix(".download")
        print("Downloading public JARVIS DOS reference (155 MB; cached for reuse)...", flush=True)
        with requests.get(ARCHIVE_URL, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1 << 20):
                    handle.write(chunk)
        if temporary.stat().st_size != ARCHIVE_BYTES or hashlib.md5(temporary.read_bytes()).hexdigest() != ARCHIVE_MD5:
            raise ValueError("DOS archive differs from pinned Figshare checksum; no reference generated.")
        temporary.replace(path)
    if path.stat().st_size != ARCHIVE_BYTES or hashlib.md5(path.read_bytes()).hexdigest() != ARCHIVE_MD5:
        raise ValueError("Cached DOS archive failed its pinned checksum.")
    return path


def build_reference(catalog, cache, output=REFERENCE):
    """Select by source catalog formulas only; never inspect film labels or splits."""
    import htem_chemistry as chemistry
    catalog = Path(catalog)
    labels = sorted({str(label).strip() for row in pd.read_parquet(catalog).deposition_compounds
                     if row is not None for label in row if label is not None and str(label).strip()})
    compositions, unsupported = set(), []
    for label in labels:
        try:
            compositions.add(chemistry.composition_key(label))
        except ValueError:
            unsupported.append(label)
    archive = fetch_archive(cache)
    print("Extracting composition-matched DOS summaries; no phase is assigned to HTEM targets...", flush=True)
    with zipfile.ZipFile(archive) as zipped, zipped.open(MEMBER) as raw, io.TextIOWrapper(raw, encoding="utf-8") as stream:
        references, coverage = curate_records(iter_json_array(stream), compositions)
    result = {"version": 1, "compositions": references,
              "energy_grid_eV": ENERGY_EV.tolist(), "bin_edges_eV": BIN_EDGES_EV.tolist(),
              "provenance": {"archive_url": ARCHIVE_URL, "archive_md5": ARCHIVE_MD5,
                  "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                  "dataset_doi": "https://doi.org/10.6084/m9.figshare.14745327.v3",
                  "paper": "https://doi.org/10.1007/s11837-022-05199-y",
                  "authors": "Prathik R. Kaundinya, Kamal Choudhary, Surya R. Kalidindi",
                  "license": "CC BY 4.0", "member": MEMBER,
                  "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
                  "source_labels": labels, "unsupported_labels": unsupported, **coverage,
                  "channel": "edos_up only; normalized electronic spin-up DOS, not phonons or spin-summed DOS",
                  "processing": "300 samples: -5 inclusive to 10 exclusive, spacing 0.05 eV relative to Fermi; integrals restricted to sampled support [-5, 9.95]; unit-area normalization; fixed integrals/moments; arithmetic mean across all valid composition-matched structures",
                  "limitations": "Reference structures are NOT verified source phases; no guessed structure, phase selection, graph embedding, carrier density, measured HTEM film data, or inferred mixture DOS. Missing compositions remain missing; across-reference spread is not calibrated uncertainty."}}
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(references)} composition references to {output}", flush=True)
    return result


if __name__ == "__main__":
    module = Path(__file__).resolve().parent
    build_reference(module / "data/htem_spatial_libraries.parquet", module / "outputs/htem_source_reference_cache")
