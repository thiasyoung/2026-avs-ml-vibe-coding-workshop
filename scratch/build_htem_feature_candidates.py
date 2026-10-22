"""Instructor-only external feature extraction; no HTEM outcomes are read here.

Run with the isolated extractor environment documented in the study notebook.
Participant preparation only needs the resulting compact JSON, not DGL/ALIGNN.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "03_htem_conductivity"
# The isolated environment supplies DGL/JARVIS; reuse the locked workshop's
# numerical packages without installing or altering anything in that environment.
for site in [ROOT / ".venv/Lib/site-packages", ROOT / ".venv/lib/python3.11/site-packages"]:
    if site.exists():
        sys.path.append(str(site))
sys.path.insert(0, str(MODULE))

import numpy as np
import pandas as pd
import requests

import htem_chemistry as chemistry
import htem_source_dos as dos

CACHE = MODULE / "outputs/htem_feature_study/reference_cache"
DESTINATION = MODULE / "data/htem_feature_candidates.json"
MAGPIE_COMMIT = "bd1386e2412b3367b8e11388d02092524a6dafcb"
MEGNET_COMMIT = "f3705760266f24b62c4810dd4abacfc25f4f64ae"
MODEL_URL = "https://figshare.com/ndownloader/files/31458679"
MODEL_SHA256 = "d80f5e067c0b0b0f569b8158ecfeeeb68472bebe2ceb27ff787dcfcc68948e20"
TABLES = ["Column", "Row", "MendeleevNumber", "CovalentRadius", "MeltingT",
          "NsValence", "NpValence", "NdValence", "NfValence", "NValence", "IsDBlock"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(url, name):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists() or path.stat().st_size == 0:
        print("Downloading", name, flush=True)
        response = requests.get(url, timeout=(15, 180))
        response.raise_for_status()
        if not response.content:
            raise RuntimeError(f"Empty download: {url}; retry, not a valid cached reference.")
        path.write_bytes(response.content)
    return path


def composition_reference(formulas):
    """Curated Magpie statistics and genuine formation-energy-trained vectors.

    MEGNet's published 16-D ELEMENT vectors are fraction-averaged. This is a
    composition embedding, NOT a Roost/AtomSets network or crystal embedding.
    """
    provenance, tables = {}, {}
    for name in TABLES:
        url = (f"https://raw.githubusercontent.com/hackingmaterials/matminer/{MAGPIE_COMMIT}/"
               f"src/matminer/utils/data_files/magpie_elementdata/{name}.table")
        path = fetch(url, name + ".table")
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                values.append(float(line.strip()))
            except ValueError:
                values.append(float("nan"))
        tables[name] = values
        provenance[name] = {"url": url, "sha256": digest(path)}
    url = (f"https://raw.githubusercontent.com/materialsvirtuallab/megnet/{MEGNET_COMMIT}/"
           "megnet/data/resources/elemental_embedding_1MEGNet_layer.json")
    path = fetch(url, "megnet_element_vectors.json")
    vectors = json.loads(path.read_text(encoding="utf-8"))
    provenance["composition_embedding"] = {
        "url": url, "sha256": digest(path), "dimension": 16,
        "method": "stoichiometric mean of frozen MEGNet formation-energy-trained element embeddings; no message passing at extraction",
        "paper": "https://doi.org/10.1021/acs.chemmater.9b01294",
    }
    result = {}
    for formula in formulas:
        parsed = chemistry.parse_formula(formula)
        weights = np.array([float(f) for _, f in parsed])
        z = [int(chemistry.reference_data()["elements"][e]["atomic_number"]) for e, _ in parsed]
        descriptor = {}
        for name in TABLES:
            values = np.array([tables[name][index - 1] for index in z])
            if not np.isfinite(values).all():
                raise ValueError(f"Missing {name} for {formula}; do not invent values.")
            descriptor["magpie_mean_" + name] = float(weights @ values)
            if name not in ["IsDBlock", "NsValence", "NpValence", "NdValence", "NfValence", "NValence"]:
                descriptor["magpie_range_" + name] = float(np.ptp(values))
                descriptor["magpie_std_" + name] = float(np.sqrt(weights @ (values - weights @ values) ** 2))
        valence = descriptor["magpie_mean_NValence"]
        for orbital in "spdf":
            descriptor[f"valence_fraction_{orbital}"] = descriptor[f"magpie_mean_N{orbital}Valence"] / valence
        descriptor["stoichiometry_L2"] = float(np.linalg.norm(weights))
        descriptor["stoichiometry_L3"] = float(np.linalg.norm(weights, ord=3))
        embedding = weights @ np.array([vectors[e] for e, _ in parsed], dtype=float)
        if embedding.shape != (16,) or not np.isfinite(embedding).all():
            raise ValueError("Invalid published MEGNet element embedding.")
        result[chemistry.composition_key(formula)] = {
            "curated": descriptor, "composition_embedding": embedding.tolist(),
        }
    return result, provenance


def write_notices():
    """Retain full upstream redistribution notices with the compact references."""
    sources = {
        "matminer": ("hackingmaterials/matminer", MAGPIE_COMMIT, "LICENSE"),
        "megnet": ("materialsvirtuallab/megnet", MEGNET_COMMIT, "LICENSE.md"),
        "alignn": ("usnistgov/alignn", "f2366daa3413d28a825b46e34d001b5549b05a40", "LICENSE.rst"),
    }
    notices = {}
    for name, (repo, revision, filename) in sources.items():
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{filename}"
        path = fetch(url, name + "_LICENSE.txt")
        notices[name] = {"url": url, "sha256": digest(path), "full_notice": path.read_text(encoding="utf-8")}
    path = MODULE / "data/htem_feature_reference_licenses.json"
    path.write_text(json.dumps(notices, indent=2) + "\n", encoding="utf-8")


def write_element_reference():
    """Small pinned elemental tables for the participant's worked calculation."""
    elements = {e: {} for e in chemistry.reference_data()["elements"]}
    provenance = {}
    for name in TABLES:
        url = (f"https://raw.githubusercontent.com/hackingmaterials/matminer/{MAGPIE_COMMIT}/"
               f"src/matminer/utils/data_files/magpie_elementdata/{name}.table")
        path = fetch(url, name + ".table")
        lines = path.read_text(encoding="utf-8").splitlines()
        provenance[name] = {"url": url, "sha256": digest(path)}
        for element in elements:
            z = int(chemistry.reference_data()["elements"][element]["atomic_number"])
            try:
                value = float(lines[z - 1])
                elements[element][name] = value if np.isfinite(value) else None
            except (ValueError, IndexError):
                elements[element][name] = None
    payload = {"provenance": provenance, "elements": elements, "property_order": TABLES,
               "units": {"MeltingT": "K (elemental, not compound)", "CovalentRadius": "pm",
                         "valence_counts": "electrons per atom", "others": "dimensionless"}}
    (MODULE / "data/htem_feature_element_reference.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def crystal_reference(allowed_keys):
    """Frozen official ALIGNN; average ALL exact-composition reference graphs."""
    import torch
    from alignn.models.alignn import ALIGNN, ALIGNNConfig
    from alignn.graphs import Graph
    from jarvis.core.atoms import Atoms

    torch.set_num_threads(4)
    model_path = fetch(MODEL_URL, "jv_formation_energy_peratom_alignn.zip")
    if digest(model_path) != MODEL_SHA256:
        raise ValueError("Pretrained ALIGNN download differs from pinned upstream archive.")
    with zipfile.ZipFile(model_path) as archive:
        config_name = next(n for n in archive.namelist() if n.endswith("config.json"))
        checkpoint_names = sorted(n for n in archive.namelist() if n.endswith(".pt") and "checkpoint_" in n)
        best = [n for n in archive.namelist() if n.endswith("best_model.pt")]
        if len(best) == 1:
            checkpoint_name = best[0]
        elif len(checkpoint_names) == 1:
            checkpoint_name = checkpoint_names[0]
        else:
            raise ValueError("Ambiguous model checkpoint; select from upstream validation metadata, not HTEM scores.")
        config = json.loads(archive.read(config_name))
        model = ALIGNN(ALIGNNConfig(**config["model"]))
        # The published 2021 training checkpoint includes optimizer/scheduler
        # Python objects, not just tensors. Its exact upstream archive is pinned
        # above. Load only in this instructor extraction environment, never from
        # user-provided paths; participants receive plain JSON vectors only.
        weights = torch.load(io.BytesIO(archive.read(checkpoint_name)), map_location="cpu", weights_only=False)
        model.load_state_dict(weights["model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    captured = []
    hook = model.fc.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().cpu().numpy().copy()))
    raw_archive = MODULE / "outputs/htem_source_reference_cache/jarvis_edos_29216859.zip"
    if not raw_archive.exists() or digest(raw_archive) != "151592ed0502ec956f6ce7bcb1eee6435287cac7248e1549e78d7e1a9aeddcc4":
        raise ValueError("Missing/changed pinned JARVIS archive. Run the DOS reference builder first.")
    by_key, failures, count = defaultdict(list), [], 0
    checkpoints = CACHE / "alignn_structure_embeddings.json"
    existing = json.loads(checkpoints.read_text(encoding="utf-8")) if checkpoints.exists() else {}
    with zipfile.ZipFile(raw_archive) as archive, archive.open(dos.MEMBER) as member:
        stream = io.TextIOWrapper(member, encoding="utf-8")
        for record in dos.iter_json_array(stream):
            key = dos.composition_from_atoms(record["atoms"])
            if key not in allowed_keys:
                continue
            jid = str(record["jid"])
            try:
                if jid in existing:
                    vector = np.asarray(existing[jid]["embedding"])
                else:
                    atoms = Atoms.from_dict(record["atoms"])
                    lattice = np.asarray(atoms.lattice_mat, dtype=float)
                    if lattice.shape != (3, 3) or not np.isfinite(lattice).all() or abs(np.linalg.det(lattice)) < 1e-8:
                        raise ValueError("Invalid lattice.")
                    coords = np.asarray(atoms.frac_coords, dtype=float)
                    if coords.shape != (len(atoms.elements), 3) or not np.isfinite(coords).all():
                        raise ValueError("Invalid atomic coordinates.")
                    graph, line = Graph.atom_dgl_multigraph(atoms, cutoff=8., max_neighbors=12,
                                                           atom_features="cgcnn", compute_line_graph=True)
                    captured.clear()
                    with torch.inference_mode():
                        output = model([graph, line, torch.tensor(lattice, dtype=torch.float32)])
                    vector = captured[-1].reshape(-1)
                    # Confirm this is exactly the pre-head representation.
                    with torch.inference_mode():
                        reconstructed = model.fc(torch.from_numpy(vector).reshape(1, -1))
                    np.testing.assert_allclose(output.detach().numpy().reshape(-1), reconstructed.numpy().reshape(-1), rtol=1e-5, atol=1e-6)
                    existing[jid] = {"composition": key, "embedding": vector.tolist(),
                                     "structure_sha256": hashlib.sha256(json.dumps(record["atoms"], sort_keys=True).encode()).hexdigest()}
                    if len(existing) % 20 == 0:
                        checkpoints.write_text(json.dumps(existing), encoding="utf-8")
                if vector.shape != (256,) or not np.isfinite(vector).all():
                    raise ValueError("Expected finite 256-D pre-head ALIGNN embedding.")
                by_key[key].append((jid, vector))
                count += 1
                if count % 25 == 0:
                    print(f"ALIGNN: {count} reference structures / {len(by_key)} compositions", flush=True)
            except Exception as error:
                failures.append({"jid": jid, "composition": key, "error": str(error)})
    hook.remove()
    checkpoints.write_text(json.dumps(existing), encoding="utf-8")
    result = {}
    for key, items in by_key.items():
        matrix = np.stack([v for _, v in items])
        result[key] = {"mean": matrix.mean(axis=0).tolist(),
                       "mean_coordinate_std": float(matrix.std(axis=0).mean()),
                       "reference_ids": [jid for jid, _ in items]}
    if failures:
        print("Explicit structure extraction failures:", len(failures), flush=True)
    if not result:
        raise RuntimeError("No real atomic-graph embeddings extracted.")
    return result, {
        "url": MODEL_URL, "archive_sha256": digest(model_path), "checkpoint": checkpoint_name,
        "model": config["model"], "dimension": 256,
        "pooling": "official mean atom readout immediately before final scalar head; mean over all matching reference structures",
        "graph": {"cutoff_A": 8., "max_neighbors": 12, "atom_features": "cgcnn", "line_graph": True},
        "structure_archive_sha256": digest(raw_archive), "structure_archive_url": dos.ARCHIVE_URL,
        "packages": {name: importlib.metadata.version(name) for name in ["alignn", "jarvis-tools", "dgl", "torch"]},
        "reference_structures": count, "failures": failures,
        "paper": "https://doi.org/10.1038/s41524-021-00650-1",
        "caveat": "reference crystal structures, not verified phases of HTEM source targets or films",
    }


def main():
    write_notices()
    write_element_reference()
    libraries = MODULE / "data/htem_spatial_libraries.parquet"
    catalog = pd.read_parquet(libraries, columns=["deposition_compounds"])
    formulas = sorted({str(v) for values in catalog.deposition_compounds if values is not None for v in values if v})
    for formula in formulas:
        chemistry.parse_formula(formula)
    composition, provenance = composition_reference(formulas)
    crystals, crystal_provenance = crystal_reference(set(composition))
    for key, values in composition.items():
        values["crystal_embedding"] = crystals.get(key)
    payload = {"schema_version": 1, "catalog_sha256": digest(libraries), "formulas": formulas,
               "provenance": {**provenance, "crystal_embedding": crystal_provenance}, "compositions": composition}
    DESTINATION.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(composition)} source compositions; {len(crystals)} with real atomic embeddings: {DESTINATION}", flush=True)


if __name__ == "__main__":
    main()
