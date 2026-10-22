"""Offline descriptors of source formulas, never measured deposited-film data."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import htem_source_dos as dos
import htem_features as frozen_features

REFERENCE = Path(__file__).with_name("data") / "htem_chemistry_reference.json"
SOURCE_PROPERTIES = [
    "present", "formula_known", "element_count", "composition_entropy",
    "mean_atomic_number", "mean_atomic_mass_u", "mean_electronegativity",
    "range_electronegativity", "std_electronegativity", "mean_atomic_radius_A",
    "mean_first_ionization_eV", *[f"fraction_{e}" for e in ["O", "N", "S", "Se", "Te"]],
    "reference_bandgap_eV", "reference_bandgap_range_eV", "bandgap_reference_known",
    *dos.DOS_PROPERTIES,
]
AGGREGATE_PROPERTIES = [
    "source_total_power_W", "source_formula_power_coverage", "source_bandgap_power_coverage",
    "power_weighted_electronegativity", "power_weighted_atomic_mass_u",
    "power_weighted_first_ionization_eV", "power_weighted_reference_bandgap_eV",
    "source_dos_power_coverage", "power_weighted_reference_dos_centroid_eV",
    "power_weighted_reference_dos_near_fermi_fraction",
]
CHEMISTRY_FEATURES = [f"target_{i}_{name}" for i in range(1, 4) for name in SOURCE_PROPERTIES]
CHEMISTRY_FEATURES += [f"target_{i}_power_fraction" for i in range(1, 4)] + AGGREGATE_PROPERTIES


@lru_cache(maxsize=1)
def reference_data():
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def dos_reference_data():
    return json.loads(dos.REFERENCE.read_text(encoding="utf-8"))


def reference_fingerprint():
    return {"reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
            "featurizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dos_reference_sha256": hashlib.sha256(dos.REFERENCE.read_bytes()).hexdigest(),
            "dos_featurizer_sha256": hashlib.sha256(Path(dos.__file__).read_bytes()).hexdigest(),
            **frozen_features.fingerprint()}


@lru_cache(maxsize=1024)
def parse_formula(formula):
    """Strict case-sensitive formula parser: nested parentheses/decimal amounts.

    No guessed names, isotope/charge syntax, hydrates, trade names, or variable x.
    Fraction arithmetic makes ZnO, OZn and Zn2O2 composition-equivalent.
    """
    tokens = re.findall(r"[A-Z][a-z]?|\d+(?:\.\d+)?|[()]", formula)
    if not formula or "".join(tokens) != formula:
        raise ValueError(f"Unsupported source formula: {formula!r}")
    stack = [defaultdict(Fraction)]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "(":
            stack.append(defaultdict(Fraction))
            index += 1
            continue
        if token == ")":
            if len(stack) == 1 or not stack[-1]:
                raise ValueError("Unbalanced/empty formula parentheses.")
            amounts = stack.pop()
        elif token in reference_data()["elements"]:
            amounts = {token: Fraction(1)}
        else:
            raise ValueError(f"Unknown element or unsupported formula token: {token}")
        index += 1
        multiplier = Fraction(1)
        if index < len(tokens) and tokens[index][0].isdigit():
            multiplier = Fraction(tokens[index])
            index += 1
        if multiplier <= 0:
            raise ValueError("Element amounts must be positive.")
        for element, amount in amounts.items():
            stack[-1][element] += multiplier * amount
    if len(stack) != 1 or not stack[0]:
        raise ValueError("Unbalanced/empty formula.")
    total = sum(stack[0].values())
    return tuple(sorted((element, amount / total) for element, amount in stack[0].items()))


def composition_key(formula):
    return ";".join(f"{element}:{fraction}" for element, fraction in parse_formula(formula))


@lru_cache(maxsize=1024)
def source_descriptors(formula):
    values = {key: np.nan for key in SOURCE_PROPERTIES}
    values.update(present=float(formula != "__NONE__"), formula_known=0., bandgap_reference_known=0.,
                  reference_dos_known=0., reference_dos_records=0.)
    if formula == "__NONE__":
        # Absent source has zero composition contribution, not a zero-eV gap.
        values.update({key: 0. for key in SOURCE_PROPERTIES
                       if not key.startswith(("reference_bandgap", "reference_dos"))})
        return values
    try:
        parsed = parse_formula(formula)
    except (ValueError, TypeError):
        return values  # Unknown names are audited; no invented chemistry or gap.
    elements = reference_data()["elements"]
    fractions = np.array([float(f) for _, f in parsed])
    values.update(formula_known=1., element_count=float(len(parsed)),
                  composition_entropy=float(-np.sum(fractions * np.log(fractions))))
    for field, output in [("atomic_number", "atomic_number"), ("atomic_mass_u", "atomic_mass_u"),
                          ("electronegativity", "electronegativity"), ("atomic_radius_A", "atomic_radius_A"),
                          ("first_ionization_eV", "first_ionization_eV")]:
        props = np.array([elements[e].get(field, np.nan) for e, _ in parsed], dtype=float)
        if np.isfinite(props).all():
            mean = float(fractions @ props)
            values["mean_" + output] = mean
            if field == "electronegativity":
                values["range_electronegativity"] = float(np.ptp(props))
                values["std_electronegativity"] = float(np.sqrt(fractions @ (props - mean) ** 2))
    composition = dict(parsed)
    values.update({f"fraction_{e}": float(composition.get(e, 0)) for e in ["O", "N", "S", "Se", "Te"]})
    gap = reference_data()["bandgaps"].get(composition_key(formula))
    if gap is not None:
        values.update(reference_bandgap_eV=gap["median_eV"],
                      reference_bandgap_range_eV=gap["max_eV"] - gap["min_eV"], bandgap_reference_known=1.)
    dos_match = dos_reference_data()["compositions"].get(composition_key(formula))
    if dos_match is not None:
        values.update(dos_match["features"])
    return values


def recipe_descriptors(targets, powers):
    """Preserve each source/power slot; power weighting is NOT film composition."""
    source = [source_descriptors(t) for t in targets]
    result = {f"target_{i}_{key}": value for i, values in enumerate(source, 1) for key, value in values.items()}
    powers = np.asarray(powers, dtype=float)
    active = np.isfinite(powers) & (powers > 0)
    total = float(powers[active].sum())
    weights = np.where(active, powers, 0.) / total if total else np.zeros(3)
    result.update({f"target_{i}_power_fraction": float(w) for i, w in enumerate(weights, 1)})
    result["source_total_power_W"] = total
    for label, key in [("formula", "formula_known"), ("bandgap", "bandgap_reference_known"),
                       ("dos", "reference_dos_known")]:
        result[f"source_{label}_power_coverage"] = float(weights @ np.array([s[key] for s in source]))
    for output, field in [("electronegativity", "mean_electronegativity"),
                          ("atomic_mass_u", "mean_atomic_mass_u"),
                          ("first_ionization_eV", "mean_first_ionization_eV"),
                          ("reference_bandgap_eV", "reference_bandgap_eV"),
                          ("reference_dos_centroid_eV", "reference_dos_centroid_eV"),
                          ("reference_dos_near_fermi_fraction", "reference_dos_near_fermi_fraction")]:
        properties = np.array([s[field] for s in source])
        result["power_weighted_" + output] = (
            float(weights[active] @ properties[active]) if total and np.isfinite(properties[active]).all() else np.nan)
    return result


def coverage_report(clean):
    records = []
    for formula in sorted(set(clean[[f"target_{i}" for i in range(1, 4)]].to_numpy().ravel()) - {"__NONE__"}):
        descriptor = source_descriptors(formula)
        mask = np.zeros(len(clean), dtype=bool)
        for i in range(1, 4):
            mask |= (clean[f"target_{i}"].eq(formula) & clean[f"target_{i}_power_W"].gt(0)).to_numpy()
        gap = reference_data()["bandgaps"].get(composition_key(formula)) if descriptor["formula_known"] else None
        dos_match = dos_reference_data()["compositions"].get(composition_key(formula)) if descriptor["formula_known"] else None
        records.append({"formula": formula, "libraries": clean.loc[mask, "sample_library_id"].nunique(),
                        "formula_known": bool(descriptor["formula_known"]),
                        "reference_bandgap_eV": descriptor["reference_bandgap_eV"],
                        "bandgap_min_eV": gap["min_eV"] if gap else np.nan,
                        "bandgap_max_eV": gap["max_eV"] if gap else np.nan,
                        "reference_records": gap["records"] if gap else 0,
                        "reference_dos_known": bool(descriptor["reference_dos_known"]),
                        "reference_dos_records": descriptor["reference_dos_records"],
                        "reference_dos_near_fermi_fraction": descriptor["reference_dos_near_fermi_fraction"],
                        "reference_dos_mean_bin_std": descriptor["reference_dos_mean_bin_std"],
                        "dos_reference_ids": ";".join(dos_match["reference_ids"]) if dos_match else "",
                        "dos_source": dos_reference_data()["provenance"]["dataset_doi"] if dos_match else "not available",
                        "source": reference_data()["provenance"]["bandgaps"]["doi"] if gap else "not available"})
    return pd.DataFrame(records)
