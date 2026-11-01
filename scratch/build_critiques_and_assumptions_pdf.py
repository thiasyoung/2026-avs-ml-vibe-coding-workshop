"""Create a standalone critique of the one-shot boiling-point workflow."""

from __future__ import annotations

import json
import textwrap
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "01_boiling_point" / "outputs" / "bp_one_shot_2"
PDF_PATH = ARTIFACTS / "critiques_and_assumptions.pdf"

experiments = pd.read_csv(ARTIFACTS / "experiment_summary.csv")
final_metrics = pd.read_csv(ARTIFACTS / "final_metrics.csv").set_index("split")
baseline_metrics = pd.read_csv(ARTIFACTS / "baseline_metrics.csv").set_index("split")
audit = pd.read_csv(ARTIFACTS / "cleaning_audit.csv").iloc[0]
manifest = json.loads((ARTIFACTS / "run_manifest.json").read_text(encoding="utf-8"))
selected = json.loads((ARTIFACTS / "selected_configuration.json").read_text(encoding="utf-8"))
validation_mae = {
    row.experiment_id: float(row.validation_MAE_K)
    for row in experiments.itertuples()
}


BLUE = "#1D4E89"
ORANGE = "#B05E27"
GREEN = "#2E7D32"
LIGHT_BLUE = "#EAF2F8"
LIGHT_ORANGE = "#FCEFE7"
GRAY = "#4A4A4A"


def add_header(fig, title: str, subtitle: str) -> None:
    fig.text(0.075, 0.94, title, color=BLUE, fontsize=20, fontweight="bold", va="top")
    fig.text(0.075, 0.902, subtitle, color=GRAY, fontsize=10, va="top")
    fig.lines.append(plt.Line2D([0.075, 0.925], [0.885, 0.885], color=BLUE, linewidth=1.2))


def add_section(fig, y: float, heading: str, paragraphs: list[str], size: float = 9.6) -> float:
    fig.text(0.075, y, heading, color=ORANGE, fontsize=12, fontweight="bold", va="top")
    y -= 0.031
    for paragraph in paragraphs:
        for line in textwrap.wrap(paragraph, width=112, break_long_words=False, break_on_hyphens=False):
            fig.text(0.085, y, line, color="#202020", fontsize=size, va="top")
            y -= 0.021
        y -= 0.009
    return y


def add_footer(fig, page_number: int) -> None:
    fig.text(
        0.075,
        0.045,
        f"Critiques and assumptions | one-shot boiling-point workflow | page {page_number}",
        color=GRAY,
        fontsize=8,
    )


def new_page(pdf: PdfPages, page_number: int, title: str, subtitle: str):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    add_header(fig, title, subtitle)
    return fig


with PdfPages(PDF_PATH) as pdf:
    fig = new_page(
        pdf,
        1,
        "Critiques and assumptions",
        "Self-evaluation of 04_BP_One_Shot_Workflow.ipynb",
    )
    y = 0.84
    y = add_section(
        fig,
        y,
        "What the workflow did",
        [
            "The workflow predicted the NIST-derived table's reported boiling-point labels from SMILES only. "
            "It canonicalized structures, excluded unsupported multi-component entries, froze an 80/10/10 random split, "
            "compared tree ensembles on validation data, made one bounded fingerprint-length change, and opened the test set once.",
            f"The final selected estimator was Extra Trees with molecular descriptors plus a {selected['fp_bits']}-bit Morgan fingerprint. "
            "Random Forest was a comparison candidate, not the final model. This distinction matters because the two algorithms "
            "make different randomization choices while training.",
        ],
    )
    y = add_section(
        fig,
        y,
        "Measured result",
        [
            f"On {int(final_metrics.loc['test', 'n'])} held-out molecules, the selected model achieved MAE "
            f"{final_metrics.loc['test', 'MAE_K']:.2f} K, RMSE {final_metrics.loc['test', 'RMSE_K']:.2f} K, and R² "
            f"{final_metrics.loc['test', 'R2']:.3f}. The predefined training-median baseline had test MAE "
            f"{baseline_metrics.loc['test_median_baseline', 'MAE_K']:.2f} K.",
            f"The recorded workflow time was {manifest['workflow_runtime_seconds']:.1f} seconds on the execution machine. "
            "The model materially beat the baseline, so it satisfies the notebook's modest workshop success criterion. "
            "That result does not by itself establish readiness for new chemical families or experimental conditions.",
        ],
    )
    y = add_section(
        fig,
        y,
        "Overall self-evaluation",
        [
            "The workflow is strong as a compact, reproducible teaching example: it has an explicit source, a clear no-leakage "
            "sequence, saved split assignments, experiment records, figures, a reloadable model, and an inference helper that "
            "rejects invalid and multi-component inputs.",
            "Its main scientific weakness is evaluation breadth. A single random split favors interpolation among structurally related "
            "molecules. The validation set was also used for four deliberate comparisons, so the selected result may be mildly "
            "optimized to that validation partition. The test set was sealed until the final score, but one test split is still only one draw.",
        ],
    )
    add_footer(fig, 1)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    fig = new_page(
        pdf,
        2,
        "Why these tree models?",
        "Selection rationale, suitability, and the actual validation evidence",
    )
    y = 0.84
    y = add_section(
        fig,
        y,
        "a. Why select tree ensembles for this workshop?",
        [
            "The data contain a few thousand labeled molecules, a mixture of continuous physicochemical descriptors and high-dimensional "
            "binary fingerprint bits, and no need for an external model download. Tree ensembles fit that setting well: they run quickly "
            "on a CPU, accept nonlinear descriptor effects and binary substructure indicators, and do not require feature scaling.",
            "A training-median DummyRegressor established the minimum useful comparison. Random Forest and Extra Trees then provided "
            "two related but distinct nonlinear model families. With the same combined representation and settings, their comparison "
            "isolated a meaningful algorithm choice without making the workshop expensive.",
        ],
    )
    y = add_section(
        fig,
        y,
        "b. Why can they be good models here?",
        [
            "Boiling point reflects several interacting structural factors: molecular size and dispersion, polarity, hydrogen-bonding "
            "capacity, ring and aromatic content, flexibility, branching, and local functional groups. Trees can learn threshold-like "
            "and interaction effects without forcing a linear relationship between those proxies and the target.",
            "Random Forest averages trees built on bootstrap samples and random feature subsets, which can reduce variance. Extra Trees "
            "adds more randomized split selection; this often trains quickly and can be robust when many correlated fingerprint bits are present. "
            "Neither property makes either method physically interpretable or automatically better for extrapolation.",
        ],
    )
    table_data = [["Experiment", "Model / features", "Validation MAE (K)", "Validation R²"]]
    for _, row in experiments.sort_values("validation_MAE_K").iterrows():
        model = "Extra Trees" if row.model_family == "extra_trees" else "Random Forest"
        features = "descriptors + Morgan" if row.feature_set == "descriptors_plus_morgan" else "descriptors only"
        table_data.append([row.experiment_id.replace("experiment_", "E"), f"{model}; {features}", f"{row.validation_MAE_K:.2f}", f"{row.validation_R2:.3f}"])
    table = fig.add_axes([0.075, 0.13, 0.85, 0.19])
    table.axis("off")
    rendered = table.table(cellText=table_data, cellLoc="left", loc="center", colWidths=[0.12, 0.48, 0.2, 0.16])
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8.5)
    rendered.scale(1, 1.42)
    for (row, _), cell in rendered.get_celld().items():
        cell.set_edgecolor("#C8C8C8")
        if row == 0:
            cell.set_facecolor(LIGHT_BLUE)
            cell.set_text_props(weight="bold", color=BLUE)
        elif row == 1:
            cell.set_facecolor("#EEF7EE")
    fig.text(0.075, 0.345, "Validation comparison used the identical frozen train/validation partitions; no test values were accessed.", color=GREEN, fontsize=9)
    fig.text(0.075, 0.095, "Caution: E004 beat the Random Forest candidate by only 0.81 K and the initial Extra Trees run by 0.66 K. "
             "A single split cannot show that these small differences are stable.", color=GRAY, fontsize=8.5)
    add_footer(fig, 2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    fig = new_page(
        pdf,
        3,
        "Why these features?",
        "SMILES-derived descriptors and Morgan fingerprints",
    )
    y = 0.84
    y = add_section(
        fig,
        y,
        "a. Why select this representation?",
        [
            "The input contract allowed only SMILES. The workflow therefore derived every feature from an RDKit molecule rather than "
            "using the provided identifier, target, stored molecular weight, or 3-D record. Compact descriptors give the model explicit "
            "proxies for global molecular character; Morgan fingerprints add a broad, automatically generated view of local structure.",
            "This combined representation was selected by a controlled ablation. Descriptor-only Extra Trees reached validation MAE "
            f"{validation_mae['experiment_001']:.2f} K, while Extra Trees with "
            f"descriptors plus 512-bit fingerprints reached {validation_mae['experiment_003']:.2f} K. "
            "The 1024-bit version then improved the selected validation MAE to "
            f"{validation_mae['experiment_004']:.2f} K.",
        ],
    )
    y = add_section(
        fig,
        y,
        "b. Why can they be good features?",
        [
            "MolWt and heavy-atom count describe size; MolLogP and TPSA summarize hydrophobicity and polarity; hydrogen-bond donor and "
            "acceptor counts reflect interaction opportunities; rotatable bonds, ring counts, aromatic rings, fraction sp³, and formal charge "
            "describe shape, rigidity, and electronic character. These are chemically plausible correlates of cohesive interactions and volatility.",
            "A radius-2 Morgan fingerprint encodes atom neighborhoods out to two bonds. It can distinguish local functional groups, branching, "
            "heteroatom environments, and aromatic fragments that a small hand-chosen descriptor set misses. Increasing from 512 to 1024 bits "
            "reduces hash collisions, although it also increases model size.",
        ],
    )
    y = add_section(
        fig,
        y,
        "Critique of the representation",
        [
            "The descriptors are proxies, not a thermodynamic model. The fingerprint is a hashed structural summary and does not directly "
            "encode crystal packing, conformational ensembles, measurement pressure, phase behavior, or a solvent. It can also collide distinct "
            "substructures in the same bit. The observed improvement supports this representation on this validation split; it is not proof "
            "that each descriptor or fingerprint fragment causally controls boiling point.",
        ],
    )
    add_footer(fig, 3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    fig = new_page(
        pdf,
        4,
        "Data loading, preprocessing, and assumptions",
        "What was done before modeling and what it assumes",
    )
    y = 0.84
    y = add_section(
        fig,
        y,
        "Data loading and provenance",
        [
            "The notebook looked first for 01_boiling_point/outputs/nist_acquisition/NIST_BP_SMILES.parquet and used it without combining it "
            "with the fallback table. It read Parquet with pandas, required SMILES and BP columns, converted BP to numeric, and saved a SHA-256 "
            "checksum in the manifest. The adjacent NIST cleaning audit and acquisition notebook establish BP as kelvin values.",
            "Important assumption: the retained source is suitable for learning a single reported BP target. The provenance does not establish "
            "that all labels are normal boiling points measured at one pressure. This model predicts the source table's labels, not an independently "
            "verified thermodynamic property under a specified condition.",
        ],
    )
    y = add_section(
        fig,
        y,
        "Deterministic preprocessing",
        [
            f"Of {int(audit.source_rows):,} source rows, the audit found no invalid SMILES, no missing/nonfinite BP labels, and no labels outside "
            f"1–2000 K. It flagged {int(audit.multi_component_SMILES)} multi-component SMILES, {int(audit.charged_structures_flagged)} charged "
            f"structures, and {int(audit.stereochemical_SMILES_flagged)} stereochemical SMILES. The single multi-component row was excluded by "
            "a stated support policy; charged single-component structures were retained.",
            "Each eligible structure was canonicalized with stereochemistry retained. Equivalent canonical identities were assigned together; "
            "if repeats had existed, their median BP would have been used and their original row references and BP range retained. In this run, "
            "there were no exact duplicate source rows.",
        ],
    )
    y = add_section(
        fig,
        y,
        "Split and leakage controls",
        [
            "After identity and validity checks, 4,658 unique eligible molecules were split randomly into 3,726 training, 466 validation, "
            "and 466 test molecules using fixed seeds 42 and 43. The notebook asserts that canonical identities never overlap across splits. "
            "Target values, source-row identifiers, and supplied molecular properties are excluded from features.",
            "No scaling or imputation was needed after filtering, and tree models do not require feature scaling. Feature construction is a "
            "deterministic SMILES transformation. The test rows were not transformed, predicted, or scored during the experiment and optimization "
            "steps; final test scoring occurred only after the chosen configuration was written.",
        ],
    )
    add_footer(fig, 4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    fig = new_page(
        pdf,
        5,
        "Is the comparison fair?",
        "Internal experiments: mostly fair. Comparison with other notebooks: not yet a benchmark.",
    )
    y = 0.84
    y = add_section(
        fig,
        y,
        "Fairness within this one-shot workflow",
        [
            "The internal algorithm comparison is reasonably fair. Experiments used the same canonicalized dataset, frozen split assignments, "
            "metrics, random seed, tree count (220), minimum leaf size (2), and maximum feature fraction (0.65). Experiment 002 versus 003 is "
            "the cleanest model-family comparison because both use the same descriptor-plus-512-bit-Morgan representation.",
            "The feature ablation is also informative: Experiment 001 versus 003 holds Extra Trees settings fixed and changes from descriptors "
            "only to descriptors plus fingerprints. Experiment 004 changes only fingerprint length from 512 to 1024 bits after validation selected "
            "the initial leader. The test set was preserved for the final frozen choice.",
        ],
    )
    y = add_section(
        fig,
        y,
        "Internal limitations",
        [
            "This is not exhaustive model selection. Only one random split was used, no confidence interval or repeated split was calculated, "
            "and the validation set guided four choices. Small validation differences may be noise. The training MAE (4.05 K) is much lower than "
            f"validation MAE ({final_metrics.loc['validation', 'MAE_K']:.2f} K), which is consistent with some overfitting or with an easier "
            "training neighborhood. No explicit uncertainty calibration was performed.",
        ],
    )
    y = add_section(
        fig,
        y,
        "Comparison with Chemprop and MoLFormer notebooks",
        [
            "It would not be fair to rank the reported one-shot score directly against the neighboring Chemprop GNN or frozen MoLFormer notebooks "
            "yet. Those notebooks use their own data sampling and preprocessing paths, may use a 4,600-row sample rather than this 4,658-molecule "
            "eligible set, and their split assignments are not imported from this workflow. A shared seed alone does not guarantee the same rows or "
            "the same split after different sorting, sampling, or deduplication.",
            "The modeling budgets also differ. Chemprop uses an iterative neural training procedure with checkpoint selection; MoLFormer uses "
            "pretrained external representations plus a neural head; this workflow uses fixed RDKit representations and four small CPU tree experiments. "
            "Those are valid teaching alternatives, but they are not controlled head-to-head trials.",
        ],
    )
    y = add_section(
        fig,
        y,
        "How to make a fair cross-solution benchmark",
        [
            "Create one canonicalized data table and one shared split_assignments.csv; require every solution to consume it unchanged; define the "
            "same target units and metrics; state equal or deliberately reported tuning and compute budgets; evaluate each frozen model once on the "
            "same test set; and report repeated random or scaffold-grouped splits with uncertainty. Include preprocessing, invalid-input policy, "
            "and external pretraining access in the comparison table.",
        ],
    )
    add_footer(fig, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    fig = new_page(
        pdf,
        6,
        "Bottom-line critique and next steps",
        "What to keep, what to improve, and what claims are defensible",
    )
    y = 0.84
    y = add_section(
        fig,
        y,
        "Defensible claims",
        [
            "For this specific NIST-derived table and fixed random split, an Extra Trees regressor using RDKit descriptors and a 1024-bit Morgan "
            "fingerprint substantially outperformed a training-median baseline. The workflow is reproducible from its source checksum, split file, "
            "configuration, experiment metadata, final predictions, saved model, and reload check.",
            "The local-support value reported by the inference helper is a nearest-training-structure similarity diagnostic. It is useful for "
            "flagging unfamiliar queries but is not a prediction interval, a probability of correctness, or a chemical applicability guarantee.",
        ],
    )
    y = add_section(
        fig,
        y,
        "Claims to avoid",
        [
            "Do not claim that 11.57 K is expected accuracy for new scaffolds, mixtures, salts, ionic systems, a specified normal boiling point, "
            "or another measurement database. Do not infer causal importance from tree representations. Do not call the present score better or "
            "worse than Chemprop or MoLFormer without a shared benchmark.",
        ],
    )
    y = add_section(
        fig,
        y,
        "Recommended next experiment",
        [
            "Run repeated scaffold-grouped cross-validation with the same cleaning and feature policy, and reserve one shared external or grouped "
            "test split. Compare the selected tree model, Chemprop, and MoLFormer under a documented budget. Add calibrated prediction intervals "
            "only through a held-out calibration design, then report coverage and interval width on the final test set.",
        ],
    )
    y = add_section(
        fig,
        y,
        "Document provenance",
        [
            f"Generated {date.today().isoformat()} from the executed one-shot artifacts in "
            "01_boiling_point/outputs/bp_one_shot_2. All numerical claims in this document come from the saved experiment summary, final metrics, "
            "cleaning audit, selected configuration, and run manifest.",
        ],
    )
    add_footer(fig, 6)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"Wrote {PDF_PATH.relative_to(ROOT)}")
