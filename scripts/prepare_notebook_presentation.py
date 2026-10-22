"""Add curated teaching notes and input-collapse hints without executing notebooks.

Run from any directory; --check verifies coverage and freshness without writing.
Notes follow code-cell order. Review the matching entry whenever code changes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", ".ipynb_checkpoints", "archive", "outputs"}
MARKER = "workshop_presentation"
GUIDE = (
    "### Reading and presenting this notebook\n\n"
    "The bullet points above each code cell explain what to notice; saved outputs remain visible. "
    "In **VS Code**, open the Command Palette (`Ctrl+Shift+P` on Windows/Linux, "
    "`Cmd+Shift+P` on macOS) and choose **Notebook: Collapse All Cell Inputs** "
    "(search for `Collapse All Cell Inputs`). Use **Expand All Cell Inputs** to edit, "
    "or expand an individual cell using its input control. Do not collapse outputs. "
    "VS Code can remember your previous view, so use the command if code is still shown. "
    "In **JupyterLab**, code inputs are initially hidden via notebook metadata; "
    "expand them with the left-hand cell collapser or the View menu. "
    "Hiding code does not disable execution or remove it from the file.\n"
)

# Each tuple contains the important discussion points for ONE code cell.
NOTES: dict[str, list[tuple[str, ...]]] = {
    "00_Test_Installation.ipynb": [
        ("Confirm Python 3.11 and select the workshop .venv as the notebook kernel.",
         "Run this notebook from the repository root; a failed assertion means setup needs attention."),
        ("Import the packages needed by all three workshop modules, including NISTChemPy and Chemprop.",
         "This is an import smoke test, not a dataset download or model-training test."),
    ],
    "00_Ten_Steps_in_Prompts.ipynb": [
        ("Load the reaction CSV named in the prompt and preview its rows.",
         "This is a generic practice template; supply that CSV before running it."),
        ("Practice cell: ask for column meanings, missing values, numeric summaries, and readable data figures.",
         "The cell is intentionally blank; expand it and insert the generated code."),
        ("Practice cell: inspect duplicate rows, missing targets, and inconsistent entries.",
         "Review flagged records before deciding whether to change them."),
        ("Practice cell: flag serious anomalies and explain why each deserves inspection.",
         "An unusual observation is not automatically an incorrect measurement."),
        ("Practice cell: apply justified cleaning decisions and summarize the edits.",
         "Save a new working dataset and preserve the original source."),
        ("Practice cell: propose raw, transformed, and interaction features using domain knowledge.",
         "Exclude information that would be unavailable when making a real prediction."),
        ("Practice cell: compare feature sets or inspect feature importance with a suitable plot.",
         "Use training/validation data for decisions; do not select features using test results."),
        ("Practice cell: export the selected features and target as a clearly named modeling dataset.",
         "Preserve row identity so feature and target arrays stay aligned."),
        ("Practice cell: build reusable starter models appropriate for the prediction task.",
         "Keep model names, settings, and training interfaces consistent across experiments."),
        ("Practice cell: create reproducible train, validation, and test splits.",
         "Choose grouping, stratification, or time ordering to match the real prediction setting."),
        ("Practice cell: create experiment folders, metadata, predictions, and a running results table.",
         "Record train/validation results now; reserve test evaluation for the final analysis."),
        ("Practice cell: compare model families on the same training and validation observations.",
         "Track settings and compare validation performance against a simple baseline."),
        ("Practice cell: vary settings for a promising model while keeping data and splits fixed.",
         "Save training histories when the algorithm has epochs; do not invent curves for other models."),
        ("Practice cell: test alternative feature sets against the current validation baseline.",
         "Keep the experiment controlled and leave the test set untouched."),
        ("Practice cell: rank experiments using validation metrics and freeze the finalists.",
         "Only then evaluate those finalists on the held-out test data."),
        ("Practice cell: produce readable data, experiment-comparison, and final-performance figures.",
         "Label axes and units, save PNG files, and report limitations alongside the results."),
    ],
    "01_boiling_point/00_BP_Get_Data_NISTChemPy_v2.ipynb": [
        ("Configure the tested NISTChemPy version, polite request delays, and resumable output paths.",
         "Live collection is off by default so an existing Parquet dataset can be reused."),
        ("Discover neutral compounds with phase-change records, subdividing searches when results are truncated.",
         "Save the candidate IDs for the later resumable collection step; discovery runs only in live mode."),
        ("Parse NIST boiling-point records only when the reported units are kelvin, and derive canonical SMILES.",
         "This cell also fetches one illustrative compound from NIST, so it still needs internet access."),
        ("Download missing records with checkpoints and a failure audit, or reuse an existing raw dataset.",
         "Preserve raw SMILES/BP_K measurements before preparing the final two-column dataset."),
        ("Clean cached data without repeating the long download: validate SMILES and finite, plausible kelvin values.",
         "Combine repeated canonical structures using median BP, save SMILES/BP in kelvin, and write a cleaning audit.",
         "A plausibility filter does not establish the correct units of an ambiguous legacy measurement."),
    ],
    "01_boiling_point/02_BP_1_Chemprop_GNN.ipynb": [
        ("Load boiling points in kelvin and make reproducible molecular train/validation/test splits shared with the foundation-model demo.",
         "Only SMILES supplies input features; depth controls message-passing steps and hidden dimension controls vector width.",
         "Review the architecture, 20-epoch segments, and learning-rate settings before starting the longer training cell."),
        ("Train quietly in 20-epoch Chemprop CLI segments and stitch the recorded training/validation losses.",
         "At segment boundaries, stop only if both the raw and top-five-trimmed 20-epoch validation slopes rise and validation loss exceeds training loss.",
         "Segments transfer weights, not optimizer state. Select the lowest eligible native segment checkpoint, with a fallback if none qualify."),
        ("Plot the stitched loss curves separately from training and mark the actual selected checkpoint.",
         "Generate all split predictions using that checkpoint, not the last training epoch."),
        ("Compare train, validation, and test MAE/RMSE/R² against a training-mean baseline in kelvin.",
         "Use common-axis parity plots and best/worst held-out molecules to discuss strengths and failures.",
         "Save aligned predictions, metrics, architecture settings, and the selected checkpoint path."),
        ("Enter a valid SMILES string, such as CCO for ethanol, and predict with the selected final checkpoint.",
         "The returned scalar is a boiling point in kelvin; a plausible example is not a substitute for held-out evaluation."),
        ("Optional blank practice cell: expand it to add a question or your own follow-up calculation.",
         "No additional analysis is implemented in this cell."),
    ],
    "01_boiling_point/03_BP_2_Foundation_Model.ipynb": [
        ("Use the same reproducible molecular splits as the Chemprop example and keep boiling points in kelvin.",
         "Configure a frozen MoLFormer encoder and a 64 → 32 → 16 feed-forward head without extra molecular descriptors.",
         "The head uses a small learning rate, cosine warm restarts, and the robust 20-epoch stopping rule."),
        ("Load the public MoLFormer model and freeze its weights; the first run requires a model download.",
         "Convert SMILES into fixed embeddings, then fit embedding and target standardization on training data only."),
        ("Train only the regression head with AdamW/MSE, cosine warm restarts, and gradient clipping.",
         "Require positive raw and top-five-trimmed validation slopes plus validation loss above training loss to stop.",
         "Restore the lowest eligible validation-loss epoch, with an overall-minimum fallback; the epoch cap is not proof of convergence."),
        ("Plot training/validation standardized MSE and learning rate after training completes.",
         "Mark the selected checkpoint and any robust-slope stopping epoch, rather than labeling every upward fluctuation."),
        ("Undo target standardization before calculating boiling-point errors in kelvin.",
         "Compare all three split parity plots on the same axes and inspect good/bad held-out molecules.",
         "Save the metrics and test predictions for the restored head."),
        ("Predict a new SMILES using the frozen encoder, training-fitted scaling, and restored head.",
         "Edit the example SMILES and read the resulting boiling point in kelvin."),
        ("Optional blank practice cell: expand it for your own follow-up prediction or analysis.",
         "This cell does not contain another model or training step."),
    ],
    "01_boiling_point/05_BP_4_One_Shot_Workflow.ipynb": [
        ("Locate the workshop dataset and create separate experiment/figure output folders.",
         "Set the seed and SMILES-derived feature policy; prefer acquired NIST data over the packaged fallback."),
        ("Audit canonical molecular identity, missing/nonfinite targets, components, charge, and plausible temperature bounds.",
         "Check the source provenance for units; a numerical range alone cannot confirm kelvin."),
        ("Review flagged rows and repeated canonical identities before modeling.",
         "Repeated measurements can disagree; inspect their BP range rather than silently discarding them."),
        ("Retain eligible single-component molecules and summarize repeated identities by median BP.",
         "Freeze an 80/10/10 split of unique canonical structures and assert no molecular identity overlap."),
        ("Inspect the training target distribution and individual training labels.",
         "These plots describe the training data; unusual values still need source-based review."),
        ("Build molecular size/polarity descriptors and radius-two Morgan fingerprints directly from SMILES.",
         "Save the representation specification; neither BP nor source identifiers are model features."),
        ("Define comparable regression metrics and tree-model builders.",
         "Fit a training-median dummy baseline to establish a useful minimum benchmark."),
        ("Compare descriptor-only and fingerprint-augmented tree models on fixed training/validation splits.",
         "Save hypotheses, settings, runtime, and predictions for each experiment; no test score is used for selection."),
        ("Compare validation MAE across experiments against the training-median baseline.",
         "Lower is better; this is a validation comparison, not a ranking by test performance."),
        ("Inspect the leading validation model's train–validation gap and largest errors by molecular size.",
         "Use these patterns to propose an experiment, not to claim a causal chemical explanation."),
        ("Try a 1024-bit fingerprint representation with the leading model family and fixed split.",
         "Keep the change only if validation MAE improves, then save the frozen final configuration."),
        ("Fit the chosen configuration on training data and now report independent test metrics.",
         "Save aligned predictions and compare held-out performance to the training-median baseline."),
        ("Read train/validation/test parity plots and the held-out residual plot in kelvin.",
         "Review the largest test errors to describe limitations, not to retune the selected model."),
        ("Save the final inference bundle and predict new SMILES with validity checks.",
         "Training-set fingerprint similarity is a heuristic local-support diagnostic, not calibrated prediction uncertainty."),
        ("Reload the saved model in a fresh Python interpreter without retraining.",
         "Check finite predictions, row alignment, and deliberate rejection of unsupported inputs."),
        ("Save a reproducibility manifest with dataset checksum, software versions, split policy, and chosen configuration.",
         "Use it to distinguish a repeatable experiment from an undocumented one-shot answer."),
    ],
    "02_ir_functional_groups/00_IR_Get_Data_NISTChemPy_v2.ipynb": [
        ("Load candidate NIST IDs and configure polite, resumable IR acquisition with the tested NISTChemPy version.",
         "Live collection defaults to off; keep the native checkpoint so the workshop need not repeat scraping."),
        ("Fetch the first available IR spectrum and derive canonical SMILES from the NIST molecular structure.",
         "Preserve native x/y arrays and units; derive 12 binary functional-group labels using explicit SMARTS rules."),
        ("Collect only outstanding records, checkpoint successful downloads, and retain retrieval failures.",
         "Acquisition preserves native spectra; conversion, cropping, padding, and regridding belong in notebook 01."),
        ("Reload the native checkpoint and inspect recorded coordinate/signal units and example arrays.",
         "No spectral normalization or data carpentry is applied in this preview."),
        ("Verify that molecular IDs, SMILES, native arrays, unit fields, and all labels are present.",
         "Continue to notebook 01 for data examination and carpentry, not directly to training."),
    ],
    "02_ir_functional_groups/01_IR_Vibe_Coding_Data_Carpentry.ipynb": [
        ("Load native arrays and exclude the frozen validation/test molecular observations from the training pool.",
         "Preview the recorded numbers and the fixed 500-point reference grid; this is intentionally a minimal baseline."),
        ("Use recorded numeric signals as-is, require full reference coverage, and linearly resize to a uniform grid.",
         "Apply per-spectrum min–max normalization and inspect the resulting numeric curves and audit counts."),
        ("Save the baseline training table together with the unchanged evaluation files and a data manifest.",
         "Reload and verify finite uniform features, binary labels, and disjoint identities before using static IR 02."),
    ],
    "02_ir_functional_groups/02_IR_Optimized_Training.ipynb": [
        ("Select the carpentry output directory while keeping the model configuration fixed across data comparisons.",
         "Verify the frozen evaluation hashes, 500-point grid, binary labels, and molecular identity separation.",
         "Fit StandardScaler on training features only; seed the run and set the initial learning rate to 1e-5."),
        ("Define a 256 → 128 ReLU classifier with batch normalization and dropout; output one logit per functional group.",
         "Use training-derived class weights in BCEWithLogitsLoss, AdamW, and cosine annealing with warm restarts.",
         "Create a unique run folder, TensorBoard log, and reproducibility metadata before training."),
        ("Train with gradient clipping and compare training/validation loss in the same evaluation mode.",
         "Stop only when raw and top-five-trimmed 20-epoch validation slopes are positive and validation loss exceeds training loss.",
         "Save periodic checkpoints and restore the lowest eligible validation loss, or its documented fallback; report if the epoch cap is reached."),
        ("Choose per-label probability thresholds using validation data only, then evaluate the held-out test set.",
         "Report macro/micro F1, exact match, and per-group confusion-derived metrics; plot loss/LR and classification scores.",
         "Save the restored model, scaling, thresholds, predictions, and run documentation."),
        ("Upload one CSV/Parquet spectrum with wavenumber and IR columns to use the trained classifier.",
         "Supply signal values already prepared consistently with the selected carpentry dataset; this helper only regrids and applies the fitted scaler.",
         "The uploaded spectrum must cover the training coordinate range; the helper rejects unsupported bounds."),
    ],
    "02_ir_functional_groups/03_IR_Compare_Carpentry.ipynb": [
        ("Locate the baseline and four worked carpentry strategies and choose a comparison output folder.",
         "TRAIN_ALL is off by default; turning it on executes the existing static trainer for all options."),
        ("Accept only completed runs whose dataset checksums, frozen holdouts, grid, labels, and training protocol match.",
         "Compare F1, sample counts, selected epoch, and stopping reason; flag missing/stale runs rather than inventing scores.",
         "These strategies bundle multiple edits, so a score difference does not isolate one cleansing step."),
        ("Compare aggregate F1 and per-functional-group test scores for compatible runs.",
         "Read both gains and regressions honestly; do not select another model by repeatedly optimizing the test set."),
    ],
    "03_htem_conductivity/00_HTEM_Get_Data_NREL.ipynb": [
        ("Locate the HTEM module, load the shared acquisition helpers, and configure readable workshop plots.",
         "Keep paths relative to the workshop so the notebook works from the module or repository root."),
        ("Download all public libraries with compact resumable recipe/scalar caches, or explicitly select the offline snapshot.",
         "Retain each deposition/position and substrate x/y coordinates, process settings, and scalar property labels—not full spectra.",
         "Check provenance, failures, and sample-count gaps; do not silently fill missing coordinates or claim unavailable records were downloaded."),
        ("Preview per-location process/label rows and the number of positions in each deposition.",
         "Preserve missing targets and substrate-grid identity for later masked training and spatial maps."),
    ],
    "03_htem_conductivity/01_HTEM_Process_Data_Carpentry.ipynb": [
        ("Load the shared HTEM preparation helpers and use repository-relative module paths.",
         "Set a reproducible seed; every later architecture consumes this same preparation."),
        ("Verify the acquired dataset checksum and inspect missing fields, ranges, units, and unique deposition/position keys.",
         "Substrate coordinates are millimeters, thickness is micrometers, and conductivity is S/cm."),
        ("Build the audited spatial cohort with the frozen grouped/material-balanced split and explicit missing-label masks.",
         "Use recipe, physical XY, local temperature, and instructor-selected source descriptors; measured film properties are labels only.",
         "Inspect source chemistry/DOS coverage rather than treating missing references as known material phases."),
        ("See how a real reference atomic structure becomes a frozen ALIGNN graph embedding.",
         "This cell only defines the optional instructor extraction function; the class uses shipped compact features and need not install ALIGNN."),
        ("Rebuild curated elemental descriptors and a 16-component reference-crystal PCA fitted on unique training compositions only.",
         "Combine them with the existing recipe/atomic/reference-gap/DOS inputs and train-fit imputation/scaling.",
         "Assert exact agreement with all three shared frozen matrices; reference crystals are not verified film or source phases."),
        ("Verify disjoint recipe/library groups, material-class split coverage, label counts, and frozen evaluation identity.",
         "Singleton/two-group classes have explicit rare-class exceptions; missing labels remain masked, not imputed.",
         "Preserve full raw planned grids separately from the rows with available targets."),
        ("Inspect training-only property distributions and flag IQR anomalies for review without automatically removing them.",
         "Count categorical support across independent libraries to expose sparse training coverage."),
    ],
    "03_htem_conductivity/03_HTEM_Compare_Architectures.ipynb": [
        ("Load completed runs only after checking shared data, preprocessing, labeled locations, and result fingerprints.",
         "Compare five algorithms on expanded test labels and separately on the original fixed paired holdout.",
         "Rank using validation performance; the default participant model remains linear Ridge and this notebook does not retrain."),
        ("Compare validation standardized error, test thickness MAE, and test conductivity log-MAE.",
         "These metrics use different units/scales; lower is better within each panel, not across panels."),
        ("Plot test-set thickness and log10 conductivity parity for every algorithm using the verified saved predictions.",
         "Use shared property-wise axes and observed-label masks so model comparisons are fair; no refitting or silent omission of invalid predictions."),
    ],
    "scratch/HTEM_Feature_Selection_Study.ipynb": [
        ("Instructor-only study: compare predefined chemistry feature blocks with fixed Ridge and forest selectors.",
         "Use existing grouped training/validation observations; do not consult test scores to choose features."),
        ("Inspect independent reference-crystal provenance, checkpoint checksums, coverage, and extraction failures.",
         "Run the candidate study using only training-fitted transforms and validation-based selection."),
        ("Compare each selector's validation errors and the equal-selector relative ranking.",
         "Report the exploratory gain interval and selector/single-holdout dependence; an interval spanning zero is not proof of improvement."),
        ("Deploy the already chosen feature specification and compact source table for shared workshop preparation.",
         "Then rerun participant 01, every architecture, and comparison 03; this is an instructor publishing step, not routine model tuning."),
    ],
}

IR_VARIANTS = [
    ("01a_IR_Omit_Transmittance.ipynb", "Keep absorbance records only; require full spectral coverage and use linear resampling.",
     "Canonicalize molecular identities and deduplicate training presentations; inspect how many examples are lost."),
    ("01b_IR_Convert_Transmittance.ipynb", "Convert supported transmittance to absorbance, require full coverage, and linearly resample.",
     "Check categorical labels against SMILES/SMARTS and audit inconsistent training records."),
    ("01c_IR_Padding_and_Quality.ipynb", "Convert supported signals to absorbance and allow zero-padding for partial coverage of at least 80%.",
     "Apply training-only roughness-based quality filtering; statistical anomalies need not be measurement errors."),
    ("01d_IR_Resampling_and_Balancing.ipynb", "Convert supported signals, pad acceptable partial coverage, then bin-downsample and upsample to the shared grid.",
     "Use capped rare-label oversampling on training only; resizing does not restore lost spectral information."),
]
for filename, spectral, cleansing in IR_VARIANTS:
    NOTES["02_ir_functional_groups/examples/" + filename] = [
        ("Select this example's signal, bounds, resampling, and cleansing policies and its own output folder.",
         "Exclude frozen validation/test molecular observations before any training-data changes."),
        ("Examine native signal/coordinate units, NaNs, array lengths, coverage, label categories, duplicates, and statistical anomalies.",
         "Use the preceding hypothesis/prompt to explain what to fix; inspect flagged observations rather than assuming they are wrong."),
        (spectral, cleansing,
         "Review the action audit, label counts, padded/resampled records, and uniform example spectra."),
        ("Save this strategy's training output alongside the byte-identical frozen evaluation tables and manifest.",
         "Reload and check finite normalized 500-point arrays, 12 binary labels, and disjoint identities; point static IR 02 at the printed directory."),
    ]

HTEM_SETUP = (
    "Load notebook 01's shared frozen features, target transforms, masks, and material-balanced group splits without rebuilding them.",
    "Each input row includes process conditions and that location's x/y; properties are two standardized log targets with separate observed-label masks.",
    "Write this architecture's artifacts to its own experiment folder so comparisons remain separate.",
)
HTEM_EVALUATE = (
    "Predict each labeled location and save models, aligned split predictions, metrics, and reproducibility fingerprints.",
    "Undo target transforms for physical-unit reporting and inspect train/validation/test parity plots; missing labels are excluded per property, never imputed.",
)
HTEM_MAP = (
    "Reconstruct actual and predicted maps for one held-out deposition from its individual location predictions.",
    "Pivot on exact substrate x/y coordinates without averaging depositions or interpolating holes; compare both level and spatial gradients.",
)
HTEM_INFER = (
    "Reload the saved model and first verify that the unchanged recipe/grid reproduces its reference predictions.",
    "Then edit planned recipe settings or the requested x/y/local-temperature grid and save new predicted thickness/conductivity maps.",
    "No measured film property is required as an input; point models predict each row independently, while the GNN requires its full planned grid.",
)
HTEM_CURVES = (
    "Display training/validation loss and learning-rate history separately from model fitting.",
    "Mark the restored validation-minimum epoch and read the stopping note; a finite epoch cap does not guarantee convergence.",
)
NOTES["03_htem_conductivity/02_HTEM_Process_Emulator_Vibe_Coding.ipynb"] = [
    HTEM_SETUP,
    ("Fit linear Ridge regression with alpha=10, separately using each property's observed training labels.",
     "This is the default process emulator and has no neural epochs or early-stopping curve; tune it through the workshop prompts."),
    HTEM_EVALUATE, HTEM_MAP, HTEM_INFER,
]
NOTES["03_htem_conductivity/examples/02a_HTEM_Kernel_Ridge.ipynb"] = [
    HTEM_SETUP,
    ("Search nine alpha/RBF-width settings and select by mean standardized validation RMSE, never test error.",
     "Fit full exact float64 kernels with checked solver residuals; this can require substantial RAM and has no epoch loss curve."),
    HTEM_EVALUATE, HTEM_MAP, HTEM_INFER,
]
NOTES["03_htem_conductivity/examples/02b_HTEM_Feed_Forward_Network.ipynb"] = [
    HTEM_SETUP,
    ("Fit a 64 → 32 feed-forward network on the unchanged prepared per-location feature rows.",
     "Use AdamW, cosine decay, observed-label loss masks, and validation-minimum checkpointing with a patience limit."),
    HTEM_CURVES, HTEM_EVALUATE, HTEM_MAP, HTEM_INFER,
]
NOTES["03_htem_conductivity/examples/02c_HTEM_Grid_GNN.ipynb"] = [
    HTEM_SETUP,
    ("Build one geometry-only neighbor graph for each full planned deposition grid, retaining unlabeled nodes.",
     "Visualize horizontal/vertical connectivity; never connect different depositions or construct edges from measured properties."),
    ("Train a width-64, two-message-layer graph model with one output per substrate location, not one pooled output per deposition.",
     "Use observed-label masks, cosine decay, and validation-minimum checkpointing with patience-based stopping."),
    HTEM_CURVES, HTEM_EVALUATE, HTEM_MAP, HTEM_INFER,
]
NOTES["03_htem_conductivity/examples/02d_HTEM_Random_Forest.ipynb"] = [
    HTEM_SETUP,
    ("Fit a separate 300-tree Random Forest for each property using its observed training labels.",
     "Keep a fixed seed, bounded depth/threads, and no row-level out-of-bag substitute for independent deposition validation."),
    HTEM_EVALUATE, HTEM_MAP, HTEM_INFER,
]


def notebook_paths(root: Path = ROOT) -> list[Path]:
    return sorted(p for p in root.rglob("*.ipynb") if not EXCLUDED.intersection(p.relative_to(root).parts))


def source_text(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


def markdown_cell(relative: str, key: str, text: str, metadata: dict) -> dict:
    return {"cell_type": "markdown", "id": str(uuid.uuid5(uuid.NAMESPACE_URL, relative + ":" + key)),
            "metadata": {MARKER: metadata}, "source": text.splitlines(keepends=True)}


def prepare_notebook(notebook: dict, relative: str) -> dict:
    """Pure, idempotent transform; preserve code, outputs and all original prose."""
    result = deepcopy(notebook)
    original = [c for c in result["cells"] if MARKER not in c.get("metadata", {})]
    code_cells = [c for c in original if c["cell_type"] == "code"]
    if not code_cells:
        return result
    notes = NOTES.get(relative)
    if notes is None or len(notes) != len(code_cells):
        raise ValueError(f"{relative}: review teaching notes for {len(code_cells)} code cells "
                         f"(catalog contains {len(notes) if notes is not None else 0})")
    cells = []
    note_index = 0
    guide_inserted = False
    for index, cell in enumerate(original):
        # Keep the original first-cell title/images/captions intact.
        if not guide_inserted and (index == 1 or (index == 0 and cell["cell_type"] == "code")):
            cells.append(markdown_cell(relative, "guide", GUIDE, {"kind": "guide"}))
            guide_inserted = True
        if cell["cell_type"] == "code":
            key = cell.get("id", f"code-{note_index}")
            digest = hashlib.sha256(source_text(cell).encode("utf-8")).hexdigest()
            text = "**What to notice**\n\n" + "\n".join("- " + point for point in notes[note_index]) + "\n"
            cells.append(markdown_cell(relative, key, text,
                                       {"kind": "notes", "code_cell": key, "source_sha256": digest}))
            jupyter = cell.setdefault("metadata", {}).setdefault("jupyter", {})
            jupyter.update(source_hidden=True, outputs_hidden=False)
            # Legacy collapsed=True hides OUTPUTS, not code. Explicitly keep outputs open.
            cell["metadata"]["collapsed"] = False
            note_index += 1
        cells.append(cell)
    result["cells"] = cells
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify notes/collapse metadata without writing")
    args = parser.parse_args()
    changed = []
    count = code_count = 0
    # Prepare every file before writing any: a missing catalog entry must not cause a partial rewrite.
    updates = []
    for path in notebook_paths():
        relative = path.relative_to(ROOT).as_posix()
        original = json.loads(path.read_text(encoding="utf-8"))
        updated = prepare_notebook(original, relative)
        n_code = sum(c["cell_type"] == "code" for c in original["cells"])
        count += bool(n_code)
        code_count += n_code
        if updated != original:
            changed.append(relative)
            updates.append((path, updated))
    if args.check and changed:
        print("Presentation notes/settings need updating:", *changed, sep="\n- ")
        return 1
    for path, updated in updates:
        path.write_text(json.dumps(updated, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{'Checked' if args.check else 'Prepared'} {code_count} code cells in {count} notebooks; "
          f"{len(changed)} notebook(s) {'need updates' if args.check else 'updated'}. No code executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
