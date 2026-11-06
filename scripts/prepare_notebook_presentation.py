"""Add concise code summaries and input-collapse hints without executing notebooks.

Run from any directory; --check verifies coverage and freshness without writing.
Summaries follow code-cell order. Review the matching entry whenever code changes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", ".quarto", "_site", ".ipynb_checkpoints", "archive", "outputs"}
# Executed assistant-generated runs keep their own explanations and are preserved byte-for-byte.
PRESERVED_DIRECTORIES = ("01_boiling_point/examples/prompt_iteration/notebooks",)
MARKER = "workshop_presentation"
GUIDE = (
    "### Reading and presenting this notebook\n\n"
    "Each code block has a descriptive heading followed by a short explanation labeled **Code Summary:**; "
    "saved outputs remain visible. "
    "In **VS Code**, open the Command Palette (`Ctrl+Shift+P` on Windows/Linux, "
    "`Cmd+Shift+P` on macOS) and choose **Notebook: Collapse All Cell Inputs** "
    "(search for `Collapse All Cell Inputs`). Use **Expand All Cell Inputs** to edit, "
    "or expand an individual cell using its input control. Do not collapse outputs. "
    "VS Code can remember your previous view, so use the command if code is still shown. "
    "In **JupyterLab**, code inputs are initially hidden via notebook metadata; "
    "expand them with the left-hand cell collapser or the View menu. "
    "Hiding code does not disable execution or remove it from the file.\n"
)

# Each paragraph summarizes ONE code cell. Blank exercises describe intended work,
# not completed analysis. Keep the NOTES name and generated-cell metadata stable.
NOTES: dict[str, list[str]] = {
    "00_Test_Installation.ipynb": [
        "Checks that the notebook is running from the workshop folder with Python 3.11 and the workshop virtual environment.",
        "Checks that the packages used throughout the workshop can be imported. This verifies installation without downloading data or training a model.",
    ],
    "00_Ten_Steps_in_Prompts.ipynb": [
        "Loads and previews the reaction dataset named in the prompt. This practice template requires you to supply that CSV file.",
        "This blank practice cell is for code that explains the dataset's columns, missing values, and distributions using summaries and plots.",
        "This blank practice cell is for checking duplicate rows, missing targets, and inconsistent entries before choosing any corrections.",
        "This blank practice cell is for identifying unusual observations and explaining why they warrant inspection, without assuming that every outlier is an error.",
        "This blank practice cell is for applying justified corrections and saving a cleaned dataset while preserving the original data.",
        "This blank practice cell is for creating useful model inputs from domain knowledge, including transformations and interactions available at prediction time.",
        "This blank practice cell is for comparing feature sets or plotting feature importance using training and validation data, not test results.",
        "This blank practice cell is for exporting the chosen inputs and target as an aligned modeling dataset.",
        "This blank practice cell is for defining reusable starter models with consistent settings and training interfaces for later comparisons.",
        "This blank practice cell is for creating reproducible training, validation, and test sets that reflect the intended prediction setting.",
        "This blank practice cell is for recording experiment settings, predictions, and training/validation results in organized output folders.",
        "This blank practice cell is for comparing model families against a simple baseline using the same training and validation observations.",
        "This blank practice cell is for varying model settings while holding the data and splits fixed, recording training histories where applicable.",
        "This blank practice cell is for testing alternative feature sets against the current validation baseline while leaving the test set untouched.",
        "This blank practice cell is for selecting finalists using validation results, then evaluating those fixed choices on held-out test data.",
        "This blank practice cell is for producing labeled figures that summarize the data, experiments, final performance, and limitations.",
    ],
    "01_boiling_point/00_BP_Get_Data_NISTChemPy_v2.ipynb": [
        "Sets up NIST access, request delays, and resumable data files. Live collection is off by default so existing Parquet data can be reused.",
        "Finds neutral compounds with phase-change data and saves their NIST identifiers for collection. Large searches are subdivided to avoid truncated results; this runs only in live mode.",
        "Defines how to extract canonical SMILES and boiling-point measurements reported in kelvin. An illustrative compound is fetched from NIST, so this cell requires internet access.",
        "Collects missing measurements with checkpoints and a failure log, or reloads the existing raw dataset. Original SMILES and kelvin measurements are preserved before cleaning.",
        "Cleans cached measurements without downloading them again, checks molecular structures and plausible temperatures, and combines repeated structures by median boiling point. Saves a two-column SMILES/BP table in kelvin and an audit; range checks alone cannot confirm ambiguous source units.",
    ],
    "01_boiling_point/examples/reference_models/01_BP_Chemprop_GNN.ipynb": [
        "Loads boiling points in kelvin, creates the molecular splits shared with the foundation-model example, and configures Chemprop training. Molecular graphs derived from SMILES are the only inputs.",
        "Trains Chemprop in 20-epoch segments, carrying forward weights but not optimizer state. Stopping checks require rising validation-loss trends both before and after removing the five highest losses, with validation loss above training loss. Saves the lowest eligible segment checkpoint, with a documented fallback.",
        "Plots the combined training and validation loss histories, marks the selected checkpoint, and uses that checkpoint to predict boiling points for all three splits.",
        "Compares model errors against a simple mean baseline, shows train/validation/test parity plots and good/bad held-out examples, and saves predictions, metrics, and model settings. Boiling-point errors are reported in kelvin.",
        "Predicts the boiling point in kelvin for an editable SMILES string using the selected final model. The example input is ethanol, written as CCO.",
        "This optional practice cell is blank and ready for a follow-up question or calculation.",
    ],
    "01_boiling_point/examples/reference_models/02_BP_Foundation_Model.ipynb": [
        "Loads the same molecular splits as Chemprop and configures a frozen MoLFormer encoder with a small feed-forward regression head. Training uses cosine warm restarts and the robust 20-epoch stopping rule; boiling points remain in kelvin.",
        "Loads the pretrained MoLFormer encoder and converts SMILES into fixed embeddings without updating the encoder. Embedding and target scaling are fitted on training data only; the first run downloads the model.",
        "Trains the regression head using AdamW, MSE, and cosine warm restarts. Stopping requires rising raw and outlier-trimmed validation trends with validation loss above training loss. Restores the lowest eligible validation-loss weights, or the documented fallback if no epoch qualifies.",
        "Shows training/validation loss and learning-rate histories, marking the restored model and any slope-based stopping point rather than every upward fluctuation.",
        "Converts predictions back to kelvin, compares train/validation/test parity plots, and examines good/bad held-out examples. Saves evaluation metrics and test predictions from the restored model.",
        "Predicts a boiling point in kelvin for an editable SMILES string using the frozen encoder, training-fitted scaling, and restored regression head.",
        "This optional practice cell is blank and ready for another prediction or analysis.",
    ],
    "02_ir_functional_groups/00_IR_Get_Data_NISTChemPy_v2.ipynb": [
        "Loads candidate NIST identifiers and sets up resumable IR collection with request delays. Live downloading is off by default so the workshop can reuse the native-spectrum checkpoint.",
        "Defines retrieval of each compound's first available IR spectrum and canonical SMILES. Native coordinates, signals, and units are preserved, while molecular substructure rules assign 12 functional-group labels.",
        "Downloads outstanding records, saves progress, and logs retrieval failures. Spectra remain in their original form; signal conversion and grid preparation are deferred to notebook 01.",
        "Reloads the native-spectrum checkpoint and previews coordinate units, signal units, and example arrays without normalizing or reshaping the spectra.",
        "Checks that the acquired table contains molecular identifiers, SMILES, native arrays, units, and functional-group labels before data carpentry in notebook 01.",
    ],
    "02_ir_functional_groups/examples/01_IR_Vibe_Coding_Data_Carpentry.ipynb": [
        "Loads native spectra, separates the frozen validation/test molecules from the training pool, and previews the fixed 500-point reference grid for a minimal preparation baseline.",
        "Uses the recorded numeric signals directly, retains spectra covering the full reference range, and linearly resamples them to a common grid. Each spectrum is min–max normalized, then plotted alongside preparation counts.",
        "Saves the baseline training data, unchanged evaluation tables, and a manifest. Reloading checks uniform finite inputs, binary labels, and separated molecular identities before training in notebook 02.",
    ],
    "02_ir_functional_groups/02_IR_Optimized_Training.ipynb": [
        "Loads the selected carpentry output and checks its grid, labels, molecular separation, and frozen evaluation files. Fits input standardization on training data only and keeps the training configuration fixed for data comparisons, starting at a learning rate of 1e-5.",
        "Builds the multi-label neural classifier and configures class-weighted binary cross-entropy, AdamW, and cosine warm restarts. Creates an experiment folder, TensorBoard log, and settings record before training.",
        "Trains the classifier, saves periodic checkpoints, and monitors comparable training/validation losses. Stopping requires rising 20-epoch validation trends both with and without the five highest losses, with validation loss above training loss. Restores the lowest eligible validation-loss model or the documented fallback, and reports any epoch-limit stop.",
        "Selects classification thresholds on validation data, evaluates the restored model on the test set, and plots training histories and functional-group scores. Saves the model, scaling, thresholds, predictions, and metrics for comparison.",
        "Classifies an uploaded CSV or Parquet spectrum using the trained model. Signals must already match the chosen carpentry preparation; this helper only regrids and scales them, rejecting spectra that do not cover the required coordinate range.",
    ],
    "02_ir_functional_groups/examples/03_IR_Compare_Carpentry.ipynb": [
        "Locates the baseline and four worked carpentry outputs and sets the comparison folder. Training is off by default; enabling TRAIN_ALL runs the same static trainer for every option.",
        "Checks completed runs for compatible datasets, frozen holdouts, and training settings, then compares scores, sample counts, and stopping information. Missing or stale results are flagged. Each strategy combines several changes, so score differences cannot isolate a single cleansing step.",
        "Plots overall and per-functional-group test F1 for compatible runs, showing both improvements and regressions from the different data-preparation strategies.",
    ],
    "03_htem_conductivity/00_HTEM_Get_Data_NREL.ipynb": [
        "Locates the HTEM module and loads its shared download helpers and plotting settings, using paths that work from either the module or repository folder.",
        "Downloads compact process settings, substrate coordinates, and scalar properties from public HTEM libraries, or loads the explicitly selected offline snapshot. Resumable caches and acquisition logs preserve provenance; full spectra are excluded and unavailable records are not filled in.",
        "Previews the per-location dataset and counts positions within each deposition. Missing property labels and substrate-grid identities are preserved for later training and spatial maps.",
    ],
    "03_htem_conductivity/01_HTEM_Process_Data_Carpentry.ipynb": [
        "Loads the shared HTEM preparation helpers and sets a reproducible seed. Every model in this module will use the same prepared dataset.",
        "Verifies the downloaded dataset and examines missing values, ranges, units, and deposition/location identifiers. Coordinates are in millimeters, thickness in micrometers, and conductivity in S/cm.",
        "Builds the spatial modeling dataset with grouped, material-balanced splits and masks for missing labels. Inputs combine recipe settings, position, local temperature, and source-material descriptors; measured film properties remain targets rather than prediction inputs.",
        "Defines an optional instructor function for turning a reference atomic structure into a frozen ALIGNN embedding. It does not run extraction; participants use the shipped compact features without installing ALIGNN.",
        "Reconstructs elemental descriptors and reduced reference-crystal embeddings using training-fitted transformations, then checks agreement with the shared feature matrices. These reference structures describe candidate source chemistry, not verified phases of the deposited film.",
        "Checks separated recipe/library groups, material-class coverage, available labels, and frozen evaluation identities. Rare classes have documented split exceptions; missing labels remain masked, and full substrate grids are kept separately for spatial modeling.",
        "Plots training-property distributions, flags statistical outliers for review without removing them automatically, and counts independent libraries per material category to highlight sparse coverage.",
    ],
    "03_htem_conductivity/examples/03_HTEM_Compare_Architectures.ipynb": [
        "Loads verified results from five models trained on the shared features, without retraining. Compares expanded test observations and the original paired holdout separately, with model ranking based on validation performance.",
        "Compares validation error, test thickness error, and test conductivity error in separate panels. Each panel uses its own scale and units, with lower error indicating better performance.",
        "Plots test predictions against measured thickness and log conductivity for each algorithm using shared axes and the same available labels. Results come from verified saved predictions, not refitted models.",
    ],
    "scratch/HTEM_Feature_Selection_Study.ipynb": [
        "Sets up an instructor comparison of candidate chemistry features using fixed Ridge and Random Forest models on grouped training/validation data. Test results are not used to choose features.",
        "Checks reference-crystal provenance and extraction coverage, then evaluates candidate features with training-fitted transformations and validation-based selection.",
        "Compares validation errors and relative feature rankings across the two models. The reported exploratory gain interval reflects uncertainty; an interval spanning zero does not establish an improvement.",
        "Publishes the selected feature specification and compact source table for the shared workshop dataset. This instructor step is followed by rerunning preparation, all architectures, and their comparison.",
    ],
}

IR_VARIANTS = [
    ("01a_IR_Omit_Transmittance.ipynb",
     "Keeps only absorbance spectra with full reference coverage, resamples them linearly, and deduplicates training molecules. Displays the resulting spectra, label counts, and action audit to show what was retained and removed."),
    ("01b_IR_Convert_Transmittance.ipynb",
     "Converts supported transmittance signals to absorbance and linearly resamples fully covered spectra. Checks functional-group labels against molecular substructures and displays the preparation audit and resulting spectra."),
    ("01c_IR_Padding_and_Quality.ipynb",
     "Converts supported signals to absorbance, zero-pads spectra covering at least 80% of the reference range, and filters training spectra using roughness. Displays the audit and resulting spectra; statistical anomalies are not necessarily measurement errors."),
    ("01d_IR_Resampling_and_Balancing.ipynb",
     "Converts supported signals, pads acceptable partial spectra, and downsamples then upsamples to the shared grid. Capped oversampling increases rare-label representation in training only. Displays the audit and resulting spectra; resizing cannot restore missing spectral detail."),
]
for filename, preparation_summary in IR_VARIANTS:
    NOTES["02_ir_functional_groups/examples/" + filename] = [
        "Sets this example's signal-conversion, grid, and cleansing policies and its output folder. Frozen validation/test molecules are excluded before any training-data changes.",
        "Examines native units, missing values, array lengths, coverage, labels, duplicates, and statistical anomalies. The tables support the preceding prompt's hypothesis without automatically treating every flagged observation as incorrect.",
        preparation_summary,
        "Saves this strategy's training table alongside unchanged frozen evaluation files and a manifest. Reloading verifies finite normalized 500-point spectra, 12 binary labels, and separated identities; the printed folder can be selected in the static trainer.",
    ]

HTEM_SETUP = (
    "Loads notebook 01's shared features, target transforms, label masks, and grouped splits without rebuilding them. "
    "Each row represents process conditions at one substrate location; this model's results are saved in a separate experiment folder."
)
HTEM_EVALUATE = (
    "Predicts each location, converts results back to physical units, and shows train/validation/test parity plots. "
    "Saves the model, aligned predictions, metrics, and reproducibility records; evaluation uses only observed labels for each property."
)
HTEM_MAP = (
    "Reconstructs measured and predicted maps for one held-out deposition using each location's exact x/y coordinates. "
    "Side-by-side maps reveal property levels and spatial gradients without averaging different depositions or filling missing locations."
)
HTEM_INFER = (
    "Reloads the model and checks that it reproduces saved predictions, then predicts thickness/conductivity maps for editable process settings and substrate locations. "
    "No measured film properties are required as inputs; the GNN uses a full planned grid rather than independent rows."
)
HTEM_CURVES = (
    "Plots training/validation loss and learning-rate histories, marks the restored validation-minimum model, and reports why training stopped. "
    "Reaching the epoch limit alone does not demonstrate convergence."
)
NOTES["03_htem_conductivity/examples/02_HTEM_Process_Emulator_Vibe_Coding.ipynb"] = [
    HTEM_SETUP,
    "Fits a separate linear Ridge model for each property's observed training labels as the workshop baseline. This non-neural model has no epoch-based training history; the prompts guide subsequent model experiments.",
    HTEM_EVALUATE, HTEM_MAP, HTEM_INFER,
]
NOTES["03_htem_conductivity/examples/02a_HTEM_Kernel_Ridge.ipynb"] = [
    HTEM_SETUP,
    "Compares nine RBF kernel Ridge settings and selects by validation error. Exact kernel fitting includes numerical checks and can require substantial memory; it does not produce an epoch-based loss curve.",
    HTEM_EVALUATE, HTEM_MAP, HTEM_INFER,
]
NOTES["03_htem_conductivity/examples/02b_HTEM_Feed_Forward_Network.ipynb"] = [
    HTEM_SETUP,
    "Trains a feed-forward neural network on the shared per-location features using AdamW and cosine learning-rate decay. Missing targets are masked, and patience-based stopping restores the lowest-validation-loss model.",
    HTEM_CURVES, HTEM_EVALUATE, HTEM_MAP, HTEM_INFER,
]
NOTES["03_htem_conductivity/examples/02c_HTEM_Grid_GNN.ipynb"] = [
    HTEM_SETUP,
    "Builds and visualizes a neighbor graph for each full substrate grid, including unlabeled positions. Connections depend only on geometry and never link different depositions.",
    "Trains a graph neural network to predict properties at each substrate location rather than one value per deposition. Uses observed-label masks, cosine learning-rate decay, and patience-based stopping to restore the lowest-validation-loss model.",
    HTEM_CURVES, HTEM_EVALUATE, HTEM_MAP, HTEM_INFER,
]
NOTES["03_htem_conductivity/examples/02d_HTEM_Random_Forest.ipynb"] = [
    HTEM_SETUP,
    "Fits a separate 300-tree Random Forest for each property's observed training labels, using reproducible settings and bounded computational resources. Evaluation retains the same independent deposition splits as the other models.",
    HTEM_EVALUATE, HTEM_MAP, HTEM_INFER,
]

# Headings name the purpose of the following code cell, in the same order as NOTES.
HEADINGS: dict[str, list[str]] = {
    "00_Test_Installation.ipynb": [
        "Check the workshop kernel",
        "Verify required packages",
    ],
    "00_Ten_Steps_in_Prompts.ipynb": [
        "Load the practice dataset",
        "Practice: explore the dataset",
        "Practice: inspect data quality",
        "Practice: investigate unusual observations",
        "Practice: clean and save the data",
        "Practice: engineer model inputs",
        "Practice: compare feature sets",
        "Practice: export aligned modeling data",
        "Practice: define starter models",
        "Practice: split data reproducibly",
        "Practice: organize experiment tracking",
        "Practice: compare model families",
        "Practice: tune model settings",
        "Practice: test alternative features",
        "Practice: select and evaluate finalists",
        "Practice: visualize results and limitations",
    ],
    "01_boiling_point/00_BP_Get_Data_NISTChemPy_v2.ipynb": [
        "Configure NIST data acquisition",
        "Find compounds with phase-change data",
        "Extract SMILES and kelvin measurements",
        "Collect or reload raw boiling points",
        "Clean and export boiling-point data",
    ],
    "01_boiling_point/examples/reference_models/01_BP_Chemprop_GNN.ipynb": [
        "Prepare molecular splits and configure Chemprop",
        "Train Chemprop and select a checkpoint",
        "Plot loss histories and generate predictions",
        "Evaluate boiling-point predictions",
        "Predict boiling point from a SMILES string",
        "Optional practice: extend the prediction analysis",
    ],
    "01_boiling_point/examples/reference_models/02_BP_Foundation_Model.ipynb": [
        "Prepare shared splits and configure the regression head",
        "Generate frozen MoLFormer embeddings",
        "Train the regression head and restore selected weights",
        "Visualize loss and learning-rate histories",
        "Evaluate predictions in kelvin",
        "Predict with the frozen encoder and trained head",
        "Optional practice: explore another prediction",
    ],
    "02_ir_functional_groups/00_IR_Get_Data_NISTChemPy_v2.ipynb": [
        "Configure resumable NIST IR collection",
        "Define native-spectrum retrieval and functional-group labels",
        "Collect native spectra and checkpoint progress",
        "Preview recorded units and spectral arrays",
        "Verify the acquisition table before carpentry",
    ],
    "02_ir_functional_groups/examples/01_IR_Vibe_Coding_Data_Carpentry.ipynb": [
        "Load native spectra and reserve evaluation molecules",
        "Create the minimal uniform-spectrum baseline",
        "Save and validate the baseline training dataset",
    ],
    "02_ir_functional_groups/02_IR_Optimized_Training.ipynb": [
        "Load the selected dataset and standardize training inputs",
        "Define the classifier and configure optimization",
        "Train with robust stopping and checkpoint selection",
        "Select thresholds and evaluate functional-group predictions",
        "Classify an uploaded prepared spectrum",
    ],
    "02_ir_functional_groups/examples/03_IR_Compare_Carpentry.ipynb": [
        "Locate carpentry experiments and configure the comparison",
        "Validate completed runs and assemble their results",
        "Compare overall and per-group classification scores",
    ],
    "03_htem_conductivity/00_HTEM_Get_Data_NREL.ipynb": [
        "Load HTEM acquisition helpers",
        "Download compact process and scalar-property records",
        "Preview substrate locations and available labels",
    ],
    "03_htem_conductivity/01_HTEM_Process_Data_Carpentry.ipynb": [
        "Configure shared HTEM data preparation",
        "Audit process records, units, and location identities",
        "Build process-only features and grouped spatial splits",
        "Define optional reference-crystal embedding extraction",
        "Reconstruct and verify the frozen chemistry features",
        "Check split coverage, label masks, and full grids",
        "Examine training distributions and material coverage",
    ],
    "03_htem_conductivity/examples/03_HTEM_Compare_Architectures.ipynb": [
        "Verify and assemble saved architecture results",
        "Compare validation and test errors by property",
        "Plot held-out parity across all architectures",
    ],
    "scratch/HTEM_Feature_Selection_Study.ipynb": [
        "Configure the instructor chemistry-feature study",
        "Verify reference features and evaluate candidates",
        "Compare validation rankings and exploratory gains",
        "Publish the selected shared feature specification",
    ],
}

IR_PREPARATION_HEADINGS = {
    "01a_IR_Omit_Transmittance.ipynb": "Retain absorbance spectra and deduplicate molecules",
    "01b_IR_Convert_Transmittance.ipynb": "Convert transmittance and check functional-group labels",
    "01c_IR_Padding_and_Quality.ipynb": "Pad partial spectra and filter training roughness",
    "01d_IR_Resampling_and_Balancing.ipynb": "Resample spectra and balance rare training labels",
}
for filename, preparation_heading in IR_PREPARATION_HEADINGS.items():
    HEADINGS["02_ir_functional_groups/examples/" + filename] = [
        "Configure this carpentry strategy and protect evaluation data",
        "Examine native spectra and data-quality flags",
        preparation_heading,
        "Save and validate the prepared training dataset",
    ]

HTEM_RESULT_HEADINGS = [
    "Evaluate per-location predictions and save the model",
    "Reconstruct measured and predicted substrate maps",
    "Reload the model and predict an editable process grid",
]
HEADINGS["03_htem_conductivity/examples/02_HTEM_Process_Emulator_Vibe_Coding.ipynb"] = [
    "Load the shared process features and spatial splits",
    "Fit the linear Ridge process-emulator baseline",
    *HTEM_RESULT_HEADINGS,
]
HEADINGS["03_htem_conductivity/examples/02a_HTEM_Kernel_Ridge.ipynb"] = [
    "Load the shared process features and spatial splits",
    "Select an RBF kernel Ridge model using validation error",
    *HTEM_RESULT_HEADINGS,
]
HEADINGS["03_htem_conductivity/examples/02b_HTEM_Feed_Forward_Network.ipynb"] = [
    "Load the shared process features and spatial splits",
    "Train a per-location feed-forward process emulator",
    "Inspect neural training and checkpoint selection",
    *HTEM_RESULT_HEADINGS,
]
HEADINGS["03_htem_conductivity/examples/02c_HTEM_Grid_GNN.ipynb"] = [
    "Load the shared process features and spatial splits",
    "Construct geometry-only graphs of substrate grids",
    "Train a graph model with per-location predictions",
    "Inspect graph-model training and checkpoint selection",
    *HTEM_RESULT_HEADINGS,
]
HEADINGS["03_htem_conductivity/examples/02d_HTEM_Random_Forest.ipynb"] = [
    "Load the shared process features and spatial splits",
    "Fit a Random Forest for each observed property",
    *HTEM_RESULT_HEADINGS,
]


# The BP guide adapts the generic practice template to the prepared local dataset.
BP_GUIDE = "01_boiling_point/01_Ten_Steps_in_Prompts.ipynb"
NOTES[BP_GUIDE] = list(NOTES["00_Ten_Steps_in_Prompts.ipynb"])
NOTES[BP_GUIDE][0] = (
    "Locates the prepared BP Parquet, preferring the existing NIST acquisition over the packaged fallback, "
    "and previews SMILES and boiling points in kelvin without downloading data."
)
NOTES[BP_GUIDE][14] = (
    "This blank practice cell is for selecting finalists using validation results only; "
    "the test set stays sealed until optimization is finished in Step 9."
)
NOTES[BP_GUIDE][9] = (
    "This blank practice cell is for verifying the saved molecule-disjoint splits and "
    "training-only preprocessing, without creating new assignments to improve a score."
)
NOTES[BP_GUIDE][15] = (
    "This blank practice cell is for producing training/validation figures and preparing final "
    "test-evaluation plots, which must wait until the model is frozen after Step 9."
)
HEADINGS[BP_GUIDE] = list(HEADINGS["00_Ten_Steps_in_Prompts.ipynb"])
HEADINGS[BP_GUIDE][0] = "Load the prepared boiling-point dataset"
HEADINGS[BP_GUIDE][9] = "Practice: verify the frozen molecular splits"
HEADINGS[BP_GUIDE][14] = "Practice: analyze validation finalists"

def notebook_paths(root: Path = ROOT) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.ipynb")
        if not EXCLUDED.intersection(p.relative_to(root).parts)
        and not p.relative_to(root).as_posix().startswith(PRESERVED_DIRECTORIES)
    )


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
    headings = HEADINGS.get(relative)
    if headings is None or len(headings) != len(code_cells):
        raise ValueError(f"{relative}: review descriptive headings for {len(code_cells)} code cells "
                         f"(catalog contains {len(headings) if headings is not None else 0})")
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
            text = f"### {headings[note_index]}\n\nCode Summary: {notes[note_index]}\n"
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
    parser.add_argument("--check", action="store_true", help="Verify summaries/collapse metadata without writing")
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
        print("Presentation summaries/settings need updating:", *changed, sep="\n- ")
        return 1
    for path, updated in updates:
        path.write_text(json.dumps(updated, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{'Checked' if args.check else 'Prepared'} {code_count} code cells in {count} notebooks; "
          f"{len(changed)} notebook(s) {'need updates' if args.check else 'updated'}. No code executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
