"""Temporary instructor study, separate from the participant's numbered flow."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md("""
    # Temporary instructor study: choose and freeze HTEM chemical features

    Students change **models**, not the feature representation. This instructor-only study compares
    nine predeclared feature sets with two fixed selectors (Ridge and Random Forest). Every trial
    uses the same original grouped training/validation data, observed-label masks and target scaling.
    **No test artifact or test score is read during selection.** The previously inspected test set is
    not a new untouched confirmation set; this study does not claim universal superiority.

    Candidates: existing chemistry with/without DOS, curated Magpie-style descriptors, published
    MEGNet composition vectors, real pretrained ALIGNN reference-crystal vectors, and combinations.
    Selection: mean relative validation RMSE across the two selectors, normalized by their existing-DOS
    baselines, with equal weights. A lower score is better. The selector models are not new workshop
    architecture defaults; notebook 02 stays linear Ridge.
    """),
    code("""
    from pathlib import Path
    import sys, json
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / '03_htem_conductivity/lib/htem_feature_study.py').exists())
    MODULE = ROOT / '03_htem_conductivity'
    sys.path.insert(0, str(MODULE / 'lib'))
    import htem_feature_study as study
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
    display(pd.DataFrame([{'candidate': name, 'added_blocks': ', '.join(blocks) or 'none'}
                          for name, blocks in study.CANDIDATES.items()]))
    print('Fixed selectors:', study.ESTIMATORS)
    """),
    md("""
    ## Independent references, not measured film information

    - **Curated descriptors:** stoichiometry norms, periodic row/column, Mendeleev number, covalent
      radius, elemental melting-temperature statistics and valence-electron statistics/fractions.
      Elemental melting points are not an inferred melting point of the compound or mixture.
    - **Composition:** stoichiometric mean of published 16-D formation-energy-trained MEGNet element
      vectors. This is a real pretrained element representation, **not a full Roost/AtomSets model**.
    - **Crystal:** official frozen ALIGNN formation-energy checkpoint; mean atom readout before its
      scalar head. Extract from all exact-composition JARVIS reference structures. Fit 16-component PCA
      on **unique active training compositions only**, then transform other references unchanged.
      These are reference phases, not verified HTEM source targets or deposited-film structures.
    - **DOS:** existing pinned spin-up reference shape summaries, not film spectra or physical mixture DOS.

    Source slots and powers remain paired. No labels, measured film composition, spectra, IDs,
    source-gun geometry guesses, or target-specific test information enter any feature.

    Instructor extraction (already run for the supplied compact cache):
    `scratch/build_htem_feature_candidates.py` in the isolated extraction environment. The workshop
    environment does not need ALIGNN/DGL/JARVIS packages. See the module README for reproduction.
    References: [MEGNet](https://doi.org/10.1021/acs.chemmater.9b01294),
    [ALIGNN](https://doi.org/10.1038/s41524-021-00650-1),
    [Magpie](https://doi.org/10.1038/npjcompumats.2016.28).
    """),
    code("""
    reference = json.loads(study.REFERENCE.read_text(encoding='utf-8'))
    crystals = reference['provenance']['crystal_embedding']
    print('Source compositions:', len(reference['compositions']))
    print('Real atomic-graph reference extractions:', crystals['reference_structures'])
    print('Reported extraction failures:', len(crystals['failures']))
    print('Frozen model:', crystals['checkpoint'])
    print('Reference/checkpoint archive SHA256:', crystals['archive_sha256'])
    results, ranking, decision = study.run_study()
    """),
    md("""
    ## Validation results and uncertainty

    Lower is better. Scores average the two training-standardized log-target RMSEs, so conductivity
    does not dominate merely because its numerical units differ. The final selection averages each
    selector's score relative to its own existing-DOS baseline. The paired group-bootstrap interval
    resamples entire validation recipe groups, not individual grid points. It is descriptive after
    selection, not an independent significance test. Feature improvements may differ by architecture.
    """),
    code("""
    display(results.round(5))
    display(ranking.round(5))
    print('Selected:', decision['selected_candidate'])
    print('Exploratory relative-gain interval:', decision['bootstrap']['relative_gain_interval_95'])
    pivot = results.pivot(index='candidate', columns='estimator', values='validation_score').loc[ranking.candidate]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pivot.plot.barh(ax=axes[0], color=['#377eb8', '#66a61e'])
    axes[0].set(xlabel='Validation mean standardized RMSE', ylabel='')
    axes[0].invert_yaxis()
    axes[1].barh(ranking.candidate, ranking.mean_relative_validation_score, color='#377eb8')
    axes[1].axvline(1, color='gray', linestyle='--', label='Existing DOS baseline')
    axes[1].set(xlabel='Equal-selector mean relative validation score', ylabel='')
    axes[1].invert_yaxis()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(study.STUDY / 'feature_selection.png', bbox_inches='tight')
    plt.show()
    """),
    md("""
    ## Explicit instructor deployment

    Publish the chosen feature specification and compact offline source table. This does **not**
    train on validation, change splits, or read the test set. Rerun notebook 01 afterward to create
    the shared frozen matrices, followed by all five architecture notebooks and comparison 03.
    No architecture may fit its own featurizer. The deployed files retain selection provenance.
    """),
    code("""
    deployed = study.deploy_choice()
    print('Published fixed feature specification:', deployed['selected_candidate'])
    print('Next: run participant 01, all five 02 architectures, then 03.')
    """),
]
notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python (workshop .venv)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
})
nbf.write(notebook, ROOT / 'scratch/HTEM_Feature_Selection_Study.ipynb')
