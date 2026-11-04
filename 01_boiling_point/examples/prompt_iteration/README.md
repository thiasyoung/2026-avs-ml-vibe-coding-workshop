# Prompt strategies in action

The [strategy toolbox in the 02 workbook](../../02_BP_Vibe_Coding_Workbook.ipynb) lists levers you
can add to a prompt. These four runs show them in use: the same task (SMILES → reported
boiling point in kelvin, following the [ten-step template](../../01_Ten_Steps_in_Prompts.ipynb)),
given to one assistant with prompts that pull different levers. Each notebook is the executed
result, with saved outputs, preserved as generated. Treat them as worked examples of
how a prompt shapes the *process and deliverables*, not as a benchmark.

## Which levers each prompt pulled

| Lever | [P0](prompts/P0_zero_shot_prompt.txt) | [P1](prompts/P1_one_shot_prompt.txt) | [P2](prompts/P2_one_shot_prompt.txt) | [P3](prompts/P3_one_shot_prompt.txt) |
| --- | :---: | :---: | :---: | :---: |
| Hypothesis first: any creative idea must beat the baseline | yes | yes | yes | yes |
| Name the guardrails (no leakage, sealed test set) | yes | yes | yes | yes |
| Anchor to the ten-step template | no | yes | yes | yes |
| Role and audience | audience only | yes | yes | yes |
| Plan before code | no | no | yes | yes |
| Small verified steps | no | no | yes | yes |
| Ask for specific checks (assertions, fresh-process reload, applicability) | light | some | yes | yes |
| Self-consistency at key decisions | no | no | yes | yes |
| Critique and revise | no | no | no | yes |
| Prompt length (words) | about 400 | about 1,400 | about 2,600 | about 3,200 |

Executed notebooks: [P0](notebooks/P0_BP_Zero_Shot.ipynb), [P1](notebooks/P1_BP_One_Shot.ipynb),
[P2](notebooks/P2_BP_One_Shot.ipynb), [P3](notebooks/P3_BP_One_Shot.ipynb).

Based on reading each prompt and scanning its text for these instructions. P3 also
uses a named structure (Context → Role → Ask → Format → Tests) and names several prompting
techniques outright.

## What showed up in the output

All four notebooks built a median baseline, compared several models, selected on
validation data, evaluated the test set once, and ended with a prediction function. A scan of the
notebook text shows where they differ:

- **P0** has a simple element and size warning, but no similarity-based applicability
  check, permutation importance, run manifest or saved inference module.
- **P1** adds applicability checks, permutation importance, error breakdowns, a manifest
  with checksums, and a saved inference module.
- **P2** adds self-consistency analysis and reloads the saved model in a fresh process.
- **P3** adds the most documentation and also reloads in a fresh process; its extra
  length went into process and explanation more than into modelling.

Accuracy was similar across the four. Each run applied its own cleaning rules and split,
so test sets differ (317–321 molecules) and the scores cannot rank the prompts. The more
useful lesson is that **more detailed prompts mainly changed rigor, documentation and
extras**. If you want to know whether a lever improves *your* result, change one lever
and keep your splits and evaluation fixed.

## Mine the strategies yourself

[Compare prompt strategies](Compare_Prompt_Strategies.ipynb) holds one copyable prompt that
asks your assistant to identify the strategies each prompt used and what each added to
the output, with evidence from these files. Running the notebook itself calls no LLM.

## Generated artifacts are not stored here

Each run also wrote models, figures, per-experiment predictions and helper modules
under `01_boiling_point/outputs/`. Those folders are git-ignored, so a fresh checkout has
only the notebooks and prompts. The notebooks regenerate their artifacts when re-executed,
and their saved outputs show the results without rerunning. Some code cells and
comments still mention `prompt_test/` and the original output folder names because the
notebooks are preserved exactly as generated.
