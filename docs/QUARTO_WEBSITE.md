# Workshop reading website

Quarto turns the existing Markdown and Jupyter notebooks into a searchable website.
Code is folded behind **Show Python code**; explanations, tables, and saved plots
remain visible. The original notebooks remain the executable source of truth.

## What gets published

The explicit list in `_quarto.yml` includes the getting-started notebooks, setup
guide, all three modules, their worked examples, and module guides. It excludes
instructor scratch work, archives, generated notebook copies, datasets, model
weights, and the local environment. Notebook downloads are byte-for-byte copies
of the source files, not newly executed notebooks.

Both `execute.enabled: false` and the workflow's `--no-execute` prevent acquisition,
Python commands, and training from running during publishing. Saved outputs are
displayed as they stand; an unexecuted notebook does not acquire invented plots.
To update a result, run and save that notebook using the workshop environment,
review its outputs, then commit it. A working website does not prove the models
were rerun or their results improved. See [Quarto's notebook rendering guidance](https://quarto.org/docs/computations/python.html).

## Preview locally (Windows, macOS, or Linux)

Install the separate [Quarto CLI](https://quarto.org/docs/download/). The CI build
is pinned to **1.9.38**; use that version to reproduce it. Quarto is not a Python
package installed by `uv sync`. The optional Quarto VS Code extension is convenient,
but is not required and does not replace the CLI. Students only reading the site
need neither Quarto nor Python; running notebooks still requires the workshop setup.

### Open a local reading preview

Code Summary: Builds and serves the reading website from saved outputs without executing notebook cells.

From the workshop repository root, in PowerShell or a macOS/Linux terminal:

```text
quarto preview --no-execute
```

Open the localhost address printed in the terminal. Press Ctrl+C to stop.

### Build and check the complete website

Code Summary: Renders every listed page and checks navigation, folded code, saved outputs, and unchanged notebook downloads.

```text
quarto render --no-execute
uv run --locked python scripts/ci/validate_website.py
```

The generated `_site/` and `.quarto/` directories are ignored by Git. Do not commit
them or add generated HTML to the source notebooks. The validator itself uses only
Python's standard library; Pages CI does not install UV, PyTorch, Jupyter, or data.
It checks local file links, not remote websites or dynamic HTML fragments.
Some existing paper graphics and notebook widget scripts use external URLs, so
the reading site is not a fully offline bundle. Ordinary saved plot images are
included locally; widget displays are not live Python kernels.

## Activate GitHub Pages (one-time repository-owner steps)

1. **Review publication/privacy first.** Pages usually publishes a public website,
   including expanded code and saved outputs. Folding is not redaction. Review
   notebook text/output for private information before enabling publishing. This
   setup does not change repository visibility or upload anything by itself.
2. GitHub Free supports Pages for **public** repositories, not private repositories.
   If this repository is still private on Free, keep using the local preview or
   choose a plan that supports private-source Pages. Making the repository public
   is a separate owner decision; do not do it solely to troubleshoot a build.
   A private source repository does not automatically make the website private.
   See [GitHub Pages availability](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site).
3. Commit these source/configuration files and get them into **main** using your
   normal Git workflow. Keep the original `.ipynb` files and `uv.lock` in Git.
4. In the repository, open **Settings → Pages → Build and deployment → Source**
   and select **GitHub Actions**. Do not create another suggested workflow; this
   repository already has `.github/workflows/publish-website.yml`.
5. Ensure Actions permits GitHub's official actions and `quarto-dev/quarto-actions`.
   You can leave the default repository token permissions read-only. Only the
   deployment job requests `pages: write` and `id-token: write`; it does not receive
   repository write access, an API key, or a personal access token.
6. In **Actions → Publish workshop website**, choose **Run workflow → main** for
   the first deployment. If the `github-pages` environment requires review, approve
   that deployment or adjust its settings yourself. Its allowed deployment branches
   should include main, not development branches.
7. After **Render saved notebook outputs** and **Deploy to GitHub Pages** succeed,
   open the URL shown in the deployment job or **Settings → Pages → Visit site**.

The expected address, once activated, is
[the workshop reading website](https://thiasyoung.github.io/2026-avs-ml-vibe-coding-workshop/).
See [GitHub's custom Pages workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Updating and using it

- Pushes to main rebuild/deploy. Pull requests into main build and validate only;
  they cannot deploy the site. Dev-branch work is not the published workshop until
  it is merged into main. This does not change the Issue-triggered Codex workflow.
- The website is built independently of Codex and uses no OpenAI API calls; the
  separate Codex startup failure does not block a normal Pages build.
- Students can read in the browser, expand code, and download the repository for
  execution in VS Code. The site is not a notebook server or a Python runtime.
  Individual notebooks need their helper files and datasets from the repository.
- To make a PDF, use browser **Print → Save as PDF**. Expand the code you want
  included first. This is a static printout of one page, not an interactive notebook
  or a separately typeset whole-workshop PDF. No LaTeX installation is needed.
- If you fork/rename the repository, update `website.site-url`, `website.repo-url`,
  GitHub links in `index.qmd`/the footer, and `SITE_PATH` in the validator.
- For a new participant notebook, add its literal path to the render list, sidebar,
  and resource/download list. Resource paths must start with `/` to anchor them at
  the project root. Do not use wildcards or recursive basename searches: those can
  match nested archives and checkpoints. Tests check all three lists.

No Pages settings or deployments are activated by a local render. A 404 at the
expected URL before the first successful deployment is normal. Missing saved
outputs require running/saving the notebook, not enabling execution in Pages CI.
