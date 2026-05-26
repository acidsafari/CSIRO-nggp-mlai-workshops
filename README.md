# NextGen ML/AI Workshops

This repository contains the notebooks and materials for the NextGen ML/AI Workshops.

The workshops introduces core machine learning concepts through practical examples. Each workshop is delivered using Jupyter notebooks so you can run code, inspect results, change parameters, and experiment.

These workshops are designed to make foundational ML/AI concepts accessible to newcomers while hopefully facilitating a deeper understanding for experienced practitioners.

Before our first session, please work through the README and make sure you can access and run the notebook under the `workshop0-setup` branch.

## How the Workshops Work

Each workshop is structured around guided notebooks that serve as shared reference points for discussion, experimentation, and collective reasoning during the session.

The notebooks include:

- short explanations
- runnable code cells
- visual outputs
- small experiments
- questions for discussion
- optional extension tasks

You can run the notebooks in [Google Colab](https://colab.research.google.com/), JupyterLab, VS Code, or another notebook environment.

Similar to the coding bootcamps we will provide a branch per workshop session.

## Running the Notebooks

### Option 1: Use Google Colab

Google Colab is the easiest option. It runs in your browser, so you do not need to install Python locally.

To open the first workshop notebook in Colab:

1. Go to [Google Colab](https://colab.research.google.com/).
2. Select **File > Open notebook**.
3. Choose the **GitHub** tab.
4. Paste the workshop repository URL.
5. Choose the branch for your workshop session.
6. Open:

```text
notebooks/00_learning_from_data.ipynb
```

When you run a notebook in Colab, run the setup cell at the top first. Colab opens a single notebook from GitHub, but it does not automatically clone the rest of the repository. The setup cell clones the selected workshop branch into `/content/nextgen2026-mlai-workshops` and installs the repository package so notebook code can import workshop utilities.

### Option 2: Run Locally

For the base development environment, follow the setup process from the NextGen Coding Bootcamp:

https://github.com/nextgenerationgraduatesprogram/nextgen2026-coding-bootcamp

That guide covers the common setup we use across NextGen repositories: WSL/Ubuntu for Windows users, VS Code, Python, Git, and `uv`.

### Step 1: Clone the repository and open it

Run:

```bash
git clone <repository-url>
cd <repository-name>
```

Use the workshop branch provided for your session. 

```bash
git fetch origin --prune
git branch -r
git switch --track origin/<workshop-branch>
git branch --show-current
```

Open in VS Code:

```bash
code .
```

Workshop 1 starts at:

```text
notebooks/00_learning_from_data.ipynb
```

The remaining numbered notebooks continue the workshop sequence.

### Step 2: Install the workshop environment

From the repository root, run:

```bash
uv sync
```

This installs the Python environment and all dependencies needed for the workshop notebooks.

### Step 3: Open the notebook

In VS Code, open:

```text
notebooks/00_learning_from_data.ipynb
```

The numbered notebooks are the shared reference points for the workshop discussion, demonstrations, and experiments.

### Step 4: Select the notebook kernel

In the top right of the notebook window, click **Select Kernel**.

Choose the Python environment for this repository:

```text
.venv/bin/python
```

If you are using Windows with WSL, make sure you select the kernel from the WSL/Linux environment, not a Windows Python installation.

Once the kernel is selected, run the notebook cells in order.

## Using Workshop Utilities

Shared Python utilities should live in the package under:

```text
src/nextgen2026_mlai_workshops/
```

After running `uv sync` locally and selecting the `.venv/bin/python` kernel, notebooks can import those utilities normally:

```python
import nextgen2026_mlai_workshops
```

You do not need to add `src/` to `sys.path` for local development. `uv sync` installs this repository in editable mode, so changes under `src/` are available to the notebook kernel.

For Colab, keep the first setup cell in each workshop notebook. It clones this repository branch and adds the cloned `src/` directory to `sys.path` for the current kernel. If you create a new workshop branch, update the branch name in that setup cell.

Runtime dependencies needed by workshop code should be added to `pyproject.toml` under `[project] dependencies`. Local-only tools such as Jupyter and notebook kernels should stay in the `dev` dependency group.

If you see `ModuleNotFoundError: No module named 'nextgen2026_mlai_workshops'`, run the first setup cell in the notebook. Locally, also confirm the selected kernel is `.venv/bin/python`; in Colab, confirm the setup cell completed without a clone or install error.
