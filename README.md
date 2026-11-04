# Sprint Intelligence

Predictive and prescriptive analytics framework for Agile sprint planning. The dashboard estimates three issue-level risks, shows model explanations, and uses an expected-utility optimiser to recommend **Include**, **Monitor**, **Split**, or **Defer** for candidate sprint items.

## What is included

| Folder / file | Purpose |
|---|---|
| `dashboard/` | Runnable web application: Python backend and browser frontend. |
| `llm/` | Optional local-Qwen mitigation guidance and approved mitigation strategies. The dashboard still works without Qwen. |
| `models/` | Saved multimodal neural Resolution Risk model and its local MiniLM text encoder. |
| `data/processed/` | Clean v6.1/v6.2 derived datasets, source-aware split, selected classical models, calibration artefacts, and small demo CSVs. |
| `scripts/evaluation/` | Calibration, candidate-scoring, threshold and optimiser-evaluation scripts. |
| `notebooks/` | Colab-ready source exports for baseline and neural variants, feature audit/selection, final selection and aggregation. Some include Colab commands such as `!pip install`, so run them in Colab rather than directly in PowerShell. |
| `train_resolution_neural_colab.py` | Self-contained Colab/local training script for the saved Resolution Risk neural model. |
| `tests/` | Automated checks for optimisation, mitigations and candidate scoring. |

## Requirements

* Windows, macOS, or Linux
* Python 3.11 is recommended
* Internet is required only while installing Python packages. The saved neural text encoder is included locally, so demo scoring does not download a model.

## Quick start: run the dashboard

Open PowerShell (Windows) or a terminal in the project folder.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r dashboard\requirements.txt
python dashboard\backend\app.py
```

If PowerShell blocks environment activation, run this once in the same terminal and then repeat the activation command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Open <http://127.0.0.1:8000> in a browser. Stop the server with `Ctrl+C`.

### Demo input files

Use the upload control in the dashboard.

* `data/processed/demo_candidates_v6_2.csv` — scored by the saved models and intended for per-issue explanations and dependencies.

The second file is for user-interface demonstration. It contains illustrative risk scores; it must not be represented as a new model evaluation set.

## Run automated checks

With the virtual environment activated:

```powershell
python -m unittest tests.test_optimizer tests.test_mitigation tests.test_candidate_scoring
```

## Reproduce the saved Resolution Risk neural model

The training script uses `training_dataset_v6_1.csv`, the fixed source-aware split and `model_feature_columns_v6.json` from `data/processed/`. Its default paths expect those three files in the current directory, so copy them into a temporary training directory with the script or edit the three `Path(...)` values at the top of the script.

The script can run in Google Colab (recommended for training speed) or locally after installing `dashboard/requirements.txt`:

```powershell
python train_resolution_neural_colab.py
```

It exports `resolution_neural_bundle_v6_1/`, containing the Keras model, preprocessing pipeline, vocabulary, local MiniLM encoder, training history and result summary. Training uses a fixed seed, but small variation is still normal on different hardware.


## Model overview

| Task | Selected model in dashboard | Output |
|---|---|---|
| Requirement Volatility Risk | Logistic Regression | Probability of significant future requirement change. |
| Issue Resolution Risk | Multimodal neural network | Probability of delayed resolution. Its structured Jira, MiniLM-text and workflow-sequence branches are combined before prediction. |
| Issue Reopen Risk | Random Forest | Probability that a resolved issue is reopened. |

The optimiser uses a separately calibrated Resolution Risk probability when ranking a full sprint portfolio. It maximises expected value subject to capacity and dependencies. It is not a claim that the model causes a future outcome.

## Optional local mitigation generation

The application works without a language model, using the approved mitigation strategies in `llm/approved_strategies.py`. For locally generated, issue-specific wording, install [Ollama](https://ollama.com/) and run:

```powershell
ollama pull qwen3:1.7b
```

Keep Ollama running before starting the dashboard. No token is required and no issue data is sent to an external API by this configuration.
