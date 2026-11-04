# Candidate scoring and mitigation

The dashboard accepts either an existing risk register or a candidate-feature CSV.
Candidate CSVs must contain the 35 safe v6.1 input features, `source_collection`,
and `issue_key`. The supplied `data/processed/candidate_sprint_items.csv` is one
example. The server scores Logistic Regression for volatility, the saved multimodal
neural model for resolution, and Random Forest for reopen risk. It never reads
uploaded model files; joblib artifacts and the neural bundle must be trusted local
project files.

## Run

The current saved models were trained with scikit-learn 1.6.1. Use Python 3.12
and a project virtual environment, then install `dashboard/requirements.txt`.
For example on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r dashboard\requirements.txt
.\.venv\Scripts\python.exe dashboard\backend\app.py
```

Open `http://127.0.0.1:8000`, upload the candidate CSV, and click an issue to
see its three explanations. The linear model uses contributions relative to
the transformed mean of the original training split, in log-odds. The random
forests use Tree SHAP relative to the training-path reference, in probability
units. Contributions are associations learned by the models, not causes.

The linear reference requires `training_dataset_v6_1.csv` and
`source_aware_splits_v6_1_fixed.csv` in `data/processed`. SHAP explanations are
checked against the model output. Unseen source collections are flagged; Apache
candidate predictions are exploratory, not validated cross-project performance.
Missing effort estimates remain unknown, rather than being silently set to one.

## Neural Resolution Risk explanation

The Resolution Risk neural model uses branch-level local ablation. For the selected
issue, the dashboard calculates the normal score, then replaces one branch at a
time with the training-set reference saved in the model bundle:

- structured first-30-day signals;
- cutoff-safe issue text embedding; and
- first-30-day workflow-event sequence (replaced with an empty sequence).

The displayed contribution is `full predicted probability - ablated predicted
probability`. A positive value means that branch raises the Resolution Risk score
relative to its reference; a negative value means it lowers it. These are local
sensitivity explanations, not causal claims. Branch contributions can interact, so
they do not necessarily sum to the final probability.

Upload of an existing risk register remains supported. Any supplied `top_factors`
are displayed as unverified input rather than as calculated model explanations.

## Grounded LLM mitigation layer

The dashboard keeps the planning action deterministic: `Include`, `Monitor`,
`Split`, or `Defer` is produced by the risk aggregation rules, not by the LLM.
On request, the local Qwen/Ollama mitigation layer receives the selected highest
risk score and the associated model explanation context. It can select only one
or two IDs from `llm/approved_strategies.py`; the dashboard displays only the
canonical approved text. Invalid, unavailable, or ungrounded responses use a
task-specific approved fallback and are visibly labelled as such. The LLM never
updates Jira, commits work, or changes a risk score or planner action.

## Mitigation

The recommendation button uses a locally running Ollama model, configured via
`OLLAMA_MODEL` (default `qwen3:1.7b`) and `OLLAMA_BASE_URL` (default
`http://127.0.0.1:11434`). The model selects applicable approved strategy IDs;
the server validates the IDs and displays canonical strategy text only. A
malformed response or an unavailable Ollama server yields an explicitly labelled
approved fallback. Valid IDs do not prove the model's choice is suitable; a
Scrum Master must review suggestions. They are not applied to Jira automatically.

The previously evaluated neural experiment did not export a trained checkpoint,
text encoder, sequence vocabulary, and preprocessing pipeline. Consequently this
dashboard uses the saved classical models; it does not claim to provide neural
Integrated Gradients. A neural explanation requires saving and integrating that
complete inference pipeline first.
# Neural Resolution Risk explanation

The Resolution Risk neural model uses branch-level local ablation. For the selected issue, the dashboard calculates the normal score, then replaces one branch at a time with the training-set reference saved in the model bundle:

- structured first-30-day signals;
- cutoff-safe issue text embedding; and
- first-30-day workflow-event sequence (replaced with an empty sequence).

The displayed contribution is `full predicted probability - ablated predicted probability`. A positive value means that branch raises the Resolution Risk score relative to its reference; a negative value means it lowers it. These are local sensitivity explanations, not causal claims. Branch contributions can interact, so they do not necessarily sum to the final probability.

