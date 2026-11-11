# MLOps Pipeline — Bank Marketing Classification

End-to-end ML pipeline that trains an XGBoost binary classifier with SageMaker, registers it in the Model Registry, and deploys it to a real-time endpoint on approval.

**Prerequisite:** The [DataOps pipeline](../dataops-pipeline/) must run first to create the `bank_mktg_<stage>.campaign_results` table.

## Architecture

Three workflows make up the MLOps flow:

**`training_pipeline`** — trains and registers a model version:

```text
preprocess (GlueJobOperator)
  → train_model (SageMakerTrainingOperator)
    → evaluate_model (SageMakerProcessingOperator)
      → clarify_analysis (bias + SHAP replacement → …/evaluation/clarify/)
      → register_model (SageMakerRegisterModelVersionOperator → PendingManualApproval)
        → create_model_card (stack-bank-mktg-model-card in Studio)
```

**`deploy_pipeline`** — deploys the approved model to an endpoint (event-driven):

```text
deploy_model (SageMakerProcessingOperator → stack-prediction-<stage> endpoint)
```

**`drift_check_pipeline`** — compares live raw source to the last-read snapshot (dev gate):

```text
data_drift_check (GlueJobOperator → drift_reports/.../result.json)
```

On **E2E Deploy MLOps Pipeline** (`workflow_dispatch`), `check_drift` (default on) runs this DAG after deploy and only calls `training_pipeline` when `drift_detected=true`. Set `run_training` to force a retrain and skip the gate. Missing reference/result counts as drift so the first train can bootstrap.

**Compared datasets:** live `--source_path` (`stack/data_pipeline/data/stack-sample.csv` today; S3 table later) vs `features/drift_reference/` (parquet snapshot written on each successful preprocess from that same source).

| Step | Operator | Description |
| ---- | -------- | ----------- |
| `preprocess` | GlueJobOperator | Feature engineering from DataOps transformed data; snapshots raw `SOURCE_PATH` → `drift_reference/` |
| `train_model` | SageMakerTrainingOperator | XGBoost training via `train_xgboost.py` (script mode) with MLflow tracking |
| `evaluate_model` | SageMakerProcessingOperator | Batch evaluation on the held-out test set (`evaluate_model.py`) |
| `clarify_analysis` | SageMakerProcessingOperator | Bias + SHAP via Clarify replacement (`run_clarify.py`; AWS Clarify closed to new accounts); HTML/JSON under `…/evaluation/clarify/` — **not** the AWS Clarify Studio app |
| `register_model` | SageMakerRegisterModelVersionOperator | Registers a version in `stack-prediction-models` as `PendingManualApproval` |
| `create_model_card` | SageMakerProcessingOperator | Attaches package Model Card + Clarify metrics to the registry version (Studio **Models → version → Evaluate/Audit**); also upserts standalone `stack-bank-mktg-model-card` |
| `deploy_model` | SageMakerProcessingOperator | Creates/updates the prediction endpoint (`deploy_model.py`) |
| `data_drift_check` | GlueJobOperator | PSI / share-delta: live source vs `drift_reference/`; writes `result.json` |

Model **approval** drives deployment: approving a version fires EventBridge → Lambda → GitHub Actions → `deploy_pipeline`. See the [parent README](../../README.md#deploy-trigger-behavior) for details.

## Prerequisites

```bash
pip install aws-smus-cicd-cli
export DEV_DOMAIN_REGION=<your-region>      # required
export DEV_PROJECT_NAME=<your-dev-project>  # optional (default: e2e-data-stack-ml-ops-dev)
```

An MLflow tracking server must exist (the manifest bootstrap creates the MLflow connection; `MLFLOW_TRACKING_SERVER_NAME` defaults to `smus-integration-mlflow-us-east-1`). Account ID and domain are resolved at runtime — see the [parent e2e example README](../../README.md#prerequisites) for the full variable list and CI/CD setup.

## Deploy and Run

```bash
# Validate connectivity
aws-smus-cicd-cli describe --manifest manifest.yaml --targets dev --connect

# Deploy code and workflows to S3 + MWAA
aws-smus-cicd-cli deploy --manifest manifest.yaml --targets dev

# Run the training pipeline
aws-smus-cicd-cli run --manifest manifest.yaml --targets dev --workflow training_pipeline

# Monitor execution
aws-smus-cicd-cli monitor --manifest manifest.yaml --targets dev --live
```

> The `train_model` task downloads `train_model_sourcedir.tar.gz` from the project's shared S3 location. The CLI packages this automatically during `deploy` from the manifest's `train_model_sourcedir` storage item (`compression: tar.gz`, bundling `train_xgboost.py` + `requirements.txt`) — no separate build step is needed. Each script-mode SageMaker job gets its own per-job bundle named after the task that consumes it.

## Project Structure

```text
mlops-pipeline/
├── manifest.yaml                        # Deployment manifest (dev/test/prod stages)
├── workflows/
│   ├── training_pipeline.yaml           # preprocess → train → evaluate → register
│   ├── deploy_pipeline.yaml             # deploy_model → endpoint (event-driven)
│   └── drift_check_pipeline.yaml        # data drift gate (dev / GHA)
└── src/
    ├── feature_engineering.py           # Glue preprocess job (+ drift_reference snapshot)
    ├── glue_data_drift_check.py         # Glue PSI / share-delta drift check
    ├── train_xgboost.py                 # XGBoost training script (SageMaker script mode)
    ├── evaluate_model.py                # Batch evaluation processing script
    ├── deploy_model.py                  # Endpoint deploy processing script (packs realtime inference.py)
    ├── inference.py                     # Realtime InvokeEndpoint enrichment (score + named features + model + timestamp)
    ├── feature_schema.py                # Feature column names/order for enrichment
    ├── requirements.txt                 # Training container deps (scikit-learn pinned to container version)
    └── notebooks/
        ├── evaluate_model.ipynb         # Interactive evaluation
        └── validate_mlops.ipynb         # Post-pipeline validation
```

## Resource Naming

Stage-prefixed, timestamped names keep runs isolated across stages and reruns:

```text
Training job     : stack-xgb-<timestamp>
Model package    : stack-prediction-models (registry group, versioned)
Endpoint         : stack-prediction-<stage>   (e.g. stack-prediction-dev)
```

## Where to see Model Card + Clarify

After a successful `training_pipeline` run:

1. **Model Card (GUI)** — Unified Studio / SageMaker Studio → **Models** → **Registered models** → `stack-prediction-models` → open the **latest version** → **Evaluate** / **Audit**. That content is the package Model Card from `create_model_card` (not a separate “Model cards” menu item on the Models list).
2. **Clarify report (files)** — project S3 / Files → `shared/stack/mlops_pipeline/models/evaluation/clarify/<stack-clarify-…>/report.html` (and `clarify/latest/clarify_summary.json`). There is **no** AWS Clarify processing UI for new accounts; the HTML is the replacement.
3. **CLI check** — `aws sagemaker describe-model-package --model-package-name <arn>` and look for `ModelCard` + `CustomerMetadataProperties.clarify_report_html`.

## Validation

After the pipeline completes, use `src/notebooks/validate_mlops.ipynb` to verify the training job status, the registered model version, endpoint health (`InService`), and a sample inference.

**Realtime / batch responses:** after `deploy_pipeline`, `InvokeEndpoint` returns JSON rows with `score`, `features` (named map matching `feature_schema.py` / FE columns), `model_name`, `model_version`, and `inference_timestamp`. Batch evaluate appends the same fields to Delta `predictions_enriched` (`features` as a JSON object string).

## Troubleshooting

**Training fails importing scikit-learn (`KeyError: '__reduce_cython__'`):** `requirements.txt` pins `scikit-learn` to the container's preinstalled version to avoid a numpy/Cython ABI mismatch — don't bump it past what the `sagemaker-xgboost` container ships.

**Endpoint already exists:** the deploy step updates the existing endpoint in place when one with the same name exists.

**No promote after approval:** the promote workflow runs from the default branch via `repository_dispatch`, and the repo must have **Issues enabled** for the approval gates. See the [parent README](../../README.md#cicd).
