"""Evaluate Model - SageMaker Processing Script.

Finds the latest training job, creates an ephemeral SageMaker Model for Batch
Transform, runs predictions on the test set, computes evaluation metrics
(AUC, accuracy, F1), logs results to MLflow, then deletes the Model so only
the Model Registry group retains persisted versions.

This script is designed to run as a SageMaker Processing job via the
SageMakerProcessingOperator in the training pipeline workflow.

Arguments (passed via --arg=value):
    --training-job-prefix: Prefix for training job names (e.g., stack-xgb)
    --model-prefix: Prefix for model names (e.g., stack-model)
    --image-uri: XGBoost container image URI
    --execution-role-arn: SageMaker execution role ARN
    --region: AWS region
    --test-data-path: S3 URI to test data directory
    --output-path: S3 URI for evaluation output
    --tracking-server-arn: MLflow tracking server ARN (optional)
    --experiment-name: MLflow experiment name (optional)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import boto3
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--training-job-prefix", required=True)
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--execution-role-arn", required=True)
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    parser.add_argument("--test-data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--tracking-server-arn", default="")
    parser.add_argument("--experiment-name", default="")
    return parser.parse_args()


def find_latest_training_job(sm, prefix):
    """Find the most recent completed training job matching the prefix."""
    resp = sm.list_training_jobs(SortBy="CreationTime", SortOrder="Descending", MaxResults=20)
    matched = [
        j for j in resp["TrainingJobSummaries"]
        if j["TrainingJobName"].startswith(prefix) and j["TrainingJobStatus"] == "Completed"
    ]
    if not matched:
        raise ValueError(f"No completed training jobs found with prefix: {prefix}")
    return matched[0]["TrainingJobName"]


def create_model(sm, model_name, image_uri, model_data_url, execution_role_arn):
    """Create a SageMaker model, deleting any existing one first."""
    try:
        sm.describe_model(ModelName=model_name)
        logger.info(f"Model {model_name} exists, deleting...")
        sm.delete_model(ModelName=model_name)
    except sm.exceptions.ClientError as e:
        if "Could not find model" in str(e):
            logger.info("No existing model, creating fresh")
        else:
            raise

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": image_uri, "ModelDataUrl": model_data_url},
        ExecutionRoleArn=execution_role_arn,
    )
    logger.info(f"Created ephemeral model: {model_name}")


def delete_model_if_exists(sm, model_name):
    """Best-effort delete of a SageMaker Model (eval models are ephemeral)."""
    try:
        sm.delete_model(ModelName=model_name)
        logger.info(f"Deleted ephemeral model: {model_name}")
    except sm.exceptions.ClientError as e:
        if "Could not find model" in str(e):
            logger.info(f"Model already gone: {model_name}")
        else:
            logger.warning(f"Could not delete model {model_name}: {e}")


def cleanup_ephemeral_models(sm, name_prefix, keep=None):
    """Delete leftover eval Models matching ``name_prefix`` (except ``keep``)."""
    keep = keep or set()
    paginator = sm.get_paginator("list_models")
    deleted = 0
    for page in paginator.paginate(NameContains=name_prefix, MaxResults=100):
        for summary in page.get("Models", []):
            name = summary["ModelName"]
            if not name.startswith(name_prefix) or name in keep:
                continue
            delete_model_if_exists(sm, name)
            deleted += 1
    if deleted:
        logger.info(f"Cleaned up {deleted} ephemeral model(s) with prefix {name_prefix}")


def prepare_inference_data(s3, test_data_path, model_name):
    """Strip label column from test data and upload for batch transform."""
    parsed = urlparse(test_data_path.rstrip("/"))
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"

    # Use a run-specific inference prefix to avoid stale files
    inference_prefix = prefix.rstrip("/").rsplit("/", 1)[0] + f"/inference/{model_name}/"

    # Clean existing files in this prefix
    existing = s3.list_objects_v2(Bucket=bucket, Prefix=inference_prefix)
    for obj in existing.get("Contents", []):
        s3.delete_object(Bucket=bucket, Key=obj["Key"])

    # List test CSV files (only directly under test/)
    test_objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in test_objects.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".csv") and key.startswith(prefix) and "/" not in key[len(prefix):]:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
            # Strip first column (label) from each row
            features_only = "\n".join(
                ",".join(line.split(",")[1:]) for line in body.strip().split("\n")
            )
            dest_key = inference_prefix + key.split("/")[-1]
            s3.put_object(Bucket=bucket, Key=dest_key, Body=features_only, ContentType="text/csv")
            logger.info(f"Prepared inference data: s3://{bucket}/{dest_key}")

    return f"s3://{bucket}/{inference_prefix}"


def run_batch_transform(sm, transform_job_name, model_name, inference_data_path, output_path):
    """Run batch transform and wait for completion."""
    logger.info(f"Starting batch transform: {transform_job_name}")
    logger.info(f"  Input:  {inference_data_path}")
    logger.info(f"  Output: {output_path}")

    try:
        sm.create_transform_job(
            TransformJobName=transform_job_name,
            ModelName=model_name,
            TransformInput={
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": inference_data_path}},
                "ContentType": "text/csv",
                "SplitType": "Line",
            },
            TransformOutput={
                "S3OutputPath": output_path,
                "AssembleWith": "Line",
            },
            TransformResources={"InstanceType": "ml.m5.large", "InstanceCount": 1},
        )
    except sm.exceptions.ClientError as e:
        if "already exists" in str(e).lower():
            logger.info(f"Transform job {transform_job_name} already exists, using existing results")
        else:
            raise

    # Wait for completion
    logger.info("Waiting for transform job to complete...")
    waiter = sm.get_waiter("transform_job_completed_or_stopped")
    waiter.wait(TransformJobName=transform_job_name, WaiterConfig={"Delay": 15, "MaxAttempts": 60})

    info = sm.describe_transform_job(TransformJobName=transform_job_name)
    if info["TransformJobStatus"] != "Completed":
        raise RuntimeError(f"Transform failed: {info['TransformJobStatus']}")
    logger.info(f"Transform completed: {info['TransformOutput']['S3OutputPath']}")


def read_transform_scores(s3, eval_output_prefix):
    """Read raw score lines from Batch Transform ``*.csv.out`` files."""
    pred_parsed = urlparse(eval_output_prefix.rstrip("/"))
    pred_bucket = pred_parsed.netloc
    pred_prefix = pred_parsed.path.lstrip("/").rstrip("/") + "/"

    pred_objects = s3.list_objects_v2(Bucket=pred_bucket, Prefix=pred_prefix)
    pred_files = sorted(
        obj["Key"] for obj in pred_objects.get("Contents", [])
        if obj["Key"].endswith(".csv.out")
    )

    predictions = []
    for key in pred_files:
        body = s3.get_object(Bucket=pred_bucket, Key=key)["Body"].read().decode("utf-8")
        for line in body.strip().split("\n"):
            line = line.strip()
            if line:
                # First cell is the score (raw .out is a single value per line)
                predictions.append(float(line.split(",")[0]))
    return predictions, pred_files


def read_inference_features(s3, inference_data_path):
    """Read feature rows from the batch-transform input prefix (sorted files).

    Returns a list of feature vectors (list[float]) in the same row order the
    transform job scored them.
    """
    parsed = urlparse(inference_data_path.rstrip("/"))
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"

    objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    csv_files = sorted(
        obj["Key"]
        for obj in objects.get("Contents", [])
        if obj["Key"].endswith(".csv")
        and obj["Key"].startswith(prefix)
        and "/" not in obj["Key"][len(prefix) :]
    )

    features = []
    for key in csv_files:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        for line in body.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            features.append([float(x) for x in line.split(",")])
    return features


def write_enriched_predictions(
    table_uri,
    scores,
    model_name,
    model_version,
    inference_timestamp,
    feature_rows=None,
):
    """Append enriched scores to a Delta table (creates the table on first write).

    ``table_uri`` is a stable S3 prefix (e.g. .../evaluation/predictions_enriched)
    so successive evaluate / batch runs accumulate rows instead of overwriting.

    ``features`` is a JSON object string (named map) matching realtime
    ``inference.py``. Appends use ``schema_mode=merge`` so older 4-column
    tables evolve when the ``features`` column is introduced.

    Converts via PyArrow explicitly — newer ``deltalake`` rejects raw pandas
    DataFrames in the sklearn processing image (Arrow C stream mismatch).
    """
    try:
        import subprocess

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "deltalake",
                "pandas",
                "pyarrow",
            ],
            stdout=subprocess.DEVNULL,
        )
        import pandas as pd
        import pyarrow as pa
        from deltalake import write_deltalake
    except Exception as e:
        raise RuntimeError(
            f"Could not install/import deltalake for enriched predictions: {e}"
        ) from e

    feature_rows = feature_rows or [[] for _ in scores]
    if len(feature_rows) != len(scores):
        raise ValueError(
            f"Feature/score mismatch: {len(feature_rows)} feature rows vs {len(scores)} scores"
        )

    # Named feature maps (same shape as realtime inference.py JSON).
    try:
        from feature_schema import features_to_named
    except ImportError:
        # Processing image runs with src/ on the module path; keep a safe fallback.
        def features_to_named(values):  # type: ignore
            return {f"f{i}": float(v) for i, v in enumerate(values or [])}

    named_features = [features_to_named(f) for f in feature_rows]

    df = pd.DataFrame(
        {
            "score": [float(s) for s in scores],
            "features": [json.dumps(f) for f in named_features],
            "model_name": [str(model_name)] * len(scores),
            "model_version": [str(model_version)] * len(scores),
            "inference_timestamp": [str(inference_timestamp)] * len(scores),
        }
    )
    uri = table_uri.rstrip("/")
    table = pa.Table.from_pandas(df, preserve_index=False)
    try:
        write_deltalake(uri, table, mode="append", schema_mode="merge")
    except TypeError:
        # Older deltalake without schema_mode — fall back to overwrite on mismatch.
        try:
            write_deltalake(uri, table, mode="append")
        except Exception as append_exc:
            logger.warning(
                f"Delta append failed ({append_exc}); overwriting table with new schema"
            )
            write_deltalake(uri, table, mode="overwrite")
    except Exception as append_exc:
        # e.g. SchemaMismatchError when merge is unsupported for the change.
        logger.warning(
            f"Delta append/merge failed ({append_exc}); overwriting table with new schema"
        )
        write_deltalake(uri, table, mode="overwrite")
    logger.info(f"Appended {len(scores)} enriched predictions to Delta table {uri}")
    return uri


def compute_metrics(s3, test_data_path, scores, pred_files=None):
    """Read labels from test data and compute metrics against ``scores``."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    # Read test labels (first column of test CSV)
    parsed = urlparse(test_data_path.rstrip("/"))
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"

    test_objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    test_csv_files = sorted(
        obj["Key"] for obj in test_objects.get("Contents", [])
        if obj["Key"].endswith(".csv") and obj["Key"].startswith(prefix)
        and "/" not in obj["Key"][len(prefix):]
    )

    labels = []
    for key in test_csv_files:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        for line in body.strip().split("\n"):
            val = line.split(",")[0].strip()
            try:
                labels.append(int(float(val)))
            except ValueError:
                continue

    y_true = np.array(labels)
    y_prob = np.array(scores)

    logger.info(f"Labels count: {len(y_true)}, unique values: {np.unique(y_true)}")
    logger.info(f"Predictions count: {len(y_prob)}, range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Mismatch: {len(y_true)} labels vs {len(y_prob)} predictions. "
            f"Test files: {test_csv_files}, Pred files: {pred_files}"
        )

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "test_samples": int(len(y_true)),
    }

    logger.info("=== Evaluation Metrics ===")
    for k, v in metrics.items():
        logger.info(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k:15s}: {v}")

    return metrics


def log_to_mlflow(tracking_server_arn, experiment_name, training_job_name, model_name, metrics):
    """Log evaluation metrics to MLflow."""
    if not tracking_server_arn or not experiment_name:
        logger.info("MLflow not configured, skipping")
        return

    try:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "mlflow>=3.1,<4", "sagemaker-mlflow==0.4.0"],
            stdout=subprocess.DEVNULL,
        )
        import mlflow
    except Exception as e:
        logger.warning(f"Could not install/import mlflow: {e}. Skipping MLflow logging.")
        return

    os.environ["MLFLOW_TRACKING_ARN"] = tracking_server_arn
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)

    # Find parent run from training
    runs = mlflow.search_runs(
        filter_string=f"tags.pipeline_run = '{training_job_name}' and tags.pipeline_step = 'train_model'",
        max_results=1,
    )
    parent_run_id = runs.iloc[0]["run_id"] if not runs.empty else None

    with mlflow.start_run(run_name=f"evaluate-{model_name}", parent_run_id=parent_run_id):
        mlflow.set_tag("pipeline_step", "evaluate_model")
        mlflow.set_tag("pipeline_run", training_job_name)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_samples", metrics["test_samples"])
        logger.info("Logged evaluation metrics to MLflow")


def main():
    args = parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    # Step 1: Find latest training job
    training_job_name = find_latest_training_job(sm, args.training_job_prefix)
    job_info = sm.describe_training_job(TrainingJobName=training_job_name)
    model_data_url = job_info["ModelArtifacts"]["S3ModelArtifacts"]

    suffix = training_job_name[len(args.training_job_prefix):]
    model_name = f"{args.model_prefix}{suffix}"

    logger.info(f"Training job: {training_job_name}")
    logger.info(f"Model name:   {model_name}")
    logger.info(f"Model data:   {model_data_url}")

    # Step 2: Create ephemeral SageMaker model for Batch Transform only.
    # Registry persistence happens later in register_model (stack-prediction-models).
    create_model(sm, model_name, args.image_uri, model_data_url, args.execution_role_arn)
    try:
        # Step 3: Prepare inference data and run batch transform
        inference_data_path = prepare_inference_data(s3, args.test_data_path, model_name)
        eval_output_prefix = args.output_path.rstrip("/") + f"/{model_name}/"
        transform_job_name = f"stack-eval{suffix}"

        run_batch_transform(sm, transform_job_name, model_name, inference_data_path, eval_output_prefix)

        # Step 4: Enrich raw transform scores with features + model identity + time
        inference_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        # Pre-registry: training job name is the unique candidate version identifier.
        model_version = training_job_name
        scores, pred_files = read_transform_scores(s3, eval_output_prefix)
        feature_rows = read_inference_features(s3, inference_data_path)
        # Stable Delta path (not per-model) so runs append into one table.
        enriched_table_uri = args.output_path.rstrip("/") + "/predictions_enriched"
        enriched_uri = write_enriched_predictions(
            enriched_table_uri,
            scores,
            model_name=model_name,
            model_version=model_version,
            inference_timestamp=inference_timestamp,
            feature_rows=feature_rows,
        )

        # Step 5: Compute metrics
        metrics = compute_metrics(s3, args.test_data_path, scores, pred_files=pred_files)

        # Step 6: Save evaluation report to S3
        evaluation_report = {
            "training_job": training_job_name,
            "model_name": model_name,
            "model_version": model_version,
            "inference_timestamp": inference_timestamp,
            "enriched_predictions_uri": enriched_uri,
            "model_data_url": model_data_url,
            "metrics": metrics,
        }

        pred_parsed = urlparse(eval_output_prefix.rstrip("/"))
        report_key = pred_parsed.path.lstrip("/").rstrip("/") + "/evaluation_report.json"
        s3.put_object(
            Bucket=pred_parsed.netloc,
            Key=report_key,
            Body=json.dumps(evaluation_report, indent=2, default=str),
            ContentType="application/json",
        )
        logger.info(f"Evaluation report saved to s3://{pred_parsed.netloc}/{report_key}")

        # Step 7: Log to MLflow
        log_to_mlflow(
            args.tracking_server_arn, args.experiment_name,
            training_job_name, model_name, metrics,
        )

        # Summary
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Training Job:  {training_job_name}")
        logger.info(f"  Model:         {model_name}")
        logger.info(f"  Version:       {model_version}")
        logger.info(f"  Inferred at:   {inference_timestamp}")
        logger.info(f"  Enriched Delta: {enriched_uri}")
        logger.info(f"  AUC:           {metrics['auc']:.4f}")
        logger.info(f"  Accuracy:      {metrics['accuracy']:.4f}")
        logger.info(f"  F1:            {metrics['f1']:.4f}")
        logger.info(f"  Test Samples:  {metrics['test_samples']}")
        logger.info("=" * 60)
    finally:
        # Transform no longer needs the Model; do not leave it in "My models".
        delete_model_if_exists(sm, model_name)
        cleanup_ephemeral_models(sm, args.model_prefix, keep={model_name})


if __name__ == "__main__":
    main()
