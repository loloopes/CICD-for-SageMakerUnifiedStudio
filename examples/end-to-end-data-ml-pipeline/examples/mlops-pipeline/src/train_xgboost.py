"""
Bank Mktg XGBoost Training Script - SageMaker Script Mode
===============================================================
Custom training script for SageMaker Training jobs. Reads CSV data
from SageMaker input channels, trains an XGBoost classifier, evaluates
on validation set, logs everything to MLflow, and saves the model artifact.

SageMaker injects:
  - SM_CHANNEL_TRAIN      : /opt/ml/input/data/train/
  - SM_CHANNEL_VALIDATION : /opt/ml/input/data/validation/
  - SM_MODEL_DIR          : /opt/ml/model/
  - SM_OUTPUT_DATA_DIR    : /opt/ml/output/data/
  - Hyperparameters as command-line args or env vars

MLflow tracking:
  - MLFLOW_TRACKING_ARN   : SageMaker MLflow tracking server ARN
  - experiment_name        : MLflow experiment name (hyperparameter)
"""

import argparse
import subprocess
import sys

# Install mlflow if not available (XGBoost container doesn't include it)
try:
    import mlflow  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mlflow", "-q"])
import json
import logging
import os
import glob

import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # XGBoost hyperparameters
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--eval_metric", type=str, default="auc")
    parser.add_argument("--num_round", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--min_child_weight", type=int, default=3)
    parser.add_argument("--scale_pos_weight", type=float, default=3.0)
    parser.add_argument("--early_stopping_rounds", type=int, default=20)

    # MLflow tracking
    parser.add_argument("--tracking_server_arn", type=str, default=os.environ.get("MLFLOW_TRACKING_ARN", ""))
    parser.add_argument("--experiment_name", type=str, default="stack-experiments")

    # SageMaker environment
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))

    return parser.parse_args()


def load_csv_data(directory):
    """Load all CSV files from a directory into a numpy array (no headers)."""
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    arrays = []
    for f in files:
        data = np.loadtxt(f, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        arrays.append(data)
    combined = np.vstack(arrays)
    logger.info(f"Loaded {combined.shape[0]} rows from {len(files)} files in {directory}")
    return combined


def setup_mlflow(tracking_server_arn, experiment_name):
    """Configure MLflow tracking with SageMaker tracking server.

    The sagemaker-mlflow plugin handles authentication and URL resolution
    when the tracking URI is set to the tracking server ARN directly.
    """
    if not tracking_server_arn:
        logger.warning("No MLflow tracking server ARN provided, skipping MLflow setup")
        return False

    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed in this environment, skipping MLflow setup")
        return False

    # Set the ARN as both the environment variable and tracking URI.
    # The sagemaker-mlflow plugin intercepts requests when the tracking URI
    # is an ARN and handles URL resolution + auth header injection internally.
    os.environ["MLFLOW_TRACKING_ARN"] = tracking_server_arn
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow configured: experiment={experiment_name}, tracking_uri={tracking_server_arn}")
    return True


def train(args):
    # Load data — first column is the label (XGBoost CSV convention)
    train_data = load_csv_data(args.train)
    val_data = load_csv_data(args.validation)

    y_train, X_train = train_data[:, 0], train_data[:, 1:]
    y_val, X_val = val_data[:, 0], val_data[:, 1:]

    logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    logger.info(f"Positive rate — train: {y_train.mean():.3f}, val: {y_val.mean():.3f}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": args.objective,
        "eval_metric": args.eval_metric,
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "scale_pos_weight": args.scale_pos_weight,
    }

    logger.info(f"Training params: {json.dumps(params, indent=2)}")

    # Setup MLflow tracking
    mlflow_enabled = setup_mlflow(args.tracking_server_arn, args.experiment_name)

    if mlflow_enabled:
        import mlflow
        import mlflow.xgboost

    # Start MLflow run (or use a no-op context if disabled)
    mlflow_ctx = mlflow.start_run(run_name=os.environ.get("TRAINING_JOB_NAME", "xgb-training")) if mlflow_enabled else None

    try:
        if mlflow_ctx:
            mlflow_ctx.__enter__()

            # Log hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("num_round", args.num_round)
            mlflow.log_param("early_stopping_rounds", args.early_stopping_rounds)
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("val_samples", X_val.shape[0])
            mlflow.log_param("num_features", X_train.shape[1])
            mlflow.set_tag("pipeline_step", "train_model")
            mlflow.set_tag("framework", "xgboost")
            mlflow.set_tag("script_mode", "true")

        # Train
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=args.num_round,
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=25,
        )

        # Evaluate
        val_preds = model.predict(dval)
        val_labels = (val_preds >= 0.5).astype(int)

        metrics = {
            "auc": float(roc_auc_score(y_val, val_preds)),
            "accuracy": float(accuracy_score(y_val, val_labels)),
            "f1": float(f1_score(y_val, val_labels)),
            "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else args.num_round,
        }
        logger.info(f"Validation metrics: {json.dumps(metrics, indent=2)}")

        # Log metrics to MLflow
        if mlflow_ctx:
            mlflow.log_metrics({
                "val_auc": metrics["auc"],
                "val_accuracy": metrics["accuracy"],
                "val_f1": metrics["f1"],
                "best_iteration": metrics["best_iteration"],
            })

            # Log the XGBoost model artifact
            # Common failure: MLflow tracking server role lacks KMS permissions
            # when the artifact bucket uses SSE-KMS encryption (SageMaker default).
            # Required: kms:GenerateDataKey, kms:Decrypt on the bucket's KMS key.
            try:
                mlflow.xgboost.log_model(model, artifact_path="model")
                logger.info(f"Model logged to MLflow run: {mlflow.active_run().info.run_id}")
            except Exception as e:
                logger.warning(f"Failed to log model artifact to MLflow: {e}")
                if "AccessDenied" in str(e) or "KMS" in str(e):
                    logger.warning(
                        "This is likely a KMS permission issue. Ensure the MLflow tracking "
                        "server role has kms:GenerateDataKey and kms:Decrypt permissions "
                        "for the artifact bucket's KMS key."
                    )
                elif "403" in str(e) or "Forbidden" in str(e):
                    logger.warning(
                        "S3 access denied. Ensure the MLflow tracking server role has "
                        "s3:PutObject and s3:AbortMultipartUpload on the artifact bucket."
                    )
                logger.warning("Model is still saved to SageMaker model directory.")

        # Save metrics locally
        os.makedirs(args.output_data_dir, exist_ok=True)
        with open(os.path.join(args.output_data_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save feature importance
        importance = model.get_score(importance_type="gain")
        with open(os.path.join(args.output_data_dir, "feature_importance.json"), "w") as f:
            json.dump(importance, f, indent=2)

        # Log artifacts to MLflow
        if mlflow_ctx:
            try:
                mlflow.log_artifact(os.path.join(args.output_data_dir, "metrics.json"))
                mlflow.log_artifact(os.path.join(args.output_data_dir, "feature_importance.json"))
            except Exception as e:
                logger.warning(f"Failed to log artifacts to MLflow: {e}")

        # Save model to SageMaker model dir
        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(args.model_dir, "xgboost-model")
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

        return model

    finally:
        if mlflow_ctx:
            mlflow_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    args = parse_args()
    train(args)
