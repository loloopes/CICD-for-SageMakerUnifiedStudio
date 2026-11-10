"""Deploy Model - SageMaker Processing Script.

Finds the latest approved model version in the Model Registry, creates or
updates a SageMaker real-time endpoint with the approved model.

This script is designed to run as a SageMaker Processing job via the
SageMakerProcessingOperator in the deploy pipeline workflow.

Arguments (passed via --arg=value):
    --model-package-group: Model package group name in the registry
    --endpoint-name: Target SageMaker endpoint name
    --image-uri: XGBoost container image URI for inference
    --execution-role-arn: SageMaker execution role ARN
    --region: AWS region
    --instance-type: Endpoint instance type (default: ml.m5.large)
    --instance-count: Number of endpoint instances (default: 1)
    --tracking-server-arn: MLflow tracking server ARN (optional)
    --experiment-name: MLflow experiment name (optional)
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "/opt/ml/processing/output"
# Shipped next to this script via the deploy_pipeline ProcessingInput.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_SCRIPT = os.path.join(_SRC_DIR, "inference.py")
FEATURE_SCHEMA_SCRIPT = os.path.join(_SRC_DIR, "feature_schema.py")

# Sentinel value used by the promote/deploy GitHub workflows when an optional
# CLI flag is unset. Translates to "no override" for ARN/URL pinning.
SENTINEL_NONE = "none"


def _clean(value: str) -> str:
    """Return value unless it's empty/whitespace or the 'none' sentinel."""
    if not value:
        return ""
    stripped = value.strip()
    if not stripped or stripped.lower() == SENTINEL_NONE:
        return ""
    return stripped


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Deploy approved model to endpoint")
    parser.add_argument("--model-package-group", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--execution-role-arn", required=True)
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--tracking-server-arn", default="")
    parser.add_argument("--experiment-name", default="")
    # When the promote workflow pins a specific approved model version, it
    # passes the ARN here. Sentinel "none" (or empty) means fall back to
    # "latest approved in the registry".
    parser.add_argument(
        "--model-package-arn",
        default="",
        help="Pin a specific model package ARN. Use 'none' or omit to pick latest Approved.",
    )
    # When the promote workflow stages the model artifact into a stage-local
    # bucket, it passes the staged S3 URL here so the new SageMaker Model is
    # built off the staged copy instead of the source bucket. Sentinel "none"
    # (or empty) means use whatever ModelDataUrl is on the package.
    parser.add_argument(
        "--model-data-url-override",
        default="",
        help="Override ModelDataUrl. Use 'none' or omit to use the package's URL.",
    )
    parser.add_argument(
        "--git-commit",
        default="",
        help="Workflow git SHA, recorded on the deployment report for traceability.",
    )
    return parser.parse_args()


# =============================================================================
# Model Registry
# =============================================================================


def find_approved_model_package(sm, model_package_group):
    """Find the latest approved model version in the registry.

    Returns:
        dict with model package details, or None if no approved model exists.
    """
    logger.info(f"Querying model registry: {model_package_group}")

    try:
        response = sm.list_model_packages(
            ModelPackageGroupName=model_package_group,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
    except sm.exceptions.ClientError as e:
        if "does not exist" in str(e).lower() or "ValidationException" in str(e):
            logger.warning(f"Model package group '{model_package_group}' does not exist.")
            return None
        raise

    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        logger.warning(f"No approved model versions in group: {model_package_group}")
        return None

    model_package_arn = packages[0]["ModelPackageArn"]
    logger.info(f"Found approved model: {model_package_arn}")

    details = sm.describe_model_package(ModelPackageName=model_package_arn)
    logger.info(f"  Version:     {details.get('ModelPackageVersion', 'N/A')}")
    logger.info(f"  Created:     {details.get('CreationTime', 'N/A')}")
    logger.info(f"  Description: {details.get('ModelPackageDescription', 'N/A')}")
    return details


def describe_pinned_model_package(sm, model_package_arn):
    """Look up a specific (pinned) model package by ARN.

    Used when the promote workflow has pinned the version that flows through
    every stage. We still verify it's Approved so we don't accidentally
    deploy something that was rejected after the cascade started.
    """
    logger.info(f"Looking up pinned model package: {model_package_arn}")
    details = sm.describe_model_package(ModelPackageName=model_package_arn)
    status = details.get("ModelApprovalStatus", "")
    logger.info(f"  Version : {details.get('ModelPackageVersion', 'N/A')}")
    logger.info(f"  Created : {details.get('CreationTime', 'N/A')}")
    logger.info(f"  Approval: {status}")
    if status != "Approved":
        raise ValueError(
            f"Pinned model package {model_package_arn} has status '{status}', "
            "expected 'Approved'. Refusing to deploy."
        )
    return details


def extract_model_data_url(model_package):
    """Extract the model artifact S3 URL from a model package."""
    model_package_arn = model_package["ModelPackageArn"]

    containers = model_package.get("InferenceSpecification", {}).get("Containers", [])
    if containers:
        return containers[0].get("ModelDataUrl")

    model_data_url = model_package.get("ModelDataUrl")
    if model_data_url:
        return model_data_url

    raise ValueError(f"No model data URL found in package: {model_package_arn}")


# =============================================================================
# SageMaker Model & Endpoint
# =============================================================================


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def package_realtime_serving_artifact(
    s3,
    source_model_data_url: str,
    dest_model_data_url: str,
    model_name: str,
    model_version: str,
    inference_script: str = INFERENCE_SCRIPT,
    feature_schema_script: str = FEATURE_SCHEMA_SCRIPT,
) -> str:
    """Rebuild model.tar.gz with ``code/inference.py`` + serving metadata.

    Realtime InvokeEndpoint then returns JSON rows with score, named features,
    model_name, model_version, and inference_timestamp (see ``inference.py``).
    """
    if not os.path.isfile(inference_script):
        raise FileNotFoundError(f"Realtime inference script not found: {inference_script}")
    if not os.path.isfile(feature_schema_script):
        raise FileNotFoundError(f"Feature schema script not found: {feature_schema_script}")

    src_bucket, src_key = _parse_s3_uri(source_model_data_url)
    dest_bucket, dest_key = _parse_s3_uri(dest_model_data_url)

    # Import locally so the processing image doesn't need a package install.
    import importlib.util

    spec = importlib.util.spec_from_file_location("feature_schema", feature_schema_script)
    feature_schema = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(feature_schema)
    feature_columns = list(feature_schema.FEATURE_COLUMNS)

    with tempfile.TemporaryDirectory() as tmp:
        local_tar = os.path.join(tmp, "source-model.tar.gz")
        extract_dir = os.path.join(tmp, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        logger.info(f"Downloading model artifact: {source_model_data_url}")
        s3.download_file(src_bucket, src_key, local_tar)

        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(path=extract_dir)

        code_dir = os.path.join(extract_dir, "code")
        os.makedirs(code_dir, exist_ok=True)
        shutil.copy2(inference_script, os.path.join(code_dir, "inference.py"))
        shutil.copy2(feature_schema_script, os.path.join(code_dir, "feature_schema.py"))
        meta = {
            "model_name": model_name,
            "model_version": str(model_version),
            "feature_columns": feature_columns,
        }
        with open(os.path.join(code_dir, "serving_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        model_file = os.path.join(extract_dir, "xgboost-model")
        if not os.path.isfile(model_file):
            # Some packages nest the booster; keep whatever was extracted but warn.
            logger.warning(
                "xgboost-model not found at artifact root; "
                f"contents={sorted(os.listdir(extract_dir))}"
            )

        out_tar = os.path.join(tmp, "serving-model.tar.gz")
        with tarfile.open(out_tar, "w:gz") as tar:
            for root, _dirs, files in os.walk(extract_dir):
                for name in files:
                    full = os.path.join(root, name)
                    arcname = os.path.relpath(full, extract_dir)
                    tar.add(full, arcname=arcname)

        logger.info(f"Uploading realtime serving artifact: {dest_model_data_url}")
        s3.upload_file(out_tar, dest_bucket, dest_key)

    return dest_model_data_url


def create_sagemaker_model(
    sm,
    model_name,
    model_package_arn,
    image_uri,
    model_data_url,
    execution_role_arn,
    prefer_data_url=False,
    model_version: str = "unknown",
    use_realtime_inference_script: bool = False,
):
    """Create a SageMaker Model.

    By default tries the model package first, then falls back to image + data URL.
    When ``prefer_data_url`` is True (promote staged the artifact into the target
    project's bucket), skip the package path — ModelPackageName always points at
    the *source* ModelDataUrl, which the target stage role often cannot read.

    When ``use_realtime_inference_script`` is True, always create from image +
    ModelDataUrl and set container env so ``code/inference.py`` enriches responses.
    """
    logger.info(f"Creating SageMaker model: {model_name}")

    if use_realtime_inference_script:
        prefer_data_url = True

    if not prefer_data_url:
        try:
            sm.create_model(
                ModelName=model_name,
                Containers=[{"ModelPackageName": model_package_arn}],
                ExecutionRoleArn=execution_role_arn,
            )
            logger.info(f"  Created from model package")
            return
        except sm.exceptions.ClientError as e:
            logger.warning(f"  Model package approach failed: {e}")
            logger.info(f"  Falling back to image + model data URL")

    container = {
        "Image": image_uri,
        "ModelDataUrl": model_data_url,
    }
    if use_realtime_inference_script:
        container["Environment"] = {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            "MODEL_NAME": model_name,
            "MODEL_VERSION": str(model_version),
        }
        logger.info("  Realtime enrichment enabled (inference.py)")

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer=container,
        ExecutionRoleArn=execution_role_arn,
    )
    logger.info(f"  Created from image + data URL")


def deploy_endpoint(sm, endpoint_name, model_name, instance_type, instance_count):
    """Create or update a SageMaker real-time endpoint."""
    endpoint_config_name = f"{endpoint_name}-config-{int(time.time())}"

    # Create endpoint config
    logger.info(f"Creating endpoint config: {endpoint_config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": instance_count,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    # Check if endpoint exists (Failed endpoints must be deleted before recreate)
    endpoint_exists = False
    try:
        existing = sm.describe_endpoint(EndpointName=endpoint_name)
        status = existing.get("EndpointStatus", "")
        if status == "Failed":
            logger.warning(
                f"Endpoint {endpoint_name} is Failed "
                f"({existing.get('FailureReason', 'unknown')}); deleting before recreate"
            )
            sm.delete_endpoint(EndpointName=endpoint_name)
            waiter = sm.get_waiter("endpoint_deleted")
            waiter.wait(EndpointName=endpoint_name, WaiterConfig={"Delay": 15, "MaxAttempts": 40})
        else:
            endpoint_exists = True
    except sm.exceptions.ClientError as e:
        if "Could not find endpoint" not in str(e):
            raise

    if endpoint_exists:
        logger.info(f"Updating existing endpoint: {endpoint_name}")
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
    else:
        logger.info(f"Creating new endpoint: {endpoint_name}")
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)

    # Wait for endpoint to be InService
    logger.info("Waiting for endpoint to be InService (polling every 30s)...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name, WaiterConfig={"Delay": 30, "MaxAttempts": 40})

    endpoint_info = sm.describe_endpoint(EndpointName=endpoint_name)
    logger.info(f"Endpoint ready: {endpoint_info['EndpointStatus']}")
    return endpoint_info


# =============================================================================
# MLflow Logging
# =============================================================================


def log_to_mlflow(tracking_server_arn, experiment_name, model_package_arn, endpoint_name, model_name):
    """Log deployment event to MLflow tracking server."""
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
        logger.warning(f"Could not install/import mlflow: {e}")
        return

    # MLflow logging is best-effort deployment metadata — the endpoint is
    # already live by this point, so a tracking-server problem (e.g. an
    # unreachable/misconfigured server ARN) must not fail the deploy job.
    try:
        os.environ["MLFLOW_TRACKING_ARN"] = tracking_server_arn
        mlflow.set_tracking_uri(tracking_server_arn)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"deploy-{endpoint_name}"):
            mlflow.set_tag("pipeline_step", "deploy_model")
            mlflow.set_tag("model_package_arn", model_package_arn)
            mlflow.set_tag("endpoint_name", endpoint_name)
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("deployment_time", datetime.now(timezone.utc).isoformat())
        logger.info("Logged deployment to MLflow")
    except Exception as e:
        logger.warning(
            f"MLflow logging failed (non-fatal, endpoint already deployed): {e}"
        )


# =============================================================================
# Report
# =============================================================================


def save_report(report, filename="deployment_report.json"):
    """Save a JSON report to the processing output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved: {path}")


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("DEPLOY MODEL - STARTING")
    logger.info("=" * 60)
    logger.info(f"  Model Package Group : {args.model_package_group}")
    logger.info(f"  Endpoint Name       : {args.endpoint_name}")
    logger.info(f"  Image URI           : {args.image_uri}")
    logger.info(f"  Execution Role      : {args.execution_role_arn}")
    logger.info(f"  Region              : {args.region}")
    logger.info(f"  Instance Type       : {args.instance_type}")
    logger.info(f"  Instance Count      : {args.instance_count}")
    logger.info(f"  MLflow Server       : {args.tracking_server_arn or 'not configured'}")
    logger.info(f"  Experiment          : {args.experiment_name or 'not configured'}")
    logger.info(f"  Pinned Pkg ARN      : {_clean(args.model_package_arn) or '<latest approved>'}")
    logger.info(f"  Data URL Override   : {_clean(args.model_data_url_override) or '<from package>'}")
    logger.info(f"  Git Commit          : {_clean(args.git_commit) or '<unset>'}")
    logger.info("=" * 60)

    sm = boto3.client("sagemaker", region_name=args.region)

    # Step 1: Find or look up the model
    pinned_arn = _clean(args.model_package_arn)
    if pinned_arn:
        logger.info("[1/5] Using pinned model package from promote workflow")
        model_package = describe_pinned_model_package(sm, pinned_arn)
    else:
        logger.info("[1/5] Searching for latest Approved model in registry")
        model_package = find_approved_model_package(sm, args.model_package_group)

    if model_package is None:
        logger.info("=" * 60)
        logger.info("NO APPROVED MODEL - SKIPPING DEPLOYMENT")
        logger.info("=" * 60)
        logger.info(f"  Model Group: {args.model_package_group}")
        logger.info("  Action:      No deployment performed")
        logger.info("  Next Step:   Approve a model version in the registry")
        logger.info("=" * 60)
        save_report({
            "status": "skipped",
            "reason": "No approved model versions found",
            "model_package_group": args.model_package_group,
            "endpoint_name": args.endpoint_name,
            "git_commit": _clean(args.git_commit),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return

    model_package_arn = model_package["ModelPackageArn"]
    model_version = str(model_package.get("ModelPackageVersion", "unknown"))
    model_data_url = extract_model_data_url(model_package)
    data_url_override = _clean(args.model_data_url_override)
    if data_url_override:
        logger.info(
            f"  Overriding ModelDataUrl with staged copy: "
            f"{model_data_url} -> {data_url_override}"
        )
        model_data_url = data_url_override
    logger.info(f"  Model data: {model_data_url}")

    # Step 2: Create SageMaker Model (with realtime enrichment script)
    logger.info("[2/5] Creating SageMaker model")
    model_name = f"{args.endpoint_name}-{int(time.time())}"
    s3 = boto3.client("s3", region_name=args.region)
    # Serve from a project-local artifact that embeds code/inference.py so
    # InvokeEndpoint returns score + model identity + timestamp.
    src_bucket = urlparse(model_data_url).netloc
    src_key = urlparse(model_data_url).path.lstrip("/")
    shared_marker = "/shared/"
    if shared_marker in src_key:
        shared_root = src_key.split(shared_marker, 1)[0] + shared_marker
        serving_uri = (
            f"s3://{src_bucket}/{shared_root}"
            f"stack/mlops_pipeline/deployments/realtime-serving/{model_name}/model.tar.gz"
        )
    else:
        serving_uri = (
            f"s3://{src_bucket}/stack/mlops_pipeline/deployments/"
            f"realtime-serving/{model_name}/model.tar.gz"
        )

    logger.info(f"  Packaging realtime serving artifact → {serving_uri}")
    serving_model_data_url = package_realtime_serving_artifact(
        s3,
        source_model_data_url=model_data_url,
        dest_model_data_url=serving_uri,
        model_name=model_name,
        model_version=model_version,
    )

    # Always use image + ModelDataUrl so the custom inference.py is active.
    # Staged overrides already required this path for cross-bucket reads.
    create_sagemaker_model(
        sm,
        model_name,
        model_package_arn,
        args.image_uri,
        serving_model_data_url,
        args.execution_role_arn,
        prefer_data_url=True,
        model_version=model_version,
        use_realtime_inference_script=True,
    )

    # Step 3: Deploy endpoint
    logger.info("[3/5] Deploying endpoint")
    endpoint_info = deploy_endpoint(sm, args.endpoint_name, model_name, args.instance_type, args.instance_count)

    # Step 4: Save report
    logger.info("[4/5] Saving deployment report")
    save_report({
        "status": "deployed",
        "model_package_arn": model_package_arn,
        "model_package_group": args.model_package_group,
        "model_name": model_name,
        "model_version": model_version,
        "model_data_url": model_data_url,
        "serving_model_data_url": serving_model_data_url,
        "model_data_url_overridden": bool(data_url_override),
        "realtime_enrichment": True,
        "endpoint_name": args.endpoint_name,
        "endpoint_arn": endpoint_info.get("EndpointArn"),
        "endpoint_status": endpoint_info.get("EndpointStatus"),
        "instance_type": args.instance_type,
        "instance_count": args.instance_count,
        "git_commit": _clean(args.git_commit),
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    })

    # Step 5: Log to MLflow
    logger.info("[5/5] Logging to MLflow")
    log_to_mlflow(args.tracking_server_arn, args.experiment_name, model_package_arn, args.endpoint_name, model_name)

    # Summary
    logger.info("=" * 60)
    logger.info("DEPLOYMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Model Package: {model_package_arn}")
    logger.info(f"  Model Name:    {model_name}")
    logger.info(f"  Endpoint:      {args.endpoint_name}")
    logger.info(f"  Status:        {endpoint_info.get('EndpointStatus')}")
    logger.info(f"  Instance:      {args.instance_type} x {args.instance_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
