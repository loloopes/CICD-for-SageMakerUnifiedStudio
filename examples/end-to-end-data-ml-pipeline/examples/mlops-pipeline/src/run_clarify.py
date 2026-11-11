#!/usr/bin/env python3
"""Run SageMaker Clarify (bias + SHAP) for the latest stack-xgb model.

Creates an ephemeral SageMaker Model from the latest completed training job,
builds a headered sample of the test CSV, launches a Clarify processing job,
writes reports under ``…/models/evaluation/clarify/``, then deletes the Model.

Demo facet: continuous ``age`` with threshold 50 (older vs younger). This is
illustrative for the bank-marketing schema — replace with real protected
attributes for production fairness work.

Invoked as a SageMaker Processing job from ``training_pipeline``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from urllib.parse import urlparse

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_clarify")

# Account-mapped Clarify processing images (SageMaker Clarify 1.0).
CLARIFY_IMAGE = {
    "us-east-1": "205492490401.dkr.ecr.us-east-1.amazonaws.com/sagemaker-clarify-processing:1.0",
    "us-east-2": "211330385726.dkr.ecr.us-east-2.amazonaws.com/sagemaker-clarify-processing:1.0",
    "us-west-2": "263288472518.dkr.ecr.us-west-2.amazonaws.com/sagemaker-clarify-processing:1.0",
    "eu-west-1": "430846316453.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-clarify-processing:1.0",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-job-prefix", default="stack-xgb")
    p.add_argument("--model-prefix", default="stack-clarify")
    p.add_argument("--image-uri", required=True, help="XGBoost inference image for ephemeral Model")
    p.add_argument("--execution-role-arn", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--test-data-path", required=True, help="S3 prefix with label+feature CSVs")
    p.add_argument("--output-path", required=True, help="S3 prefix …/models/evaluation/")
    p.add_argument("--max-rows", type=int, default=500, help="Cap rows sent to Clarify (cost control)")
    p.add_argument(
        "--facet-column",
        default="age",
        help="Feature column used as Clarify bias facet (default: age)",
    )
    p.add_argument(
        "--facet-threshold",
        type=float,
        default=50.0,
        help="Threshold for continuous facet (default: 50 for age)",
    )
    p.add_argument("--probability-threshold", type=float, default=0.5)
    return p.parse_args()


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri.rstrip("/") + "/")
    return parsed.netloc, parsed.path.lstrip("/")


def find_latest_training_job(sm, prefix: str) -> tuple[str, str]:
    resp = sm.list_training_jobs(SortBy="CreationTime", SortOrder="Descending", MaxResults=25)
    matched = [
        j
        for j in resp["TrainingJobSummaries"]
        if j["TrainingJobName"].startswith(prefix) and j["TrainingJobStatus"] == "Completed"
    ]
    if not matched:
        raise SystemExit(f"No completed training jobs with prefix {prefix!r}")
    name = matched[0]["TrainingJobName"]
    info = sm.describe_training_job(TrainingJobName=name)
    return name, info["ModelArtifacts"]["S3ModelArtifacts"]


def list_csv_keys(s3, bucket: str, prefix: str) -> list[str]:
    prefix = prefix.rstrip("/") + "/"
    keys, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []) or []:
            key = obj["Key"]
            rel = key[len(prefix) :]
            if key.endswith(".csv") and rel and "/" not in rel:
                keys.append(key)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def delete_model(sm, name: str) -> None:
    try:
        sm.delete_model(ModelName=name)
        logger.info("Deleted ephemeral model %s", name)
    except sm.exceptions.ClientError as e:
        if "Could not find model" not in str(e):
            logger.warning("Could not delete model %s: %s", name, e)


def build_headered_dataset(s3, test_uri: str, max_rows: int) -> tuple[list[str], list[list[str]]]:
    """Return (headers, rows) with label column ``y`` then feature_schema columns."""
    from feature_schema import FEATURE_COLUMNS

    headers = ["y"] + list(FEATURE_COLUMNS)
    bucket, prefix = _parse_s3(test_uri)
    rows: list[list[str]] = []
    for key in list_csv_keys(s3, bucket, prefix):
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            # Pad/truncate to headers length
            if len(parts) < len(headers):
                parts = parts + ["0"] * (len(headers) - len(parts))
            rows.append(parts[: len(headers)])
            if max_rows and len(rows) >= max_rows:
                return headers, rows
    return headers, rows


def mean_baseline(headers: list[str], rows: list[list[str]]) -> list[float]:
    """Feature-only mean vector for SHAP baseline (exclude label col 0)."""
    if not rows:
        return [0.0] * (len(headers) - 1)
    n_feat = len(headers) - 1
    acc = [0.0] * n_feat
    for row in rows:
        for i in range(n_feat):
            acc[i] += float(row[i + 1])
    return [v / len(rows) for v in acc]


def main() -> int:
    args = parse_args()
    clarify_image = CLARIFY_IMAGE.get(args.region)
    if not clarify_image:
        raise SystemExit(f"No Clarify image mapped for region {args.region}")

    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    training_job, model_data_url = find_latest_training_job(sm, args.training_job_prefix)
    suffix = training_job[len(args.training_job_prefix) :]
    model_name = f"{args.model_prefix}{suffix}"[:63]
    job_name = f"stack-clarify{suffix}"[:63]

    out_root = args.output_path.rstrip("/")
    clarify_prefix = f"{out_root}/clarify/{model_name}/"
    work_prefix = f"{out_root}/clarify/_work/{model_name}/"
    out_bucket, out_key_prefix = _parse_s3(clarify_prefix)
    work_bucket, work_key_prefix = _parse_s3(work_prefix)

    logger.info("Training job : %s", training_job)
    logger.info("Model data   : %s", model_data_url)
    logger.info("Clarify out  : %s", clarify_prefix)

    headers, rows = build_headered_dataset(s3, args.test_data_path, args.max_rows)
    if not rows:
        logger.error("No rows under %s — run preprocess first", args.test_data_path)
        return 1
    if args.facet_column not in headers:
        raise SystemExit(
            f"Facet column {args.facet_column!r} not in headers {headers}. "
            "Pass --facet-column matching feature_schema."
        )
    logger.info("Clarify sample: %d rows, facet=%s>=%s", len(rows), args.facet_column, args.facet_threshold)

    # Upload headered dataset
    dataset_key = work_key_prefix.rstrip("/") + "/dataset.csv"
    csv_body = ",".join(headers) + "\n" + "\n".join(",".join(r) for r in rows) + "\n"
    s3.put_object(Bucket=work_bucket, Key=dataset_key, Body=csv_body, ContentType="text/csv")
    dataset_uri = f"s3://{work_bucket}/{dataset_key}"

    baseline = mean_baseline(headers, rows)
    analysis_config = {
        "dataset_type": "text/csv",
        "headers": headers,
        "label": "y",
        "label_values_or_threshold": [1],
        "facet": [
            {
                "name_or_index": args.facet_column,
                "value_or_threshold": [args.facet_threshold],
            }
        ],
        "probability_threshold": args.probability_threshold,
        "methods": {
            "pre_training_bias": {"methods": "all"},
            "post_training_bias": {"methods": "all"},
            "shap": {
                "baseline": baseline,
                "num_samples": min(50, max(10, len(rows) // 5)),
                "agg_method": "mean_abs",
                "use_logit": False,
                "save_local_shap_values": False,
            },
            "report": {
                "name": "report",
                "title": "Bank Marketing Clarify Report",
            },
        },
        "predictor": {
            "model_name": model_name,
            "instance_type": "ml.m5.xlarge",
            "initial_instance_count": 1,
            "accept_type": "text/csv",
            "content_type": "text/csv",
        },
    }
    config_key = work_key_prefix.rstrip("/") + "/analysis_config.json"
    s3.put_object(
        Bucket=work_bucket,
        Key=config_key,
        Body=json.dumps(analysis_config, indent=2),
        ContentType="application/json",
    )
    config_uri = f"s3://{work_bucket}/{config_key}"

    delete_model(sm, model_name)
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": args.image_uri, "ModelDataUrl": model_data_url},
        ExecutionRoleArn=args.execution_role_arn,
    )
    logger.info("Created ephemeral model %s", model_name)

    try:
        # Ensure unique job name on retries
        unique_job = f"{job_name}-{int(time.time())}"[-63:]
        sm.create_processing_job(
            ProcessingJobName=unique_job,
            RoleArn=args.execution_role_arn,
            AppSpecification={"ImageUri": clarify_image},
            ProcessingInputs=[
                {
                    "InputName": "analysis_config",
                    "S3Input": {
                        "S3Uri": config_uri,
                        "LocalPath": "/opt/ml/processing/input/config",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3CompressionType": "None",
                    },
                },
                {
                    "InputName": "dataset",
                    "S3Input": {
                        "S3Uri": dataset_uri,
                        "LocalPath": "/opt/ml/processing/input/data",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3CompressionType": "None",
                    },
                },
            ],
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "analysis_result",
                        "S3Output": {
                            "S3Uri": clarify_prefix,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge",
                    "VolumeSizeInGB": 30,
                }
            },
            StoppingCondition={"MaxRuntimeInSeconds": 3600},
        )
        logger.info("Started Clarify job %s — waiting…", unique_job)
        waiter = sm.get_waiter("processing_job_completed_or_stopped")
        waiter.wait(ProcessingJobName=unique_job, WaiterConfig={"Delay": 30, "MaxAttempts": 80})
        info = sm.describe_processing_job(ProcessingJobName=unique_job)
        status = info["ProcessingJobStatus"]
        if status != "Completed":
            logger.error("Clarify %s: %s", status, info.get("FailureReason"))
            return 1

        summary = {
            "training_job": training_job,
            "model_name": model_name,
            "model_data_url": model_data_url,
            "clarify_job": unique_job,
            "clarify_output_uri": clarify_prefix,
            "dataset_uri": dataset_uri,
            "facet_column": args.facet_column,
            "facet_threshold": args.facet_threshold,
            "sample_rows": len(rows),
            "report_html": clarify_prefix.rstrip("/") + "/report.html",
            "analysis_json": clarify_prefix.rstrip("/") + "/analysis.json",
        }
        summary_key = out_key_prefix.rstrip("/") + "/clarify_summary.json"
        s3.put_object(
            Bucket=out_bucket,
            Key=summary_key,
            Body=json.dumps(summary, indent=2),
            ContentType="application/json",
        )
        # Stable pointer for Model Card step
        latest_key = _parse_s3(out_root + "/clarify/latest/clarify_summary.json")[1]
        s3.put_object(
            Bucket=out_bucket,
            Key=latest_key,
            Body=json.dumps(summary, indent=2),
            ContentType="application/json",
        )
        logger.info("SUCCESS — Clarify report: %s", summary["report_html"])
        logger.info("Summary: s3://%s/%s", out_bucket, summary_key)
        return 0
    finally:
        delete_model(sm, model_name)


if __name__ == "__main__":
    sys.exit(main())
