#!/usr/bin/env python3
"""One-shot SageMaker Batch Transform smoke test.

Creates an ephemeral Model from the latest completed training job, strips the
label column from a features split (train/validation/test), runs Batch
Transform, then writes an enriched CSV with:

    score, features, model_name, model_version, inference_timestamp

Deletes the ephemeral Model when finished.

Example (e2e-data-stack-ml-ops-dev)::

    python examples/end-to-end-data-ml-pipeline/scripts/batch_inference_smoke.py \\
      --shared-s3-uri s3://amazon-sagemaker-442460399726-us-east-2-a1501210c441/dzd-5e6vtm4oytol49/6d3pq5vm70uxm1/shared \\
      --region us-east-2 \\
      --role-arn arn:aws:iam::442460399726:role/datazone_usr_role_6d3pq5vm70uxm1_cac5ixjkrb173d \\
      --split train
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("batch_inference_smoke")

DEFAULT_IMAGE = {
    "us-east-1": "257758044811.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1",
    "us-east-2": "257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.7-1",
}


def _parse_s3(uri: str) -> tuple[str, str]:
    p = urlparse(uri.rstrip("/") + "/")
    if p.scheme != "s3" or not p.netloc:
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    return p.netloc, p.path.lstrip("/")


def _list_csv_keys(s3, bucket: str, prefix: str) -> list[str]:
    prefix = prefix.rstrip("/") + "/"
    keys: list[str] = []
    token = None
    while True:
        kw: dict = {"Bucket": bucket, "Prefix": prefix}
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


def _list_keys(s3, bucket: str, prefix: str) -> list[str]:
    prefix = prefix.rstrip("/") + "/"
    keys: list[str] = []
    token = None
    while True:
        kw: dict = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []) or []:
            keys.append(obj["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def _latest_training_job(sm, prefix: str) -> tuple[str, str]:
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


def _resolve_model_identity(
    sm, package_group: str, training_job: str
) -> tuple[str, str]:
    """Prefer latest Approved registry version; else use training job as version."""
    model_name = package_group or "stack-prediction-models"
    try:
        resp = sm.list_model_packages(
            ModelPackageGroupName=package_group,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        pkgs = resp.get("ModelPackageSummaryList") or []
        if pkgs:
            version = str(pkgs[0].get("ModelPackageVersion") or training_job)
            return model_name, version
    except sm.exceptions.ClientError as e:
        logger.warning("Could not resolve registry version (%s); using training job", e)
    return model_name, training_job


def _delete_model(sm, name: str) -> None:
    try:
        sm.delete_model(ModelName=name)
        logger.info("Deleted ephemeral model %s", name)
    except sm.exceptions.ClientError as e:
        if "Could not find model" not in str(e):
            logger.warning("Could not delete model %s: %s", name, e)


def _read_transform_scores(s3, out_bucket: str, out_prefix: str) -> list[float]:
    scores: list[float] = []
    for key in sorted(_list_keys(s3, out_bucket, out_prefix)):
        if not key.endswith(".out") and not key.endswith(".csv.out"):
            continue
        body = s3.get_object(Bucket=out_bucket, Key=key)["Body"].read().decode("utf-8")
        for line in body.splitlines():
            cell = line.strip().split(",")[0].strip() if line.strip() else ""
            if cell:
                scores.append(float(cell))
    return scores


def _write_enriched_csv(
    s3,
    out_bucket: str,
    out_prefix: str,
    scores: list[float],
    feature_rows: list[list[str]],
    model_name: str,
    model_version: str,
    inference_timestamp: str,
) -> str:
    if len(scores) != len(feature_rows):
        raise SystemExit(
            f"Score/feature mismatch: {len(scores)} scores vs {len(feature_rows)} feature rows"
        )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["score", "features", "model_name", "model_version", "inference_timestamp"]
    )
    for score, feats in zip(scores, feature_rows):
        # Named map matches realtime inference.py / evaluate_model enrichment.
        named = {f"f{i}": float(v) for i, v in enumerate(feats)}
        writer.writerow(
            [
                score,
                json.dumps(named, separators=(",", ":")),
                model_name,
                model_version,
                inference_timestamp,
            ]
        )

    key = out_prefix.rstrip("/") + "/enriched_predictions.csv"
    body = buf.getvalue()
    s3.put_object(Bucket=out_bucket, Key=key, Body=body, ContentType="text/csv")
    return f"s3://{out_bucket}/{key}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test SageMaker Batch Transform")
    parser.add_argument(
        "--shared-s3-uri",
        required=True,
        help="Project shared root, e.g. s3://…/shared (no trailing stack/... needed)",
    )
    parser.add_argument("--region", required=True)
    parser.add_argument("--role-arn", required=True, help="SageMaker / project execution role ARN")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Which features/ split to score (default: train)",
    )
    parser.add_argument("--training-job-prefix", default="stack-xgb")
    parser.add_argument(
        "--model-package-group",
        default="stack-prediction-models",
        help="Registry group used for model_name / model_version (default: stack-prediction-models)",
    )
    parser.add_argument("--image-uri", default="", help="Defaults to region XGBoost 1.7-1 image")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500,
        help="Optional cap on rows sent to Batch Transform (0 = all). Default 500 for a quick smoke.",
    )
    args = parser.parse_args()

    image_uri = args.image_uri or DEFAULT_IMAGE.get(args.region)
    if not image_uri:
        raise SystemExit(f"Pass --image-uri for region {args.region}")

    shared = args.shared_s3_uri.rstrip("/")
    input_uri = f"{shared}/stack/mlops_pipeline/data/bank_mktg/features/{args.split}/"
    output_uri = f"{shared}/stack/mlops_pipeline/models/evaluation/batch_smoke/"

    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    training_job, model_data_url = _latest_training_job(sm, args.training_job_prefix)
    registry_name, registry_version = _resolve_model_identity(
        sm, args.model_package_group, training_job
    )
    logger.info("Training job : %s", training_job)
    logger.info("Model artifact: %s", model_data_url)
    logger.info("Model name   : %s", registry_name)
    logger.info("Model version: %s", registry_version)
    logger.info("Input split  : %s", input_uri)

    in_bucket, in_prefix = _parse_s3(input_uri)
    csv_keys = _list_csv_keys(s3, in_bucket, in_prefix)
    if not csv_keys:
        logger.error("No CSV files under %s", input_uri)
        logger.error("Run training_pipeline preprocess first, or fix --shared-s3-uri.")
        return 1
    logger.info("Found %d CSV file(s) under split", len(csv_keys))

    ts = int(time.time())
    ephemeral_model = f"stack-batch-smoke-{ts}"
    transform_name = f"stack-batch-smoke-{ts}"

    _delete_model(sm, ephemeral_model)
    sm.create_model(
        ModelName=ephemeral_model,
        PrimaryContainer={"Image": image_uri, "ModelDataUrl": model_data_url},
        ExecutionRoleArn=args.role_arn,
    )
    logger.info("Created model %s", ephemeral_model)

    infer_prefix = f"{in_prefix.rstrip('/').rsplit('/', 1)[0]}/inference/{ephemeral_model}/"
    for key in _list_keys(s3, in_bucket, infer_prefix):
        s3.delete_object(Bucket=in_bucket, Key=key)

    feature_rows: list[list[str]] = []
    for key in csv_keys:
        body = s3.get_object(Bucket=in_bucket, Key=key)["Body"].read().decode("utf-8")
        lines = [ln for ln in body.strip().split("\n") if ln.strip()]
        feat_lines: list[str] = []
        for ln in lines:
            parts = ln.split(",")
            if len(parts) < 2:
                continue
            feats = parts[1:]
            feature_rows.append(feats)
            feat_lines.append(",".join(feats))
            if args.max_rows and len(feature_rows) >= args.max_rows:
                break
        dest = infer_prefix + key.rsplit("/", 1)[-1]
        s3.put_object(
            Bucket=in_bucket,
            Key=dest,
            Body="\n".join(feat_lines) + ("\n" if feat_lines else ""),
            ContentType="text/csv",
        )
        logger.info("Staged %s (%d feature rows so far)", dest, len(feature_rows))
        if args.max_rows and len(feature_rows) >= args.max_rows:
            break

    if not feature_rows:
        logger.error("No feature rows prepared — aborting")
        _delete_model(sm, ephemeral_model)
        return 1

    infer_uri = f"s3://{in_bucket}/{infer_prefix}"
    out_uri = f"{output_uri.rstrip('/')}/{ephemeral_model}/"
    out_bucket, out_prefix = _parse_s3(out_uri)
    logger.info("Batch Transform input : %s", infer_uri)
    logger.info("Batch Transform output: %s", out_uri)

    try:
        sm.create_transform_job(
            TransformJobName=transform_name,
            ModelName=ephemeral_model,
            TransformInput={
                "DataSource": {
                    "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": infer_uri}
                },
                "ContentType": "text/csv",
                "SplitType": "Line",
            },
            TransformOutput={"S3OutputPath": out_uri, "AssembleWith": "Line"},
            TransformResources={"InstanceType": "ml.m5.large", "InstanceCount": 1},
        )
        logger.info("Started transform job %s — waiting…", transform_name)
        waiter = sm.get_waiter("transform_job_completed_or_stopped")
        waiter.wait(
            TransformJobName=transform_name,
            WaiterConfig={"Delay": 20, "MaxAttempts": 60},
        )
        info = sm.describe_transform_job(TransformJobName=transform_name)
        status = info["TransformJobStatus"]
        if status != "Completed":
            logger.error("Transform %s: %s", status, info.get("FailureReason"))
            return 1

        scores = _read_transform_scores(s3, out_bucket, out_prefix)
        inference_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        enriched_uri = _write_enriched_csv(
            s3,
            out_bucket,
            out_prefix,
            scores,
            feature_rows,
            model_name=registry_name,
            model_version=registry_version,
            inference_timestamp=inference_timestamp,
        )

        logger.info("SUCCESS — raw scores at %s", info["TransformOutput"]["S3OutputPath"])
        logger.info("SUCCESS — enriched CSV : %s", enriched_uri)
        logger.info(
            "Columns: score, features, model_name, model_version, inference_timestamp"
        )
        logger.info("Rows: %d", len(scores))

        # Preview first few enriched lines
        preview = s3.get_object(
            Bucket=out_bucket, Key=out_prefix.rstrip("/") + "/enriched_predictions.csv"
        )["Body"].read().decode("utf-8")
        for line in preview.splitlines()[:4]:
            logger.info("  %s", line)
        return 0
    finally:
        _delete_model(sm, ephemeral_model)


if __name__ == "__main__":
    sys.exit(main())
