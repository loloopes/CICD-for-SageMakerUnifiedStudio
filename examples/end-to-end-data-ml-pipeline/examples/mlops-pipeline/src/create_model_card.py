#!/usr/bin/env python3
"""Create or update a SageMaker Model Card from evaluation + Clarify outputs.

Reads the latest ``evaluation_report.json`` under the evaluation prefix and
optional ``clarify/latest/clarify_summary.json``, then creates/updates Model
Card ``stack-bank-mktg-model-card`` (Draft) so Studio shows governance content
for the bank-marketing XGBoost model.

Invoked as a SageMaker Processing job from ``training_pipeline`` after
register + Clarify.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("create_model_card")

CARD_NAME = "stack-bank-mktg-model-card"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", required=True)
    p.add_argument("--evaluation-path", required=True, help="S3 …/models/evaluation/")
    p.add_argument("--model-package-group", default="stack-prediction-models")
    p.add_argument("--model-card-name", default=CARD_NAME)
    return p.parse_args()


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri.rstrip("/") + "/")
    return parsed.netloc, parsed.path.lstrip("/")


def _list_keys(s3, bucket: str, prefix: str) -> list[str]:
    prefix = prefix.rstrip("/") + "/"
    keys, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []) or []:
            keys.append(obj["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def load_latest_evaluation(s3, eval_uri: str) -> dict:
    bucket, prefix = _parse_s3(eval_uri)
    reports = sorted(
        k for k in _list_keys(s3, bucket, prefix) if k.endswith("/evaluation_report.json")
    )
    if not reports:
        raise SystemExit(f"No evaluation_report.json under {eval_uri}")
    key = reports[-1]
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    report = json.loads(body)
    report["_s3_uri"] = f"s3://{bucket}/{key}"
    return report


def load_clarify_summary(s3, eval_uri: str) -> dict | None:
    bucket, prefix = _parse_s3(eval_uri.rstrip("/") + "/clarify/latest/")
    key = prefix.rstrip("/") + "/clarify_summary.json"
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        summary = json.loads(body)
        summary["_s3_uri"] = f"s3://{bucket}/{key}"
        return summary
    except s3.exceptions.ClientError:
        return None


def latest_package(sm, group: str) -> dict | None:
    try:
        resp = sm.list_model_packages(
            ModelPackageGroupName=group,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        pkgs = resp.get("ModelPackageSummaryList") or []
        return pkgs[0] if pkgs else None
    except sm.exceptions.ClientError as e:
        logger.warning("Could not list packages for %s: %s", group, e)
        return None


def build_card_content(eval_report: dict, clarify: dict | None, pkg: dict | None) -> str:
    metrics = eval_report.get("metrics") or {}
    metric_data = []
    for name in ("auc", "accuracy", "precision", "recall", "f1"):
        if name in metrics and isinstance(metrics[name], (int, float)):
            metric_data.append(
                {"name": name, "type": "number", "value": float(metrics[name])}
            )
    if "test_samples" in metrics:
        metric_data.append(
            {
                "name": "test_samples",
                "type": "number",
                "value": float(metrics["test_samples"]),
            }
        )

    clarify_note = "Clarify not run for this cycle."
    if clarify:
        clarify_note = (
            f"Bias+explainability report at {clarify.get('report_html')} "
            f"(engine={clarify.get('engine', 'clarify')}; "
            f"facet {clarify.get('facet_column')} >= {clarify.get('facet_threshold')}; "
            f"sample_rows={clarify.get('sample_rows')}). "
            f"Open report.html in project Files/S3 to review."
        )

    pkg_note = ""
    if pkg:
        pkg_note = (
            f" Latest registry package: {pkg.get('ModelPackageGroupName')} "
            f"v{pkg.get('ModelPackageVersion')} ({pkg.get('ModelPackageArn')})."
        )

    content = {
        "model_overview": {
            "model_description": (
                "XGBoost binary classifier for UCI Bank Marketing term-deposit "
                "subscription prediction (stack MLOps e2e example)."
            ),
            "model_artifact": [eval_report.get("model_data_url", "")],
            "algorithm_type": "xgboost",
            "problem_type": "Binary Classification",
            "model_creator": "stack-mlops-training-pipeline",
            "model_owner": "e2e-data-stack-ml-ops",
            "model_name": eval_report.get("model_name", "stack-model"),
            "model_version": str(eval_report.get("model_version", "unknown")),
        },
        "intended_uses": {
            "purpose_of_model": (
                "Score bank marketing leads for likelihood of term-deposit subscription."
            ),
            "intended_uses": (
                "Demo / CI-CD MLOps showcase for SageMaker Unified Studio. "
                "Not for production credit or employment decisions."
            ),
            "factors_affecting_model_efficiency": (
                "Feature distribution drift vs training reference; campaign seasonality; "
                "class imbalance (scale_pos_weight used in training)."
            ),
            "risk_rating": "Low",
            "explanations_for_risk_rating": (
                "Educational pipeline on public UCI data; clarify facet is illustrative (age)."
            ),
        },
        "training_details": {
            "objective_function": {
                "function": {"function": "BinaryCrossEntropy"},
                "notes": "objective=binary:logistic, eval_metric=auc",
            },
            "training_observations": (
                f"Training job {eval_report.get('training_job')}. "
                f"Evaluation report: {eval_report.get('_s3_uri')}."
                f"{pkg_note}"
            ),
        },
        "evaluation_details": [
            {
                "name": "offline-holdout",
                "evaluation_observation": (
                    f"Holdout metrics from evaluate_model at "
                    f"{eval_report.get('inference_timestamp')}. {clarify_note}"
                ),
                "metric_groups": [
                    {
                        "name": "binary_classification_metrics",
                        "metric_data": metric_data,
                    }
                ],
            }
        ],
        "additional_information": {
            "ethical_considerations": (
                "Bank marketing scores can amplify targeting bias. Review Clarify "
                "pre-/post-training bias metrics before promoting models."
            ),
            "caveats_and_recommendations": (
                "Re-run Clarify when features or label definitions change. "
                "Replace demo age facet with policy-approved protected attributes. "
                f"Card generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}."
            ),
            "custom_details": {
                "evaluation_report_uri": eval_report.get("_s3_uri", ""),
                "clarify_summary_uri": (clarify or {}).get("_s3_uri", ""),
                "clarify_report_html": (clarify or {}).get("report_html", ""),
                "enriched_predictions_uri": eval_report.get("enriched_predictions_uri", ""),
            },
        },
    }
    return json.dumps(content)


def main() -> int:
    args = parse_args()
    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    eval_report = load_latest_evaluation(s3, args.evaluation_path)
    clarify = load_clarify_summary(s3, args.evaluation_path)
    pkg = latest_package(sm, args.model_package_group)
    content = build_card_content(eval_report, clarify, pkg)

    logger.info("Evaluation : %s", eval_report.get("_s3_uri"))
    logger.info("Clarify    : %s", (clarify or {}).get("_s3_uri", "none"))
    logger.info("Package    : %s", (pkg or {}).get("ModelPackageArn", "none"))
    logger.info("Model card : %s", args.model_card_name)

    exists = False
    try:
        sm.describe_model_card(ModelCardName=args.model_card_name)
        exists = True
    except sm.exceptions.ClientError as e:
        msg = str(e)
        if "Could not find" not in msg and "does not exist" not in msg.lower() and "ResourceNotFound" not in msg:
            raise

    if exists:
        sm.update_model_card(
            ModelCardName=args.model_card_name,
            Content=content,
            ModelCardStatus="Draft",
        )
        logger.info("Updated Model Card %s", args.model_card_name)
    else:
        sm.create_model_card(
            ModelCardName=args.model_card_name,
            Content=content,
            ModelCardStatus="Draft",
            Tags=[
                {"Key": "project", "Value": "e2e-data-stack-ml-ops"},
                {"Key": "pipeline", "Value": "training_pipeline"},
            ],
        )
        logger.info("Created Model Card %s", args.model_card_name)

    # Persist card snapshot next to evaluation artifacts
    bucket, prefix = _parse_s3(args.evaluation_path)
    snap_key = prefix.rstrip("/") + f"/model_cards/{args.model_card_name}.json"
    s3.put_object(
        Bucket=bucket,
        Key=snap_key,
        Body=json.dumps(
            {
                "model_card_name": args.model_card_name,
                "status": "Draft",
                "content": json.loads(content),
            },
            indent=2,
        ),
        ContentType="application/json",
    )
    logger.info("Snapshot s3://%s/%s", bucket, snap_key)
    logger.info("View in Studio → Models / Governance → Model cards → %s", args.model_card_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
