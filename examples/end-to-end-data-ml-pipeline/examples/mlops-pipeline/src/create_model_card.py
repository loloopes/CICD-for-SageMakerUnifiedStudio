#!/usr/bin/env python3
"""Create/update governance Model Card content for the bank-marketing model.

Two surfaces (Studio shows them differently):

1. **Model package Model Card** (what you see under Models → Registered models →
   version → Evaluate / Audit) — attached via ``update_model_package(ModelCard=…)``
   using the *simplified* package schema.
2. **Standalone Model Card** ``stack-bank-mktg-model-card`` — ``create/update_model_card``
   for Governance / Model cards lists (full Model Card schema).

Also writes Clarify bias metrics into evaluation_details and puts report URIs on
the package ``CustomerMetadataProperties`` so the registry version itself points
at ``…/evaluation/clarify/…/report.html``.

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


def _clip(s: str | None, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 3] + "..."


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
    bucket, prefix = _parse_s3(eval_uri)
    # Prefer stable pointer; also try common trailing-slash key mistake from older runs.
    candidates = [
        prefix.rstrip("/") + "/clarify/latest/clarify_summary.json",
        prefix.rstrip("/") + "/clarify/latest/clarify_summary.json/",
    ]
    for key in candidates:
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
            summary = json.loads(body)
            summary["_s3_uri"] = f"s3://{bucket}/{key.rstrip('/')}"
            return summary
        except s3.exceptions.ClientError:
            continue
    return None


def load_clarify_analysis(s3, clarify: dict | None) -> dict | None:
    """Load analysis.json (bias metrics) referenced by the clarify summary."""
    if not clarify:
        return None
    uri = clarify.get("analysis_json") or ""
    if not uri.startswith("s3://"):
        return None
    bucket, key = _parse_s3(uri)
    # analysis_json is a file URI; _parse_s3 forces trailing slash — strip it.
    key = key.rstrip("/")
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        return json.loads(body)
    except s3.exceptions.ClientError as e:
        logger.warning("Could not load clarify analysis %s: %s", uri, e)
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


def _holdout_metrics(eval_report: dict) -> list[dict]:
    metrics = eval_report.get("metrics") or {}
    metric_data = []
    for name in ("auc", "accuracy", "precision", "recall", "f1"):
        if name in metrics and isinstance(metrics[name], (int, float)):
            metric_data.append(
                {"name": name, "type": "number", "value": float(metrics[name])}
            )
    if "test_samples" in metrics and isinstance(metrics["test_samples"], (int, float)):
        metric_data.append(
            {
                "name": "test_samples",
                "type": "number",
                "value": float(metrics["test_samples"]),
            }
        )
    return metric_data


def _bias_metrics(analysis: dict | None) -> list[dict]:
    if not analysis:
        return []
    bias = analysis.get("bias") or {}
    pre = bias.get("pre_training") or {}
    post = bias.get("post_training") or {}
    out = []
    for name, val in (
        ("DPL", pre.get("DPL")),
        ("DI_labels", pre.get("DI")),
        ("DPPL", post.get("DPPL")),
        ("DI_predictions", post.get("DI")),
        ("RD", post.get("RD")),
        ("AD", post.get("AD")),
    ):
        if isinstance(val, (int, float)):
            out.append({"name": name, "type": "number", "value": float(val)})
    return out


def _clarify_note(clarify: dict | None) -> str:
    if not clarify:
        return "Clarify replacement not run for this cycle."
    return _clip(
        (
            f"Bias+explainability report at {clarify.get('report_html')} "
            f"(engine={clarify.get('engine', 'clarify-replacement')}; "
            f"facet {clarify.get('facet_column')} >= {clarify.get('facet_threshold')}; "
            f"sample_rows={clarify.get('sample_rows')}). "
            "Open report.html under project Files/S3 "
            "(…/models/evaluation/clarify/) — AWS Clarify processing is unavailable "
            "to new accounts."
        ),
        2096,
    )


def _numeric_version(eval_report: dict, pkg: dict | None) -> float:
    if pkg and isinstance(pkg.get("ModelPackageVersion"), (int, float)):
        return float(pkg["ModelPackageVersion"])
    raw = eval_report.get("model_version")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def build_standalone_card_content(
    eval_report: dict,
    clarify: dict | None,
    analysis: dict | None,
    pkg: dict | None,
) -> str:
    """Full Model Card schema for CreateModelCard / UpdateModelCard."""
    holdout = _holdout_metrics(eval_report)
    bias = _bias_metrics(analysis)
    numeric_version = _numeric_version(eval_report, pkg)

    pkg_note = ""
    if pkg:
        pkg_note = (
            f" Latest registry package: {pkg.get('ModelPackageGroupName')} "
            f"v{pkg.get('ModelPackageVersion')} ({pkg.get('ModelPackageArn')})."
        )

    evaluation_details = [
        {
            "name": "offline-holdout",
            "evaluation_observation": _clip(
                f"Holdout metrics from evaluate_model at "
                f"{eval_report.get('inference_timestamp')}. {_clarify_note(clarify)}",
                2096,
            ),
            "metric_groups": [
                {"name": "binary_classification_metrics", "metric_data": holdout}
            ],
        }
    ]
    if bias:
        evaluation_details.append(
            {
                "name": "clarify-bias",
                "evaluation_observation": _clip(
                    f"Clarify-replacement bias metrics "
                    f"(facet {clarify.get('facet_column') if clarify else 'age'}). "
                    f"Full HTML: {(clarify or {}).get('report_html', '')}",
                    2096,
                ),
                "metric_groups": [{"name": "bias_metrics", "metric_data": bias}],
            }
        )

    content = {
        "model_overview": {
            "model_description": (
                "XGBoost binary classifier for UCI Bank Marketing term-deposit "
                "subscription prediction (stack MLOps e2e example)."
            ),
            "model_artifact": [eval_report.get("model_data_url", "") or "s3://"],
            "algorithm_type": "xgboost",
            "problem_type": "Binary Classification",
            "model_creator": "stack-mlops-training-pipeline",
            "model_owner": "e2e-data-stack-ml-ops",
            "model_name": eval_report.get("model_name", "stack-model"),
            "model_version": numeric_version,
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
                "function": {"function": "Maximize", "facet": "auc"},
                "notes": "objective=binary:logistic, eval_metric=auc",
            },
            "training_observations": _clip(
                f"Training job {eval_report.get('training_job')}. "
                f"Evaluation report: {eval_report.get('_s3_uri')}."
                f"{pkg_note}",
                1024,
            ),
        },
        "evaluation_details": evaluation_details,
        "additional_information": {
            "ethical_considerations": (
                "Bank marketing scores can amplify targeting bias. Review Clarify "
                "pre-/post-training bias metrics before promoting models."
            ),
            "caveats_and_recommendations": _clip(
                "Re-run Clarify when features or label definitions change. "
                "Replace demo age facet with policy-approved protected attributes. "
                f"Card generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}.",
                2048,
            ),
        },
    }
    custom = _nonempty_details(
        {
            "evaluation_report_uri": _clip(eval_report.get("_s3_uri", ""), 1024),
            "clarify_summary_uri": _clip((clarify or {}).get("_s3_uri", ""), 1024),
            "clarify_report_html": _clip((clarify or {}).get("report_html", ""), 1024),
            "enriched_predictions_uri": _clip(
                eval_report.get("enriched_predictions_uri", ""), 1024
            ),
        }
    )
    if custom:
        content["additional_information"]["custom_details"] = custom
    return json.dumps(content)


def _nonempty_details(d: dict) -> dict:
    """Drop empty custom_details values (package schema is picky)."""
    return {k: v for k, v in d.items() if isinstance(v, str) and v.strip()}


def build_package_card_content(
    eval_report: dict,
    clarify: dict | None,
    analysis: dict | None,
) -> str:
    """Simplified ModelPackageModelCard schema (Studio Models → version tabs)."""
    holdout = _holdout_metrics(eval_report)
    bias = _bias_metrics(analysis)
    artifact = eval_report.get("model_data_url") or ""
    evaluation_details = [
        {
            "name": "offline-holdout",
            "evaluation_observation": _clip(
                f"Holdout metrics from evaluate_model at "
                f"{eval_report.get('inference_timestamp')}. {_clarify_note(clarify)}",
                2096,
            ),
            "metric_groups": [
                {"name": "binary_classification_metrics", "metric_data": holdout}
            ],
        }
    ]
    if bias:
        evaluation_details.append(
            {
                "name": "clarify-bias",
                "evaluation_observation": _clip(
                    f"Clarify-replacement bias (not AWS Clarify processing). "
                    f"HTML report: {(clarify or {}).get('report_html', '')}",
                    2096,
                ),
                "metric_groups": [{"name": "bias_metrics", "metric_data": bias}],
            }
        )

    # Keep package card minimal: omit training_details.objective_function —
    # that nested schema has rejected UpdateModelPackage in practice.
    custom = _nonempty_details(
        {
            "evaluation_report_uri": _clip(eval_report.get("_s3_uri", ""), 1024),
            "clarify_report_html": _clip((clarify or {}).get("report_html", ""), 1024),
            "clarify_summary_uri": _clip((clarify or {}).get("_s3_uri", ""), 1024),
        }
    )
    content = {
        "model_overview": {
            "model_creator": "stack-mlops-training-pipeline",
            "model_artifact": [artifact] if artifact else [],
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
            "training_observations": _clip(
                f"Training job {eval_report.get('training_job')}. "
                f"objective=binary:logistic eval_metric=auc. "
                f"Eval: {eval_report.get('_s3_uri')}.",
                1024,
            ),
        },
        "evaluation_details": evaluation_details,
        "additional_information": {
            "ethical_considerations": (
                "Bank marketing scores can amplify targeting bias. Review Clarify "
                "bias metrics before promoting models."
            ),
            "caveats_and_recommendations": _clip(
                "Open clarify report.html under …/models/evaluation/clarify/. "
                "AWS Clarify processing is closed to new customers; this pipeline "
                "uses a local bias+SHAP replacement. "
                f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}.",
                2048,
            ),
            **({"custom_details": custom} if custom else {}),
        },
    }
    return json.dumps(content)


def attach_package_card(sm, pkg: dict, package_content: str, clarify: dict | None) -> None:
    arn = pkg["ModelPackageArn"]
    meta = {
        "stack_model_card": CARD_NAME[:128],
        "stack_eval_engine": "evaluate_model",
    }
    if clarify and clarify.get("report_html"):
        # CustomerMetadata values max 256 chars — prefer the S3 key tail if needed.
        report = str(clarify["report_html"])
        meta["clarify_report_html"] = _clip(report, 256)
        meta["clarify_engine"] = _clip(str(clarify.get("engine", "local")), 256)

    # 1) ModelCard only (most common failure is Content schema).
    try:
        sm.update_model_package(
            ModelPackageArn=arn,
            ModelCard={
                "ModelCardContent": package_content,
                "ModelCardStatus": "Draft",
            },
        )
        logger.info("Attached ModelCard to package %s", arn)
    except sm.exceptions.ClientError as e:
        logger.error("update_model_package(ModelCard=…) failed: %s", e)
        # 2) Retry with a bare-minimum card (metrics + intended uses only).
        minimal = {
            "model_overview": {
                "model_creator": "stack-mlops-training-pipeline",
                "model_artifact": json.loads(package_content)
                .get("model_overview", {})
                .get("model_artifact", []),
            },
            "intended_uses": {
                "purpose_of_model": "Bank marketing term-deposit scoring (demo).",
                "intended_uses": "CI/CD MLOps demo only.",
                "risk_rating": "Low",
            },
            "evaluation_details": json.loads(package_content).get("evaluation_details")
            or [],
        }
        logger.info("Retrying package ModelCard with minimal content")
        sm.update_model_package(
            ModelPackageArn=arn,
            ModelCard={
                "ModelCardContent": json.dumps(minimal),
                "ModelCardStatus": "Draft",
            },
        )
        logger.info("Attached minimal ModelCard to package %s", arn)

    # 3) Metadata separately so a metadata rejection cannot roll back the card.
    try:
        sm.update_model_package(ModelPackageArn=arn, CustomerMetadataProperties=meta)
        logger.info("Attached CustomerMetadataProperties to package %s", arn)
    except sm.exceptions.ClientError as e:
        logger.warning("CustomerMetadataProperties update failed (non-fatal): %s", e)


def upsert_standalone_card(sm, name: str, content: str) -> None:
    exists = False
    try:
        sm.describe_model_card(ModelCardName=name)
        exists = True
    except sm.exceptions.ClientError as e:
        msg = str(e)
        if (
            "Could not find" not in msg
            and "does not exist" not in msg.lower()
            and "ResourceNotFound" not in msg
        ):
            raise

    if exists:
        sm.update_model_card(
            ModelCardName=name,
            Content=content,
            ModelCardStatus="Draft",
        )
        logger.info("Updated standalone Model Card %s", name)
    else:
        sm.create_model_card(
            ModelCardName=name,
            Content=content,
            ModelCardStatus="Draft",
            Tags=[
                {"Key": "project", "Value": "e2e-data-stack-ml-ops"},
                {"Key": "pipeline", "Value": "training_pipeline"},
            ],
        )
        logger.info("Created standalone Model Card %s", name)


def main() -> int:
    args = parse_args()
    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    eval_report = load_latest_evaluation(s3, args.evaluation_path)
    clarify = load_clarify_summary(s3, args.evaluation_path)
    analysis = load_clarify_analysis(s3, clarify)
    pkg = latest_package(sm, args.model_package_group)

    standalone = build_standalone_card_content(eval_report, clarify, analysis, pkg)
    package_content = build_package_card_content(eval_report, clarify, analysis)

    logger.info("Evaluation : %s", eval_report.get("_s3_uri"))
    logger.info("Clarify    : %s", (clarify or {}).get("_s3_uri", "none"))
    logger.info("Analysis   : %s", "yes" if analysis else "none")
    logger.info("Package    : %s", (pkg or {}).get("ModelPackageArn", "none"))
    logger.info("Model card : %s", args.model_card_name)

    package_ok = False
    standalone_ok = False

    # 1) Attach to registry version — this is what Studio Models UI shows.
    if pkg and pkg.get("ModelPackageArn"):
        try:
            attach_package_card(sm, pkg, package_content, clarify)
            package_ok = True
        except sm.exceptions.ClientError as e:
            logger.error("Package ModelCard attach failed after retries: %s", e)
    else:
        logger.warning("No model package found; skipping package ModelCard attach")

    # 2) Standalone card (Governance list / export).
    try:
        upsert_standalone_card(sm, args.model_card_name, standalone)
        standalone_ok = True
    except sm.exceptions.ClientError as e:
        logger.warning("Standalone Model Card upsert failed: %s", e)

    # Persist card snapshot next to evaluation artifacts (even on partial success).
    bucket, prefix = _parse_s3(args.evaluation_path)
    snap_key = prefix.rstrip("/") + f"/model_cards/{args.model_card_name}.json"
    s3.put_object(
        Bucket=bucket,
        Key=snap_key,
        Body=json.dumps(
            {
                "model_card_name": args.model_card_name,
                "status": "Draft",
                "package_ok": package_ok,
                "standalone_ok": standalone_ok,
                "package_arn": (pkg or {}).get("ModelPackageArn"),
                "package_content": json.loads(package_content),
                "standalone_content": json.loads(standalone),
            },
            indent=2,
        ),
        ContentType="application/json",
    )
    logger.info("Snapshot s3://%s/%s", bucket, snap_key)
    logger.info(
        "Studio: Models → Registered models → %s → version → Evaluate/Audit "
        "(package Model Card). Clarify HTML: …/models/evaluation/clarify/*/report.html",
        args.model_package_group,
    )

    if not package_ok and not standalone_ok:
        logger.error("Both package and standalone Model Card writes failed")
        return 1
    if not package_ok:
        logger.warning("Package Model Card missing; standalone/snapshot still written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
