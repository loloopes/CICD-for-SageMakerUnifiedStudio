#!/usr/bin/env python3
"""One-shot SageMaker Batch Transform smoke test.

Scores a features split with the **same model currently serving** a realtime
endpoint (dev / test / prod). Creates an ephemeral Model that mirrors the
endpoint container (image + ModelDataUrl + ``inference.py`` env) so Batch
Transform scores match InvokeEndpoint, then writes enriched CSV.

Example::

    python examples/end-to-end-data-ml-pipeline/scripts/batch_inference_smoke.py \\
      --shared-s3-uri s3://…/shared \\
      --region us-east-2 \\
      --role-arn arn:aws:iam::ACCOUNT:role/datazone_usr_role_… \\
      --endpoint-name stack-prediction-test \\
      --split validation
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
from pathlib import Path
from urllib.parse import urlparse

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("batch_inference_smoke")

DEFAULT_IMAGE = {
    "us-east-1": "257758044811.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1",
    "us-east-2": "257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.7-1",
}

STAGE_ENDPOINTS = {
    "dev": "stack-prediction-dev",
    "test": "stack-prediction-test",
    "prod": "stack-prediction-prod",
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


def _resolve_endpoint_serving(
    sm, endpoint_name: str, fallback_image: str
) -> tuple[str, str, str, str, str, str, dict]:
    """Return endpoint serving details for Batch Transform.

    ``(sm_model_name, model_data_url, image_uri, display_name, display_version,
    execution_role_arn, environment)``
    """
    ep = sm.describe_endpoint(EndpointName=endpoint_name)
    status = ep.get("EndpointStatus")
    if status != "InService":
        raise SystemExit(f"Endpoint {endpoint_name} is not InService (status={status})")

    cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
    sm_model_name = cfg["ProductionVariants"][0]["ModelName"]
    model = sm.describe_model(ModelName=sm_model_name)
    execution_role_arn = model.get("ExecutionRoleArn") or ""

    containers = []
    if model.get("PrimaryContainer"):
        containers.append(model["PrimaryContainer"])
    containers.extend(model.get("Containers") or [])
    if not containers:
        raise SystemExit(f"Model {sm_model_name} has no containers")

    primary = containers[0]
    model_data_url = primary.get("ModelDataUrl") or ""
    image_uri = primary.get("Image") or fallback_image
    env = dict(primary.get("Environment") or {})

    display_name = env.get("MODEL_NAME") or sm_model_name
    display_version = env.get("MODEL_VERSION") or "unknown"

    pkg_arn = primary.get("ModelPackageName")
    if pkg_arn:
        details = sm.describe_model_package(ModelPackageName=pkg_arn)
        display_name = details.get("ModelPackageGroupName") or display_name
        display_version = str(details.get("ModelPackageVersion") or display_version)
        containers_pkg = details.get("InferenceSpecification", {}).get("Containers") or []
        if not model_data_url and containers_pkg:
            model_data_url = containers_pkg[0].get("ModelDataUrl") or ""
            image_uri = containers_pkg[0].get("Image") or image_uri

    if not model_data_url:
        raise SystemExit(
            f"Could not resolve ModelDataUrl from endpoint {endpoint_name} "
            f"(model {sm_model_name})"
        )

    # Ensure script-mode serving (same path as InvokeEndpoint).
    env.setdefault("SAGEMAKER_PROGRAM", "inference.py")
    env.setdefault("SAGEMAKER_SUBMIT_DIRECTORY", "/opt/ml/model/code")
    env.setdefault("MODEL_NAME", display_name)
    env.setdefault("MODEL_VERSION", str(display_version))

    return (
        sm_model_name,
        model_data_url,
        image_uri,
        display_name,
        display_version,
        execution_role_arn,
        env,
    )


def _delete_model(sm, name: str) -> None:
    try:
        sm.delete_model(ModelName=name)
        logger.info("Deleted ephemeral model %s", name)
    except sm.exceptions.ClientError as e:
        if "Could not find model" not in str(e):
            logger.warning("Could not delete model %s: %s", name, e)


def _collect_transform_bodies(s3, out_bucket: str, out_prefix: str) -> list[str]:
    bodies: list[str] = []
    for key in sorted(_list_keys(s3, out_bucket, out_prefix)):
        if key.rstrip("/").endswith("enriched_predictions.csv"):
            continue
        if not (key.endswith(".out") or key.endswith(".csv.out")):
            continue
        bodies.append(
            s3.get_object(Bucket=out_bucket, Key=key)["Body"].read().decode("utf-8")
        )
    return bodies


def _parse_scores_from_bodies(bodies: list[str]) -> list[float]:
    """Parse raw floats or enriched CSV/JSON produced by inference.py."""
    scores: list[float] = []
    for body in bodies:
        text = body.strip()
        if not text:
            continue
        if text.startswith("[") or text.startswith("{"):
            payload = json.loads(text)
            if isinstance(payload, dict):
                payload = [payload]
            for item in payload:
                if isinstance(item, dict) and "score" in item:
                    scores.append(float(item["score"]))
                else:
                    scores.append(float(item))
            continue
        first = text.splitlines()[0].lower()
        if "score" in first and "," in first:
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                if not row or (row.get("score") or "").strip().lower() == "score":
                    continue
                scores.append(float(str(row["score"]).strip()))
            continue
        for line in text.splitlines():
            cell = line.strip().split(",")[0].strip() if line.strip() else ""
            if cell and cell.lower() != "score":
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
    transform_bodies: list[str] | None = None,
) -> str:
    """Prefer inference.py enriched output; otherwise merge scores + features."""
    key = out_prefix.rstrip("/") + "/enriched_predictions.csv"

    # If Batch Transform already returned enriched CSV/JSON from inference.py,
    # normalize into a single CSV (same schema as the live endpoint).
    if transform_bodies:
        rows: list[dict] = []
        for body in transform_bodies:
            text = body.strip()
            if not text:
                continue
            if text.startswith("[") or text.startswith("{"):
                payload = json.loads(text)
                if isinstance(payload, dict):
                    payload = [payload]
                for item in payload:
                    if isinstance(item, dict) and "score" in item:
                        rows.append(item)
                continue
            first = text.splitlines()[0].lower()
            if "score" in first and "features" in first:
                reader = csv.DictReader(io.StringIO(text))
                for row in reader:
                    if not row or (row.get("score") or "").strip().lower() == "score":
                        continue
                    rows.append(row)
        if rows:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                ["score", "features", "model_name", "model_version", "inference_timestamp"]
            )
            for row in rows:
                feats = row.get("features", [])
                if not isinstance(feats, str):
                    feats = json.dumps(feats)
                writer.writerow(
                    [
                        row.get("score"),
                        feats,
                        row.get("model_name") or model_name,
                        row.get("model_version") or model_version,
                        row.get("inference_timestamp") or inference_timestamp,
                    ]
                )
            s3.put_object(
                Bucket=out_bucket, Key=key, Body=buf.getvalue(), ContentType="text/csv"
            )
            return f"s3://{out_bucket}/{key}"

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

    s3.put_object(Bucket=out_bucket, Key=key, Body=buf.getvalue(), ContentType="text/csv")
    return f"s3://{out_bucket}/{key}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test SageMaker Batch Transform")
    parser.add_argument(
        "--shared-s3-uri",
        required=True,
        help="Project shared root for staging + output (usually the stage project)",
    )
    parser.add_argument(
        "--features-shared-s3-uri",
        default="",
        help="Optional shared root for feature CSVs (default: --shared-s3-uri). "
        "Use DEV shared when scoring test/prod endpoints.",
    )
    parser.add_argument(
        "--features-local-dir",
        default="",
        help="Optional local directory of feature CSVs (skips S3 GetObject on the "
        "features prefix). Used when the stage role cannot read DEV shared.",
    )
    parser.add_argument("--region", required=True)
    parser.add_argument("--role-arn", required=True, help="SageMaker / project execution role ARN")
    parser.add_argument(
        "--stage",
        choices=["dev", "test", "prod"],
        default="",
        help="Maps to endpoint stack-prediction-{stage} (alternative to --endpoint-name)",
    )
    parser.add_argument(
        "--endpoint-name",
        default="",
        help="Realtime endpoint whose deployed model to score with "
        "(default: stack-prediction-{stage})",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Which features/ split to score (default: train)",
    )
    parser.add_argument("--image-uri", default="", help="Fallback image if endpoint has none")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500,
        help="Optional cap on rows sent to Batch Transform (0 = all). Default 500.",
    )
    args = parser.parse_args()

    endpoint_name = args.endpoint_name or (
        STAGE_ENDPOINTS[args.stage] if args.stage else ""
    )
    if not endpoint_name:
        raise SystemExit("Pass --stage {dev,test,prod} or --endpoint-name")

    fallback_image = args.image_uri or DEFAULT_IMAGE.get(args.region, "")
    shared = args.shared_s3_uri.rstrip("/")
    features_shared = (args.features_shared_s3_uri or args.shared_s3_uri).rstrip("/")
    input_uri = f"{features_shared}/stack/mlops_pipeline/data/bank_mktg/features/{args.split}/"
    output_uri = f"{shared}/stack/mlops_pipeline/models/evaluation/batch_smoke/"
    local_dir = Path(args.features_local_dir) if args.features_local_dir else None

    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    (
        sm_model_name,
        model_data_url,
        image_uri,
        display_name,
        display_version,
        endpoint_role_arn,
        serving_env,
    ) = _resolve_endpoint_serving(sm, endpoint_name, fallback_image)
    # Prefer the endpoint model's role so Batch Transform can read the same
    # ModelDataUrl the realtime endpoint uses (esp. test/prod staged artifacts).
    create_role_arn = endpoint_role_arn or args.role_arn

    logger.info("Endpoint     : %s", endpoint_name)
    logger.info("SM Model     : %s", sm_model_name)
    logger.info("Model artifact: %s", model_data_url)
    logger.info("Image        : %s", image_uri)
    logger.info("Model name   : %s", display_name)
    logger.info("Model version: %s", display_version)
    logger.info("Create role  : %s", create_role_arn)
    logger.info("Serving prog : %s", serving_env.get("SAGEMAKER_PROGRAM"))
    logger.info("Input split  : %s", input_uri if not local_dir else f"local:{local_dir}")
    logger.info("Output root  : %s", output_uri)

    # Collect (filename, raw csv text) either from local prefetch or S3.
    csv_payloads: list[tuple[str, str]] = []
    if local_dir:
        if not local_dir.is_dir():
            raise SystemExit(f"--features-local-dir not a directory: {local_dir}")
        local_csvs = sorted(p for p in local_dir.iterdir() if p.suffix == ".csv")
        if not local_csvs:
            logger.error("No CSV files under %s", local_dir)
            return 1
        for path in local_csvs:
            csv_payloads.append((path.name, path.read_text(encoding="utf-8")))
        logger.info("Found %d local CSV file(s)", len(csv_payloads))
    else:
        in_bucket, in_prefix = _parse_s3(input_uri)
        csv_keys = _list_csv_keys(s3, in_bucket, in_prefix)
        if not csv_keys:
            logger.error("No CSV files under %s", input_uri)
            logger.error(
                "Run training_pipeline preprocess on DEV, or fix --features-shared-s3-uri "
                "/ --features-local-dir."
            )
            return 1
        for key in csv_keys:
            body = s3.get_object(Bucket=in_bucket, Key=key)["Body"].read().decode("utf-8")
            csv_payloads.append((key.rsplit("/", 1)[-1], body))
        logger.info("Found %d CSV file(s) under split", len(csv_payloads))

    ts = int(time.time())
    ephemeral_model = f"stack-batch-smoke-{ts}"
    transform_name = f"stack-batch-smoke-{ts}"

    # Ephemeral model mirrors the live endpoint: same artifact + inference.py so
    # Batch Transform scores match InvokeEndpoint (built-in XGBoost path does not).
    _delete_model(sm, ephemeral_model)
    sm.create_model(
        ModelName=ephemeral_model,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": serving_env,
        },
        ExecutionRoleArn=create_role_arn,
    )
    logger.info("Created ephemeral batch model %s (with inference.py)", ephemeral_model)

    # Stage feature-only CSVs under the *output* project's shared prefix so the
    # stage execution role can always read them (even when features live on DEV).
    out_bucket, out_root_prefix = _parse_s3(output_uri)
    infer_prefix = f"{out_root_prefix.rstrip('/')}/_input/{ephemeral_model}/"
    for key in _list_keys(s3, out_bucket, infer_prefix):
        s3.delete_object(Bucket=out_bucket, Key=key)

    feature_rows: list[list[str]] = []
    for filename, body in csv_payloads:
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
        dest = infer_prefix + filename
        s3.put_object(
            Bucket=out_bucket,
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

    infer_uri = f"s3://{out_bucket}/{infer_prefix}"
    out_uri = f"{output_uri.rstrip('/')}/{ephemeral_model}/"
    _, out_prefix = _parse_s3(out_uri)
    logger.info("Batch Transform input : %s", infer_uri)
    logger.info("Batch Transform output: %s", out_uri)

    try:
        # No SplitType: whole CSV is one payload (same as multi-row InvokeEndpoint).
        # Accept text/csv so inference.py output_fn returns the enriched schema.
        sm.create_transform_job(
            TransformJobName=transform_name,
            ModelName=ephemeral_model,
            TransformInput={
                "DataSource": {
                    "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": infer_uri}
                },
                "ContentType": "text/csv",
            },
            TransformOutput={
                "S3OutputPath": out_uri,
                "AssembleWith": "Line",
                "Accept": "text/csv",
            },
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

        transform_bodies = _collect_transform_bodies(s3, out_bucket, out_prefix)
        scores = _parse_scores_from_bodies(transform_bodies)
        inference_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        enriched_uri = _write_enriched_csv(
            s3,
            out_bucket,
            out_prefix,
            scores,
            feature_rows,
            model_name=display_name,
            model_version=display_version,
            inference_timestamp=inference_timestamp,
            transform_bodies=transform_bodies,
        )

        logger.info("SUCCESS — raw scores at %s", info["TransformOutput"]["S3OutputPath"])
        logger.info("SUCCESS — enriched CSV : %s", enriched_uri)
        logger.info(
            "Columns: score, features, model_name, model_version, inference_timestamp"
        )
        logger.info("Rows: %d", len(scores))

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
