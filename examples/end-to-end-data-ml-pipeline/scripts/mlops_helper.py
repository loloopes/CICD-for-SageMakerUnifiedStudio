#!/usr/bin/env python3
"""
MLOps helper - consolidated CLI for the stack promotion pipeline.

Subcommands:
    select-package         Pick latest Approved model package version.
    stage-artifact         Copy model.tar.gz from source to a stage's bucket.
    check-deploy-status    Verify the latest deploy_pipeline processing job.
    capture-deploy-logs    Dump the latest deploy job's logs for postmortem.
    smoke-test             Invoke an endpoint and validate the response.
    read-drift-result      Read drift_check result.json and emit drift_detected.

Each subcommand keeps the same flags as the prior standalone scripts. Exit
codes are preserved so workflow YAMLs can keep relying on them.

Usage:
    python3 scripts/mlops_helper.py <subcommand> [--flag value ...]
"""

import argparse
import csv
import io
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Shared helpers
# =============================================================================


def write_github_output(pairs: dict[str, str]) -> None:
    """Append key=value pairs to $GITHUB_OUTPUT (or stdout when not in CI)."""
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a") as f:
            for k, v in pairs.items():
                f.write(f"{k}={v}\n")
    else:
        for k, v in pairs.items():
            print(f"{k}={v}")


# =============================================================================
# select-package
# =============================================================================


def cmd_select_package(args: argparse.Namespace) -> int:
    """Pick the latest Approved model package version in a group.

    Used at the start of promotion so the same ARN flows through all stages
    even if a new approval lands mid-promotion.
    """
    sm = boto3.client("sagemaker", region_name=args.region)
    response = sm.list_model_packages(
        ModelPackageGroupName=args.group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        logger.error(
            f"No Approved model versions in group '{args.group}'. "
            "Approve a version in the SageMaker Model Registry before promoting."
        )
        return 1

    latest = packages[0]
    arn = latest["ModelPackageArn"]
    version = latest["ModelPackageVersion"]

    logger.info("Selected model package for promotion:")
    logger.info(f"  ARN     : {arn}")
    logger.info(f"  Version : {version}")
    logger.info(f"  Created : {latest['CreationTime']}")

    write_github_output({"arn": arn, "version": str(version)})
    return 0


# =============================================================================
# stage-artifact
# =============================================================================


def _parse_domain_tags(raw: str) -> dict[str, str]:
    """Parse a comma-separated 'key=value' string into a tag dict."""
    tags: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --domain-tags entry '{item}', expected key=value")
        key, value = item.split("=", 1)
        tags[key.strip()] = value.strip()
    if not tags:
        raise ValueError("--domain-tags provided but no key=value pairs parsed")
    return tags


def _resolve_domain_id(
    region: str,
    domain_name: str | None = None,
    domain_tags: dict[str, str] | None = None,
) -> str:
    """Resolve a domain id via the packaged SMUS CLI helper.

    Reuses ``smus_cicd.helpers.datazone.resolve_domain_id`` — the same
    name/tags/auto-detect resolution the CLI uses for manifest deploys — instead
    of reimplementing domain lookup here. Raises loudly if no domain matches.
    """
    from smus_cicd.helpers import datazone as smus_datazone

    domain_id, _ = smus_datazone.resolve_domain_id(
        domain_name=domain_name, domain_tags=domain_tags, region=region
    )
    if not domain_id:
        raise ValueError(
            f"No domain found in region {region} "
            f"(name={domain_name!r}, tags={domain_tags!r})"
        )
    return domain_id


def _resolve_project_location(
    domain_id: str, project_name: str, region: str
) -> tuple[str, str]:
    """Return ``(bucket, key_prefix)`` for a project's ``default.s3_shared``.

    SageMaker Unified Studio puts every project in one domain bucket, under
    ``{domain_id}/{project_id}/shared/``. The bucket suffix is not the project
    id, so the location has to come from the connection rather than a name
    built from the account, region, and project id.
    """
    from smus_cicd.helpers import datazone as smus_datazone
    from smus_cicd.helpers.connections import (
        get_connection_s3_uri,
        get_project_connections,
    )

    project_id = smus_datazone.get_project_id_by_name(project_name, domain_id, region)
    if not project_id:
        raise ValueError(f"Project '{project_name}' not found in domain {domain_id}")

    connections = get_project_connections(project_id, domain_id, region)
    s3_uri = get_connection_s3_uri(connections)
    if not s3_uri.startswith("s3://"):
        raise ValueError(
            f"Project '{project_name}' has no default.s3_shared s3Uri "
            f"(got {s3_uri!r})"
        )

    bucket, _, prefix = s3_uri[5:].partition("/")
    return bucket, prefix.strip("/")


def _get_source_url_and_version(sm, model_package_arn: str) -> tuple[str, int]:
    pkg = sm.describe_model_package(ModelPackageName=model_package_arn)
    containers = pkg.get("InferenceSpecification", {}).get("Containers", [])
    if containers and containers[0].get("ModelDataUrl"):
        url = containers[0]["ModelDataUrl"]
    elif pkg.get("ModelDataUrl"):
        url = pkg["ModelDataUrl"]
    else:
        raise ValueError(f"No ModelDataUrl on package {model_package_arn}")
    return url, pkg.get("ModelPackageVersion", 0)


def _copy_if_changed(s3, source_url: str, dest_bucket: str, dest_key: str) -> None:
    """Copy with idempotency: re-runs skip when dest carries the source ETag.

    SSE-KMS makes the dest ETag differ from source even when bytes match, so
    we stamp the source ETag as `x-amz-meta-source-etag` and check that.
    """
    parsed = urlparse(source_url)
    src_bucket = parsed.netloc
    src_key = parsed.path.lstrip("/")

    src_meta = s3.head_object(Bucket=src_bucket, Key=src_key)
    src_etag = src_meta["ETag"]
    src_size = src_meta["ContentLength"]
    logger.info(f"  source size : {src_size:,} bytes")
    logger.info(f"  source etag : {src_etag}")

    try:
        dst_meta = s3.head_object(Bucket=dest_bucket, Key=dest_key)
        recorded = dst_meta.get("Metadata", {}).get("source-etag")
        if recorded == src_etag and dst_meta["ContentLength"] == src_size:
            logger.info(f"  destination already matches source (etag {src_etag}) - skipping copy")
            return
        logger.info(
            f"  destination present but source etag differs "
            f"(recorded={recorded!r}) - overwriting"
        )
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
            raise
        logger.info("  destination does not exist - copying")

    s3.copy_object(
        CopySource={"Bucket": src_bucket, "Key": src_key},
        Bucket=dest_bucket,
        Key=dest_key,
        Metadata={"source-etag": src_etag},
        MetadataDirective="REPLACE",
    )
    logger.info("  copy complete")


def cmd_stage_artifact(args: argparse.Namespace) -> int:
    """Copy model.tar.gz from the source bucket to a target stage bucket."""
    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    logger.info("=" * 60)
    logger.info("STAGE MODEL ARTIFACT")
    logger.info("=" * 60)
    logger.info(f"  Package ARN  : {args.model_package_arn}")
    logger.info(f"  Target proj  : {args.target_project_name}")
    if args.domain_id:
        domain_id = args.domain_id
    elif args.domain_name:
        domain_id = _resolve_domain_id(args.region, domain_name=args.domain_name)
    elif args.domain_tags:
        domain_id = _resolve_domain_id(
            args.region, domain_tags=_parse_domain_tags(args.domain_tags)
        )
    else:
        domain_id = _resolve_domain_id(args.region)
    logger.info(f"  Domain       : {domain_id}")
    logger.info("=" * 60)

    source_url, version = _get_source_url_and_version(sm, args.model_package_arn)
    logger.info(f"Source : {source_url}")
    logger.info(f"Version: {version}")

    dest_bucket, shared_prefix = _resolve_project_location(
        domain_id, args.target_project_name, args.region
    )
    relative = args.target_prefix.strip("/")
    # default.s3_shared already ends in /shared/; don't write shared/shared/...
    if shared_prefix.rstrip("/").endswith("/shared") and relative.startswith("shared/"):
        relative = relative[len("shared/") :]
    dest_key = f"{shared_prefix}/{relative}/v{version}/model.tar.gz"
    dest_url = f"s3://{dest_bucket}/{dest_key}"
    logger.info(f"Dest   : {dest_url}")

    _copy_if_changed(s3, source_url, dest_bucket, dest_key)

    write_github_output({"staged_url": dest_url})
    logger.info(f"✅ Staged: {dest_url}")
    return 0


# =============================================================================
# check-deploy-status
# =============================================================================


def cmd_check_deploy_status(args: argparse.Namespace) -> int:
    """Fail if the most recent matching processing job didn't Complete.

    When ``--endpoint-name`` is set, only jobs whose container args include
    that endpoint are considered. That stops a successful DEV deploy from
    being treated as a TEST/PROD success when no new job was created.

    Exit codes:
        0  Latest matching job is Completed.
        2  Latest matching job is Failed/Stopped.
        3  No matching job in the time window.
        4  Latest matching job is still InProgress (after monitor wait).
    """
    sm = boto3.client("sagemaker", region_name=args.region)
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=args.max_age_minutes)
    logger.info("Looking for processing jobs:")
    logger.info(f"  name_prefix   = {args.name_prefix}")
    logger.info(f"  endpoint_name = {args.endpoint_name or '<any>'}")
    logger.info(f"  region        = {args.region}")
    logger.info(f"  since         = {cutoff.isoformat()}")

    response = sm.list_processing_jobs(
        NameContains=args.name_prefix,
        CreationTimeAfter=cutoff,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=25,
    )
    jobs = response.get("ProcessingJobSummaries", [])
    if args.endpoint_name:
        matched = []
        for summary in jobs:
            detail = sm.describe_processing_job(
                ProcessingJobName=summary["ProcessingJobName"]
            )
            args_list = detail.get("AppSpecification", {}).get("ContainerArguments", [])
            if args.endpoint_name in args_list:
                matched.append(summary)
        jobs = matched

    if not jobs:
        logger.error(
            f"No processing jobs matching '{args.name_prefix}'"
            + (
                f" for endpoint '{args.endpoint_name}'"
                if args.endpoint_name
                else ""
            )
            + f" found in the last {args.max_age_minutes} minutes. "
            "The deploy DAG likely didn't trigger a deploy_model run."
        )
        return 3

    latest = jobs[0]
    name = latest["ProcessingJobName"]
    status = latest["ProcessingJobStatus"]
    logger.info(f"Latest processing job: {name}")
    logger.info(f"  Status: {status}")
    logger.info(f"  Created: {latest['CreationTime']}")

    if status == "Completed":
        logger.info("✅ Deploy processing job completed successfully")
        return 0
    if status == "InProgress":
        logger.error(
            f"Deploy processing job {name} is still InProgress. "
            "The monitor step exited before the DAG finished. Investigate."
        )
        return 4

    detail = sm.describe_processing_job(ProcessingJobName=name)
    reason = detail.get("FailureReason", "<no FailureReason on job>")
    logger.error("=" * 60)
    logger.error(f"DEPLOY PROCESSING JOB FAILED: {name}")
    logger.error("=" * 60)
    logger.error(f"  Status: {status}")
    logger.error(f"  Reason: {reason}")
    logger.error(f"Check CloudWatch logs at: /aws/sagemaker/ProcessingJobs/{name}")
    logger.error("=" * 60)
    return 2


# =============================================================================
# capture-deploy-logs
# =============================================================================


def _write_summary(detail: dict, name: str, output_dir: str) -> None:
    summary_path = os.path.join(output_dir, f"{name}.summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"ProcessingJobName: {name}\n")
        f.write(f"Status: {detail.get('ProcessingJobStatus')}\n")
        f.write(f"FailureReason: {detail.get('FailureReason', '')}\n")
        f.write(f"CreationTime: {detail.get('CreationTime')}\n")
        f.write(f"ProcessingEndTime: {detail.get('ProcessingEndTime', '')}\n")
        f.write("ContainerArguments:\n")
        for arg in detail.get("AppSpecification", {}).get("ContainerArguments", []):
            f.write(f"  {arg}\n")
    logger.info(f"  wrote {summary_path}")


def _dump_log_stream(logs, log_group: str, stream_name: str, output_dir: str) -> None:
    safe_name = stream_name.replace("/", "_")
    events_path = os.path.join(output_dir, f"{safe_name}.log")
    with open(events_path, "w") as f:
        token = None
        while True:
            kwargs = {
                "logGroupName": log_group,
                "logStreamName": stream_name,
                "startFromHead": True,
                "limit": 1000,
            }
            if token:
                kwargs["nextToken"] = token
            page = logs.get_log_events(**kwargs)
            for event in page.get("events", []):
                f.write(event["message"].rstrip() + "\n")
            next_token = page.get("nextForwardToken")
            if next_token == token or not page.get("events"):
                break
            token = next_token
    logger.info(f"  wrote {events_path}")


def cmd_capture_deploy_logs(args: argparse.Namespace) -> int:
    """Dump the latest deploy processing job's describe + CloudWatch logs.

    Best-effort: always returns 0 so the caller's `if: always()` upload step
    never fails because of capture issues.
    """
    sm = boto3.client("sagemaker", region_name=args.region)
    logs = boto3.client("logs", region_name=args.region)

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=args.max_age_minutes)
    response = sm.list_processing_jobs(
        NameContains=args.name_prefix,
        CreationTimeAfter=cutoff,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    jobs = response.get("ProcessingJobSummaries", [])
    if not jobs:
        logger.warning(
            f"No processing jobs matching '{args.name_prefix}' in the last "
            f"{args.max_age_minutes} minutes - nothing to capture."
        )
        return 0

    name = jobs[0]["ProcessingJobName"]
    status = jobs[0]["ProcessingJobStatus"]
    logger.info(f"Capturing logs for {name} (status={status})")

    os.makedirs(args.output_dir, exist_ok=True)
    detail = sm.describe_processing_job(ProcessingJobName=name)
    _write_summary(detail, name, args.output_dir)

    log_group = "/aws/sagemaker/ProcessingJobs"
    try:
        streams = logs.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=name,
            orderBy="LogStreamName",
            limit=10,
        ).get("logStreams", [])
    except logs.exceptions.ResourceNotFoundException:
        logger.warning(f"Log group {log_group} not found - skipping log capture.")
        return 0

    if not streams:
        logger.warning(f"No log streams for {name} - container may not have started.")
        return 0

    for stream in streams:
        _dump_log_stream(logs, log_group, stream["logStreamName"], args.output_dir)
    return 0


# =============================================================================
# smoke-test
# =============================================================================


# Built-in feature row sample. Edit if the stack processed feature schema
# changes. Two rows so we can verify scoring works on a tiny batch.
DEFAULT_SAMPLE = (
    "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6\n"
    "0.5,0.4,0.3,0.2,0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0\n"
)


def _load_payload(sample_csv: str) -> str:
    if sample_csv and os.path.isfile(sample_csv):
        with open(sample_csv) as f:
            return f.read()
    return DEFAULT_SAMPLE


def _parse_response(body_bytes: bytes) -> list[float]:
    text = body_bytes.decode("utf-8").strip()
    scores: list[float] = []
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        for cell in row:
            cell = cell.strip()
            if cell:
                scores.append(float(cell))
    return scores


def _resolve_endpoint_model_identity(sm, endpoint_name: str) -> tuple[str, str, str]:
    """Return ``(model_name, model_version, sagemaker_model_name)`` for an endpoint.

    Prefers Model Registry package group + version when the endpoint model was
    created from a package; otherwise falls back to the SageMaker Model name.
    """
    ep = sm.describe_endpoint(EndpointName=endpoint_name)
    cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
    sm_model_name = cfg["ProductionVariants"][0]["ModelName"]
    model = sm.describe_model(ModelName=sm_model_name)

    containers = list(model.get("Containers") or [])
    if model.get("PrimaryContainer"):
        containers = [model["PrimaryContainer"]] + containers

    for container in containers:
        pkg_arn = container.get("ModelPackageName")
        if not pkg_arn:
            continue
        details = sm.describe_model_package(ModelPackageName=pkg_arn)
        group = details.get("ModelPackageGroupName") or sm_model_name
        version = str(details.get("ModelPackageVersion", "unknown"))
        return group, version, sm_model_name

    return sm_model_name, "unknown", sm_model_name


def _format_enriched_predictions(
    scores: list[float], model_name: str, model_version: str, inference_timestamp: str
) -> str:
    """CSV with score, model_name, model_version, inference_timestamp."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["score", "model_name", "model_version", "inference_timestamp"])
    for score in scores:
        writer.writerow([score, model_name, model_version, inference_timestamp])
    return buf.getvalue()


def cmd_smoke_test(args: argparse.Namespace) -> int:
    """Invoke a SageMaker endpoint with a small CSV batch and validate scores.

    Exit codes:
        0  Endpoint InService and returned valid scores.
        2  Endpoint exists but is not InService.
        3  Wrong number of scores returned.
        4  Score(s) outside the configured min/max range.
        5  Endpoint does not exist (deploy was likely skipped).
    """
    logger.info("=" * 60)
    logger.info("ENDPOINT SMOKE TEST")
    logger.info("=" * 60)
    logger.info(f"  Endpoint : {args.endpoint_name}")
    logger.info(f"  Region   : {args.region}")
    logger.info(f"  Sample   : {args.sample_csv or '<built-in>'}")

    payload = _load_payload(args.sample_csv)
    rows = sum(1 for _ in payload.splitlines() if _.strip())
    logger.info(f"  Rows     : {rows}")

    runtime = boto3.client("sagemaker-runtime", region_name=args.region)
    sm = boto3.client("sagemaker", region_name=args.region)

    try:
        desc = sm.describe_endpoint(EndpointName=args.endpoint_name)
    except sm.exceptions.ClientError as e:
        msg = str(e)
        if "Could not find endpoint" in msg or "ValidationException" in msg:
            logger.error("=" * 60)
            logger.error(f"ENDPOINT NOT FOUND: {args.endpoint_name}")
            logger.error("=" * 60)
            logger.error("The deploy_pipeline DAG likely skipped deployment.")
            logger.error("Common causes:")
            logger.error("  1. No model versions in the Model Registry yet.")
            logger.error("  2. Model versions exist but none are Approved.")
            logger.error("  3. The deploy_pipeline DAG hit an error.")
            logger.error("=" * 60)
            return 5
        raise

    status = desc.get("EndpointStatus")
    logger.info(f"  Status   : {status}")
    if status != "InService":
        logger.error(f"Endpoint not InService (got {status})")
        return 2

    model_name, model_version, sm_model_name = _resolve_endpoint_model_identity(
        sm, args.endpoint_name
    )
    logger.info(f"  Model    : {model_name}")
    logger.info(f"  Version  : {model_version}")
    logger.info(f"  SM Model : {sm_model_name}")

    inference_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    response = runtime.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="text/csv",
        Body=payload,
    )
    scores = _parse_response(response["Body"].read())

    if len(scores) != rows:
        logger.error(f"Expected {rows} scores, got {len(scores)}")
        return 3

    out_of_range = [s for s in scores if not (args.score_min <= s <= args.score_max)]
    if out_of_range:
        logger.error(f"Out-of-range scores: {out_of_range}")
        return 4

    enriched = _format_enriched_predictions(
        scores, model_name, model_version, inference_timestamp
    )
    logger.info("Enriched live output (score, model_name, model_version, inference_timestamp):")
    for line in enriched.strip().splitlines():
        logger.info(f"  {line}")

    if args.output_csv:
        with open(args.output_csv, "w", encoding="utf-8") as f:
            f.write(enriched)
        logger.info(f"Wrote enriched predictions to {args.output_csv}")

    write_github_output(
        {
            "model_name": model_name,
            "model_version": model_version,
            "inference_timestamp": inference_timestamp,
            "scores": ",".join(str(s) for s in scores),
        }
    )
    logger.info("✅ Smoke test passed")
    return 0


# =============================================================================
# read-drift-result
# =============================================================================


def cmd_read_drift_result(args: argparse.Namespace) -> int:
    """Read drift_check result.json and emit ``drift_detected`` for GHA gates.

    Prefers ``.../drift_reports/latest/result.json`` written by the Glue job.
    Falls back to the most recently modified ``result.json`` under the reports
    prefix. Missing result => drift_detected=true (fail-open toward retrain).
    """
    if args.domain_id:
        domain_id = args.domain_id
    elif args.domain_name:
        domain_id = _resolve_domain_id(args.region, domain_name=args.domain_name)
    elif args.domain_tags:
        domain_id = _resolve_domain_id(
            args.region, domain_tags=_parse_domain_tags(args.domain_tags)
        )
    else:
        domain_id = _resolve_domain_id(args.region)

    bucket, shared_prefix = _resolve_project_location(
        domain_id, args.project_name, args.region
    )
    reports_rel = args.reports_prefix.strip("/")
    if shared_prefix.rstrip("/").endswith("/shared") and reports_rel.startswith("shared/"):
        reports_rel = reports_rel[len("shared/") :]
    base_prefix = f"{shared_prefix.rstrip('/')}/{reports_rel}".strip("/") + "/"

    s3 = boto3.client("s3", region_name=args.region)
    latest_key = f"{base_prefix}latest/result.json"

    logger.info("=" * 60)
    logger.info("READ DRIFT RESULT")
    logger.info("=" * 60)
    logger.info(f"  Project : {args.project_name}")
    logger.info(f"  Domain  : {domain_id}")
    logger.info(f"  Bucket  : {bucket}")
    logger.info(f"  Prefix  : {base_prefix}")

    body = None
    source_key = latest_key
    try:
        obj = s3.get_object(Bucket=bucket, Key=latest_key)
        body = obj["Body"].read().decode("utf-8")
        logger.info(f"  Source  : s3://{bucket}/{latest_key}")
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
            raise
        logger.warning("latest/result.json missing — scanning for newest result.json")

    if body is None:
        paginator = s3.get_paginator("list_objects_v2")
        newest = None
        for page in paginator.paginate(Bucket=bucket, Prefix=base_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/result.json"):
                    if newest is None or obj["LastModified"] > newest["LastModified"]:
                        newest = obj
        if newest is None:
            logger.warning("No drift result.json found — treating as drift (bootstrap)")
            write_github_output(
                {
                    "drift_detected": "true",
                    "reason": "no_result",
                    "result_uri": "",
                }
            )
            return 0
        source_key = newest["Key"]
        body = s3.get_object(Bucket=bucket, Key=source_key)["Body"].read().decode("utf-8")
        logger.info(f"  Source  : s3://{bucket}/{source_key}")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        logger.error(f"Invalid JSON in drift result: {exc}")
        write_github_output(
            {
                "drift_detected": "true",
                "reason": "invalid_json",
                "result_uri": f"s3://{bucket}/{source_key}",
            }
        )
        return 0

    detected = bool(payload.get("drift_detected", True))
    reason = str(payload.get("reason", "compared"))
    result_uri = f"s3://{bucket}/{source_key}"
    logger.info(f"  Detected: {detected}")
    logger.info(f"  Reason  : {reason}")
    if payload.get("columns"):
        drifted_cols = [c["column"] for c in payload["columns"] if c.get("drifted")]
        logger.info(f"  Columns : {drifted_cols or '(none drifted)'}")

    write_github_output(
        {
            "drift_detected": "true" if detected else "false",
            "reason": reason,
            "result_uri": result_uri,
        }
    )
    return 0


# =============================================================================
# CLI wiring
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<subcommand>")

    sp = sub.add_parser("select-package", help="Pick latest Approved model package version.")
    sp.add_argument("--group", required=True)
    sp.add_argument("--region", required=True)
    sp.set_defaults(func=cmd_select_package)

    sp = sub.add_parser("stage-artifact", help="Copy model.tar.gz to a stage bucket.")
    sp.add_argument("--model-package-arn", required=True)
    domain = sp.add_mutually_exclusive_group(required=False)
    domain.add_argument("--domain-id", help="DataZone domain id (dzd-...)")
    domain.add_argument("--domain-name", help="DataZone domain name")
    domain.add_argument(
        "--domain-tags",
        help=(
            "Comma-separated key=value tags to match a domain (mirrors the "
            "manifest region+tags resolution), e.g. 'purpose=smus-cicd-testing'. "
            "If none of --domain-id/--domain-name/--domain-tags is given, the "
            "single domain in the region is auto-detected."
        ),
    )
    sp.add_argument("--target-project-name", required=True)
    sp.add_argument("--region", required=True)
    sp.add_argument(
        "--target-prefix",
        default="shared/stack/mlops_pipeline/promoted-models",
        help=(
            "Path under the target project's default.s3_shared connection. "
            "A leading 'shared/' is dropped when that connection URI already "
            "ends in /shared/."
        ),
    )
    sp.set_defaults(func=cmd_stage_artifact)

    sp = sub.add_parser("check-deploy-status", help="Verify latest deploy processing job.")
    sp.add_argument("--region", required=True)
    sp.add_argument("--name-prefix", default="stack-deploy")
    sp.add_argument(
        "--endpoint-name",
        default="",
        help="Only match processing jobs that deploy this endpoint.",
    )
    sp.add_argument("--max-age-minutes", type=int, default=60)
    sp.set_defaults(func=cmd_check_deploy_status)

    sp = sub.add_parser("capture-deploy-logs", help="Dump latest deploy job's logs.")
    sp.add_argument("--region", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--name-prefix", default="stack-deploy")
    sp.add_argument("--max-age-minutes", type=int, default=60)
    sp.set_defaults(func=cmd_capture_deploy_logs)

    sp = sub.add_parser("smoke-test", help="Invoke endpoint and validate response.")
    sp.add_argument("--endpoint-name", required=True)
    sp.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    sp.add_argument("--sample-csv", default="", help="Optional CSV file with feature rows.")
    sp.add_argument("--score-min", type=float, default=0.0)
    sp.add_argument("--score-max", type=float, default=1.0)
    sp.add_argument(
        "--output-csv",
        default="",
        help="Optional path to write enriched predictions CSV "
        "(score, model_name, model_version, inference_timestamp).",
    )
    sp.set_defaults(func=cmd_smoke_test)

    sp = sub.add_parser(
        "read-drift-result",
        help="Read drift_check result.json and emit drift_detected for GHA.",
    )
    sp.add_argument("--project-name", required=True)
    sp.add_argument("--region", required=True)
    domain = sp.add_mutually_exclusive_group(required=False)
    domain.add_argument("--domain-id", help="DataZone domain id (dzd-...)")
    domain.add_argument("--domain-name", help="DataZone domain name")
    domain.add_argument(
        "--domain-tags",
        help=(
            "Comma-separated key=value tags to match a domain, "
            "e.g. 'purpose=smus-cicd-testing'."
        ),
    )
    sp.add_argument(
        "--reports-prefix",
        default="stack/mlops_pipeline/data/bank_mktg/drift_reports",
        help="Path under the project's default.s3_shared for drift reports.",
    )
    sp.set_defaults(func=cmd_read_drift_result)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
