#!/usr/bin/env python3
"""Bias + explainability for the latest stack-xgb model (Clarify replacement).

SageMaker Clarify processing is closed to new accounts (maintenance mode as of
2026-07-30). This script follows AWS guidance: compute the same standardized
bias metrics with pandas/numpy and feature attribution with SHAP (or XGBoost
gain as fallback), inside a normal Processing job.

Writes under ``…/models/evaluation/clarify/``:
  - analysis.json
  - report.html
  - clarify_summary.json (+ ``clarify/latest/`` pointer)

Demo facet: continuous ``age`` with threshold 50. Replace with real protected
attributes for production fairness work.

Invoked as a SageMaker Processing job from ``training_pipeline``.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlparse

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_clarify")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-job-prefix", default="stack-xgb")
    p.add_argument("--model-prefix", default="stack-clarify")
    # Kept for YAML compatibility; unused after Clarify container retirement.
    p.add_argument("--image-uri", default="", help="Unused (legacy Clarify Model image)")
    p.add_argument("--execution-role-arn", default="", help="Unused (legacy PassRole)")
    p.add_argument("--region", required=True)
    p.add_argument("--test-data-path", required=True, help="S3 prefix with label+feature CSVs")
    p.add_argument("--output-path", required=True, help="S3 prefix …/models/evaluation/")
    p.add_argument("--max-rows", type=int, default=500, help="Cap rows for cost/runtime")
    p.add_argument("--facet-column", default="age")
    p.add_argument("--facet-threshold", type=float, default=50.0)
    p.add_argument("--probability-threshold", type=float, default=0.5)
    return p.parse_args()


def _ensure_deps() -> None:
    """Install xgboost + shap when missing (Processing jobs usually have egress)."""
    need = []
    try:
        import xgboost  # noqa: F401
    except ImportError:
        need.append("xgboost==1.7.6")
    try:
        import shap  # noqa: F401
    except ImportError:
        need.append("shap==0.44.1")
    try:
        import pandas  # noqa: F401
    except ImportError:
        need.append("pandas")
    if not need:
        return
    logger.info("Installing deps: %s", need)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *need],
        stdout=subprocess.DEVNULL,
    )


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


def build_headered_dataset(s3, test_uri: str, max_rows: int):
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
            if len(parts) < len(headers):
                parts = parts + ["0"] * (len(headers) - len(parts))
            rows.append(parts[: len(headers)])
            if max_rows and len(rows) >= max_rows:
                return headers, rows
    return headers, rows


def download_booster(s3, model_data_url: str):
    import xgboost as xgb

    parsed = urlparse(model_data_url)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = os.path.join(tmp, "model.tar.gz")
        s3.download_file(bucket, key, tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp)
        model_path = os.path.join(tmp, "xgboost-model")
        if not os.path.exists(model_path):
            # fallback: first non-tar file
            candidates = [
                os.path.join(tmp, n)
                for n in os.listdir(tmp)
                if not n.endswith(".tar.gz") and os.path.isfile(os.path.join(tmp, n))
            ]
            if not candidates:
                raise SystemExit("xgboost-model not found in model.tar.gz")
            model_path = candidates[0]
        booster = xgb.Booster()
        booster.load_model(model_path)
        return booster


def predict_proba(booster, feature_rows: list[list[float]]) -> list[float]:
    import numpy as np
    import xgboost as xgb

    arr = np.asarray(feature_rows, dtype=float)
    dmat = xgb.DMatrix(arr)
    preds = booster.predict(dmat)
    return [float(p) for p in preds]


def bias_metrics(
    y_true: list[int],
    y_prob: list[float],
    facet_values: list[float],
    facet_threshold: float,
    prob_threshold: float,
) -> dict:
    """Standardized pre-/post-training bias metrics (Clarify-compatible formulas)."""
    import numpy as np

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    f = np.asarray(facet_values, dtype=float)
    yhat = (p >= prob_threshold).astype(int)

    # facet group = 1 when value >= threshold (e.g. age >= 50)
    g = (f >= facet_threshold).astype(int)
    n = len(y)
    if n == 0:
        return {}

    def rate(mask, labels):
        m = int(mask.sum())
        if m == 0:
            return None, 0
        return float(labels[mask].mean()), m

    # Pre-training: label rates
    p_pos_a, n_a = rate(g == 1, y)
    p_pos_d, n_d = rate(g == 0, y)
    dpl = None if p_pos_a is None or p_pos_d is None else p_pos_d - p_pos_a
    di_pre = None
    if p_pos_a not in (None, 0) and p_pos_d is not None:
        di_pre = p_pos_d / p_pos_a

    # Post-training: predicted positive rates + confusion-derived
    pp_a, _ = rate(g == 1, yhat)
    pp_d, _ = rate(g == 0, yhat)
    dppl = None if pp_a is None or pp_d is None else pp_d - pp_a
    di_post = None
    if pp_a not in (None, 0) and pp_d is not None:
        di_post = pp_d / pp_a

    def confusion(mask):
        yt, yh = y[mask], yhat[mask]
        tp = int(((yt == 1) & (yh == 1)).sum())
        tn = int(((yt == 0) & (yh == 0)).sum())
        fp = int(((yt == 0) & (yh == 1)).sum())
        fn = int(((yt == 1) & (yh == 0)).sum())
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def tpr_fpr(cm):
        tpr = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else None
        fpr = cm["fp"] / (cm["fp"] + cm["tn"]) if (cm["fp"] + cm["tn"]) else None
        return tpr, fpr

    cm_a, cm_d = confusion(g == 1), confusion(g == 0)
    tpr_a, fpr_a = tpr_fpr(cm_a)
    tpr_d, fpr_d = tpr_fpr(cm_d)
    rd = None if tpr_a is None or tpr_d is None else tpr_d - tpr_a  # recall difference
    ad = None if fpr_a is None or fpr_d is None else fpr_d - fpr_a  # acceptance/fpr diff

    return {
        "facet_group_definition": f"facet >= {facet_threshold}",
        "n_total": n,
        "n_facet": int((g == 1).sum()),
        "n_remainder": int((g == 0).sum()),
        "pre_training": {
            "CI": {"facet": p_pos_a, "remainder": p_pos_d},
            "DPL": dpl,
            "DI": di_pre,
        },
        "post_training": {
            "DPPL": dppl,
            "DI": di_post,
            "RD": rd,
            "AD": ad,
            "confusion_facet": cm_a,
            "confusion_remainder": cm_d,
        },
        "probability_threshold": prob_threshold,
    }


def explainability(booster, feature_rows: list[list[float]], feature_names: list[str]) -> dict:
    """Global feature attribution via SHAP TreeExplainer, else XGBoost gain."""
    import numpy as np

    method = "xgboost_gain"
    scores: dict[str, float] = {}

    try:
        import shap

        arr = np.asarray(feature_rows, dtype=float)
        # Cap SHAP sample for runtime
        sample = arr if len(arr) <= 100 else arr[:100]
        explainer = shap.TreeExplainer(booster)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        mean_abs = np.abs(np.asarray(sv, dtype=float)).mean(axis=0)
        scores = {feature_names[i]: float(mean_abs[i]) for i in range(len(feature_names))}
        method = "shap_tree_explainer"
    except Exception as e:  # noqa: BLE001
        logger.warning("SHAP unavailable (%s); falling back to XGBoost gain", e)
        gain = booster.get_score(importance_type="gain")
        # Keys are f0, f1, ...
        for i, name in enumerate(feature_names):
            scores[name] = float(gain.get(f"f{i}", 0.0))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "method": method,
        "global_mean_abs": {k: v for k, v in ranked},
        "top_features": [{"feature": k, "score": v} for k, v in ranked[:15]],
    }


def render_report_html(analysis: dict) -> str:
    bias = analysis.get("bias") or {}
    pre = bias.get("pre_training") or {}
    post = bias.get("post_training") or {}
    expl = analysis.get("explainability") or {}
    top = expl.get("top_features") or []

    def fmt(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4f}"
        return html.escape(str(v))

    rows = "".join(
        f"<tr><td>{html.escape(t['feature'])}</td><td>{fmt(t['score'])}</td></tr>" for t in top
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Bias &amp; Explainability Report</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;max-width:900px}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:.4rem .6rem;text-align:left}}
th{{background:#f4f4f4}}
.note{{color:#555;font-size:.9rem}}
</style></head><body>
<h1>Bank Marketing — Bias &amp; Explainability</h1>
<p class="note">Replacement for SageMaker Clarify (closed to new customers).
Metrics match Clarify formulas; attributions via {html.escape(str(expl.get('method')))}.</p>
<p><b>Model</b>: {html.escape(str(analysis.get('model_name')))}<br>
<b>Training job</b>: {html.escape(str(analysis.get('training_job')))}<br>
<b>Facet</b>: {html.escape(str(analysis.get('facet_column')))} &gt;= {html.escape(str(analysis.get('facet_threshold')))}
(n={bias.get('n_total')}, facet={bias.get('n_facet')}, remainder={bias.get('n_remainder')})<br>
<b>Generated</b>: {html.escape(str(analysis.get('generated_at')))}</p>
<h2>Pre-training bias</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>DPL (difference in positive proportions of labels)</td><td>{fmt(pre.get('DPL'))}</td></tr>
<tr><td>DI (disparate impact on labels)</td><td>{fmt(pre.get('DI'))}</td></tr>
</table>
<h2>Post-training bias</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>DPPL (difference in positive proportions of predicted labels)</td><td>{fmt(post.get('DPPL'))}</td></tr>
<tr><td>DI (disparate impact on predictions)</td><td>{fmt(post.get('DI'))}</td></tr>
<tr><td>RD (recall difference)</td><td>{fmt(post.get('RD'))}</td></tr>
<tr><td>AD (FPR difference)</td><td>{fmt(post.get('AD'))}</td></tr>
</table>
<h2>Global feature attribution</h2>
<table><tr><th>Feature</th><th>Score</th></tr>{rows}</table>
</body></html>
"""


def main() -> int:
    args = parse_args()
    _ensure_deps()

    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    training_job, model_data_url = find_latest_training_job(sm, args.training_job_prefix)
    suffix = training_job[len(args.training_job_prefix) :]
    model_name = f"{args.model_prefix}{suffix}"[:63]

    out_root = args.output_path.rstrip("/")
    clarify_prefix = f"{out_root}/clarify/{model_name}/"
    out_bucket, out_key_prefix = _parse_s3(clarify_prefix)

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
    logger.info(
        "Sample: %d rows, facet=%s>=%s", len(rows), args.facet_column, args.facet_threshold
    )

    facet_idx = headers.index(args.facet_column)
    y_true = [int(float(r[0])) for r in rows]
    facet_vals = [float(r[facet_idx]) for r in rows]
    feature_names = headers[1:]
    feature_rows = [[float(v) for v in r[1:]] for r in rows]

    booster = download_booster(s3, model_data_url)
    y_prob = predict_proba(booster, feature_rows)
    bias = bias_metrics(
        y_true, y_prob, facet_vals, args.facet_threshold, args.probability_threshold
    )
    expl = explainability(booster, feature_rows, feature_names)

    analysis = {
        "engine": "clarify-replacement-local",
        "note": (
            "SageMaker Clarify processing is unavailable to new customers; "
            "this report uses standardized bias formulas + SHAP/XGBoost gain."
        ),
        "training_job": training_job,
        "model_name": model_name,
        "model_data_url": model_data_url,
        "facet_column": args.facet_column,
        "facet_threshold": args.facet_threshold,
        "sample_rows": len(rows),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bias": bias,
        "explainability": expl,
    }

    report_html = render_report_html(analysis)
    analysis_key = out_key_prefix.rstrip("/") + "/analysis.json"
    report_key = out_key_prefix.rstrip("/") + "/report.html"
    s3.put_object(
        Bucket=out_bucket,
        Key=analysis_key,
        Body=json.dumps(analysis, indent=2),
        ContentType="application/json",
    )
    s3.put_object(
        Bucket=out_bucket,
        Key=report_key,
        Body=report_html.encode("utf-8"),
        ContentType="text/html",
    )

    summary = {
        "training_job": training_job,
        "model_name": model_name,
        "model_data_url": model_data_url,
        "clarify_job": "local-clarify-replacement",
        "clarify_output_uri": clarify_prefix,
        "facet_column": args.facet_column,
        "facet_threshold": args.facet_threshold,
        "sample_rows": len(rows),
        "engine": "clarify-replacement-local",
        "explainability_method": expl.get("method"),
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
    latest_key = _parse_s3(out_root + "/clarify/latest/clarify_summary.json")[1]
    s3.put_object(
        Bucket=out_bucket,
        Key=latest_key,
        Body=json.dumps(summary, indent=2),
        ContentType="application/json",
    )

    logger.info("SUCCESS — report: %s", summary["report_html"])
    logger.info("Summary: s3://%s/%s", out_bucket, summary_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
