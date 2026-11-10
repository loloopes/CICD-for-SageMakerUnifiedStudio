"""SageMaker XGBoost realtime inference with enriched JSON responses.

Packaged into the endpoint model artifact by ``deploy_model.py`` under
``code/inference.py``. Returns one object per input row:

    score, features (named map), model_name, model_version, inference_timestamp
"""

from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import xgboost as xgb

from feature_schema import FEATURE_COLUMNS, features_to_named


def _load_meta(model_dir: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    meta_path = os.path.join(model_dir, "code", "serving_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            raw = json.load(f)
        meta = dict(raw)
    feature_columns = meta.get("feature_columns") or FEATURE_COLUMNS
    return {
        "model_name": os.environ.get("MODEL_NAME") or str(meta.get("model_name", "unknown")),
        "model_version": os.environ.get("MODEL_VERSION")
        or str(meta.get("model_version", "unknown")),
        "feature_columns": list(feature_columns),
    }


def model_fn(model_dir: str) -> dict[str, Any]:
    """Load booster + serving identity baked in at deploy time."""
    model_path = os.path.join(model_dir, "xgboost-model")
    booster = xgb.Booster()
    booster.load_model(model_path)
    return {"booster": booster, "meta": _load_meta(model_dir)}


def input_fn(request_body: Any, content_type: str = "text/csv") -> dict[str, Any]:
    """Parse CSV body (one feature row per line) into DMatrix + feature rows."""
    ctype = (content_type or "text/csv").split(";")[0].strip().lower()
    if ctype not in ("text/csv", "text/plain"):
        raise ValueError(f"Unsupported content type: {content_type}")

    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    text = str(request_body).strip()
    if not text:
        raise ValueError("Empty request body")

    data = np.loadtxt(io.StringIO(text), delimiter=",")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    features = np.asarray(data, dtype=float).tolist()
    return {"dmatrix": xgb.DMatrix(data), "features": features}


def predict_fn(input_data: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    dmatrix = input_data["dmatrix"] if isinstance(input_data, dict) else input_data
    features = input_data.get("features") if isinstance(input_data, dict) else None
    scores = model["booster"].predict(dmatrix)
    return {
        "scores": np.atleast_1d(scores),
        "features": features,
        "meta": model["meta"],
    }


def output_fn(prediction: dict[str, Any], accept: str = "application/json") -> tuple[str, str]:
    """Enrich each score with named input features, model identity, and UTC timestamp."""
    scores = prediction["scores"]
    meta = prediction["meta"]
    features = prediction.get("features") or []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    for i, s in enumerate(scores.tolist()):
        feat_vals = features[i] if i < len(features) else []
        columns = meta.get("feature_columns") or FEATURE_COLUMNS
        named = features_to_named(feat_vals, columns=columns)
        rows.append(
            {
                "score": float(s),
                "features": named,
                "model_name": meta["model_name"],
                "model_version": str(meta["model_version"]),
                "inference_timestamp": ts,
            }
        )

    accept = (accept or "application/json").split(",")[0].strip().lower()
    if accept in ("text/csv", "text/plain"):
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["score", "features", "model_name", "model_version", "inference_timestamp"]
        )
        for r in rows:
            writer.writerow(
                [
                    r["score"],
                    json.dumps(r["features"], sort_keys=False),
                    r["model_name"],
                    r["model_version"],
                    r["inference_timestamp"],
                ]
            )
        return buf.getvalue(), "text/csv"

    return json.dumps(rows), "application/json"
