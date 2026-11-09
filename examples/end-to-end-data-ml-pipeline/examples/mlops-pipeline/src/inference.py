"""SageMaker XGBoost realtime inference with enriched JSON responses.

Packaged into the endpoint model artifact by ``deploy_model.py`` under
``code/inference.py``. Returns one object per input row:

    score, model_name, model_version, inference_timestamp
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import xgboost as xgb


def _load_meta(model_dir: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    meta_path = os.path.join(model_dir, "code", "serving_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            raw = json.load(f)
        meta = {k: str(v) for k, v in raw.items()}
    return {
        "model_name": os.environ.get("MODEL_NAME") or meta.get("model_name", "unknown"),
        "model_version": os.environ.get("MODEL_VERSION") or meta.get("model_version", "unknown"),
    }


def model_fn(model_dir: str) -> dict[str, Any]:
    """Load booster + serving identity baked in at deploy time."""
    model_path = os.path.join(model_dir, "xgboost-model")
    booster = xgb.Booster()
    booster.load_model(model_path)
    return {"booster": booster, "meta": _load_meta(model_dir)}


def input_fn(request_body: Any, content_type: str = "text/csv") -> xgb.DMatrix:
    """Parse CSV body (one feature row per line) into a DMatrix."""
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
    return xgb.DMatrix(data)


def predict_fn(input_data: xgb.DMatrix, model: dict[str, Any]) -> dict[str, Any]:
    scores = model["booster"].predict(input_data)
    return {"scores": np.atleast_1d(scores), "meta": model["meta"]}


def output_fn(prediction: dict[str, Any], accept: str = "application/json") -> tuple[str, str]:
    """Enrich each score with model identity and a per-request UTC timestamp."""
    scores = prediction["scores"]
    meta = prediction["meta"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [
        {
            "score": float(s),
            "model_name": meta["model_name"],
            "model_version": str(meta["model_version"]),
            "inference_timestamp": ts,
        }
        for s in scores.tolist()
    ]

    accept = (accept or "application/json").split(",")[0].strip().lower()
    if accept in ("text/csv", "text/plain"):
        header = "score,model_name,model_version,inference_timestamp"
        lines = [
            f"{r['score']},{r['model_name']},{r['model_version']},{r['inference_timestamp']}"
            for r in rows
        ]
        return "\n".join([header, *lines]) + "\n", "text/csv"

    return json.dumps(rows), "application/json"
