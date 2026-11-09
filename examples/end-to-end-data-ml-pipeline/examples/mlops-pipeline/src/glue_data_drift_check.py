"""
Glue Job: Data Drift Check
==========================
Compares the live raw SOURCE (CSV today; S3 table later) against the
reference snapshot taken when training last read that source
(feature_engineering.py → features/drift_reference/).

Numeric columns: Population Stability Index (PSI) over fixed quantile bins.
Categorical columns: max absolute share delta across categories.

Always succeeds and writes result.json — the GitHub Actions gate decides
whether to retrain. Missing/empty reference => drift_detected=true (bootstrap).

Args:
    --CURRENT_PATH     s3://.../stack/data_pipeline/data/stack-sample.csv (or table path)
    --REFERENCE_PATH   s3://.../stack/mlops_pipeline/data/bank_mktg/features/drift_reference/
    --REPORT_PATH      s3://.../stack/mlops_pipeline/data/bank_mktg/drift_reports/<run>/
    --PSI_THRESHOLD    Drift threshold for PSI and categorical share delta (default 0.2)
"""

import json
import logging
import math
import re
import sys
from typing import Optional

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType, ShortType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NUMERIC_TYPES = (DoubleType, FloatType, IntegerType, LongType, ShortType)
PSI_EPSILON = 1e-6
N_BINS = 10

args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "CURRENT_PATH", "REFERENCE_PATH", "REPORT_PATH", "PSI_THRESHOLD"],
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Keep file paths as-is (e.g. *.csv); normalize directory prefixes with a slash.
def _normalize_path(path: str) -> str:
    path = path.strip()
    if path.lower().endswith((".csv", ".json", ".parquet")):
        return path
    return path.rstrip("/") + "/"


CURRENT_PATH = _normalize_path(args["CURRENT_PATH"])
REFERENCE_PATH = _normalize_path(args["REFERENCE_PATH"])
REPORT_PATH = _normalize_path(args["REPORT_PATH"])
PSI_THRESHOLD = float(args["PSI_THRESHOLD"])


def _clean_column_names(df: DataFrame) -> DataFrame:
    """Match DataOps ingest / FE snapshot column sanitization."""
    cleaned = []
    for c in df.columns:
        name = re.sub(r"[^a-zA-Z0-9_.]", "", c).replace(".", "_")
        cleaned.append(name)
    return df.toDF(*cleaned)


def _try_read_dataset(path: str) -> Optional[DataFrame]:
    """Read parquet directory or CSV source; return None if missing/empty."""
    try:
        if path.lower().endswith(".csv") or path.rstrip("/").lower().endswith(".csv"):
            df = (
                spark.read.option("header", "true")
                .option("inferSchema", "true")
                .option("sep", ";")
                .csv(path)
            )
            df = _clean_column_names(df)
        else:
            try:
                df = spark.read.parquet(path)
            except Exception:
                # Future S3 table / prefix of CSV objects
                df = (
                    spark.read.option("header", "true")
                    .option("inferSchema", "true")
                    .option("sep", ";")
                    .csv(path)
                )
                df = _clean_column_names(df)
        if df.head(1) is None:
            return None
        return df
    except Exception as exc:  # noqa: BLE001 — empty/missing S3 path is expected
        logger.warning(f"Could not read dataset at {path}: {exc}")
        return None


def _is_numeric(dtype) -> bool:
    return isinstance(dtype, NUMERIC_TYPES)


def _psi_for_column(ref_df: DataFrame, cur_df: DataFrame, col_name: str) -> float:
    """PSI using reference-quantile bins shared across both datasets."""
    from pyspark.ml.feature import Bucketizer

    ref_vals = ref_df.select(F.col(col_name).cast("double").alias(col_name)).dropna()
    cur_vals = cur_df.select(F.col(col_name).cast("double").alias(col_name)).dropna()

    probs = [i / N_BINS for i in range(N_BINS + 1)]
    edges = ref_vals.approxQuantile(col_name, probs, 0.01)
    unique_edges = []
    for e in edges:
        if not unique_edges or e > unique_edges[-1]:
            unique_edges.append(float(e))
    if len(unique_edges) < 2:
        return 0.0

    # Bucketizer needs open-ended outer edges
    splits = [float("-inf")] + unique_edges[1:-1] + [float("inf")]
    # Ensure strictly increasing after injecting inf
    cleaned = [splits[0]]
    for e in splits[1:]:
        if e > cleaned[-1]:
            cleaned.append(e)
    if len(cleaned) < 2:
        return 0.0

    bucketizer = Bucketizer(
        splits=cleaned, inputCol=col_name, outputCol="_bin", handleInvalid="keep"
    )
    n_bins = len(cleaned) - 1

    def _hist(df: DataFrame) -> list[float]:
        total = float(df.count()) or 1.0
        binned = bucketizer.transform(df)
        counts = {int(r["_bin"]): r["count"] for r in binned.groupBy("_bin").count().collect()}
        return [counts.get(i, 0) / total for i in range(n_bins)]

    ref_hist = _hist(ref_vals)
    cur_hist = _hist(cur_vals)
    psi = 0.0
    for p_ref, p_cur in zip(ref_hist, cur_hist):
        p_ref = max(p_ref, PSI_EPSILON)
        p_cur = max(p_cur, PSI_EPSILON)
        psi += (p_cur - p_ref) * math.log(p_cur / p_ref)
    return float(psi)


def _categorical_share_delta(ref_df: DataFrame, cur_df: DataFrame, col_name: str) -> float:
    """Max absolute difference in category share between reference and current."""
    def _shares(df: DataFrame) -> dict[str, float]:
        total = float(df.count()) or 1.0
        rows = (
            df.select(F.coalesce(F.col(col_name).cast("string"), F.lit("__NULL__")).alias(col_name))
            .groupBy(col_name)
            .count()
            .collect()
        )
        return {str(r[col_name]): r["count"] / total for r in rows}

    ref_s = _shares(ref_df)
    cur_s = _shares(cur_df)
    keys = set(ref_s) | set(cur_s)
    if not keys:
        return 0.0
    return float(max(abs(ref_s.get(k, 0.0) - cur_s.get(k, 0.0)) for k in keys))


def _write_result(payload: dict) -> None:
    """Write result.json to REPORT_PATH and a stable latest/ copy via Hadoop FS."""
    body = json.dumps(payload, indent=2, sort_keys=True)
    logger.info(f"Drift result:\n{body}")

    jvm = spark._jvm
    conf = spark._jsc.hadoopConfiguration()
    hpath = jvm.org.apache.hadoop.fs.Path
    uri = jvm.java.net.URI(REPORT_PATH)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, conf)

    def _put(s3_dir: str, content: str) -> None:
        out_path = hpath(s3_dir.rstrip("/") + "/result.json")
        stream = fs.create(out_path, True)
        stream.write(bytearray(content, "utf-8"))
        stream.close()
        logger.info(f"Wrote {out_path}")

    _put(REPORT_PATH, body)
    # REPORT_PATH = .../drift_reports/<run>/ → also publish .../drift_reports/latest/
    parent = REPORT_PATH.rstrip("/").rsplit("/", 1)[0]
    _put(parent + "/latest", body)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
logger.info(f"CURRENT_PATH   : {CURRENT_PATH}")
logger.info(f"REFERENCE_PATH : {REFERENCE_PATH}")
logger.info(f"REPORT_PATH    : {REPORT_PATH}")
logger.info(f"PSI_THRESHOLD  : {PSI_THRESHOLD}")

ref_df = _try_read_dataset(REFERENCE_PATH)
if ref_df is None:
    _write_result(
        {
            "drift_detected": True,
            "reason": "no_reference",
            "psi_threshold": PSI_THRESHOLD,
            "columns": [],
        }
    )
    job.commit()
    sys.exit(0)

cur_df = _try_read_dataset(CURRENT_PATH)
if cur_df is None:
    _write_result(
        {
            "drift_detected": True,
            "reason": "no_current_data",
            "psi_threshold": PSI_THRESHOLD,
            "columns": [],
        }
    )
    job.commit()
    sys.exit(0)

common_cols = [f.name for f in ref_df.schema.fields if f.name in cur_df.columns]
ref_types = {f.name: f.dataType for f in ref_df.schema.fields}

column_results = []
drift_detected = False

for col_name in sorted(common_cols):
    dtype = ref_types[col_name]
    if _is_numeric(dtype):
        metric = _psi_for_column(ref_df, cur_df, col_name)
        kind = "psi"
        drifted = metric >= PSI_THRESHOLD
    else:
        metric = _categorical_share_delta(ref_df, cur_df, col_name)
        kind = "share_delta"
        drifted = metric >= PSI_THRESHOLD

    column_results.append(
        {
            "column": col_name,
            "metric_type": kind,
            "metric_value": round(metric, 6),
            "threshold": PSI_THRESHOLD,
            "drifted": drifted,
        }
    )
    if drifted:
        drift_detected = True
        logger.info(f"DRIFT on {col_name}: {kind}={metric:.4f} >= {PSI_THRESHOLD}")
    else:
        logger.info(f"ok    {col_name}: {kind}={metric:.4f}")

_write_result(
    {
        "drift_detected": drift_detected,
        "reason": "compared" if column_results else "no_common_columns",
        "psi_threshold": PSI_THRESHOLD,
        "reference_path": REFERENCE_PATH,
        "current_path": CURRENT_PATH,
        "columns": column_results,
    }
)

job.commit()
logger.info("Data drift check completed successfully")
