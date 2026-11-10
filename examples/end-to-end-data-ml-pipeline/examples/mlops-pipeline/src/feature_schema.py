"""Feature column order for bank marketing XGBoost inference.

Must match ``feature_engineering.py`` output after dropping the label ``y``
(CSV rows consumed by Batch Transform and realtime InvokeEndpoint).
"""

NUMERIC_COLS = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
]

CATEGORICAL_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

# Indexed categoricals written by StringIndexer as ``{col}_idx``.
INDEXED_COLS = [f"{c}_idx" for c in CATEGORICAL_COLS]

# Order of feature-only CSV columns (no label).
FEATURE_COLUMNS = NUMERIC_COLS + INDEXED_COLS


def features_to_named(values, columns=None) -> dict:
    """Zip a feature vector to column names (extras keyed as fN)."""
    cols = list(columns) if columns is not None else FEATURE_COLUMNS
    vals = list(values) if values is not None else []
    named = {}
    for i, v in enumerate(vals):
        key = cols[i] if i < len(cols) else f"f{i}"
        named[key] = float(v)
    return named
