"""
Bank Mktg Feature Engineering - AWS Glue Job
===================================================
Reads validated bank marketing data, encodes categorical features,
splits into train/validation/test sets, and registers them as
Glue Data Catalog tables for downstream SageMaker training.

Features:
  - Client demographics: age, job, marital, education
  - Financial profile: default, housing, loan
  - Campaign details: contact, month, day_of_week, duration, campaign, pdays, previous, poutcome
  - Economic indicators: emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed

Glue job args:
    --INPUT_PATH      s3://bucket/data/bank_mktg/raw/validated/
    --OUTPUT_PATH     s3://bucket/data/bank_mktg/features/
    --CATALOG_DB      Project catalog database name (provisioned by DataZone/SMUS)
    --TABLE_PREFIX    Prefix for catalog table names (train/validation/test)
    --TRAIN_RATIO     0.70
    --VAL_RATIO       0.15
    --TEST_RATIO      0.15
    --RANDOM_SEED     42
"""

import sys
import logging
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

args = getResolvedOptions(sys.argv, [
    'JOB_NAME', 'INPUT_PATH', 'OUTPUT_PATH', 'CATALOG_DB', 'TABLE_PREFIX',
    'TRAIN_RATIO', 'VAL_RATIO', 'TEST_RATIO', 'RANDOM_SEED'
])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

INPUT_PATH = args['INPUT_PATH']
OUTPUT_PATH = args['OUTPUT_PATH']
CATALOG_DB = args['CATALOG_DB']
TABLE_PREFIX = args['TABLE_PREFIX']
TRAIN_RATIO = float(args['TRAIN_RATIO'])
VAL_RATIO = float(args['VAL_RATIO'])
TEST_RATIO = float(args['TEST_RATIO'])
RANDOM_SEED = int(args['RANDOM_SEED'])

# -------------------------------------------------------------------------
# 1. Ensure catalog database exists, register source table, and load data
# -------------------------------------------------------------------------
logger.info(f"Ensuring catalog database {CATALOG_DB} exists...")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {CATALOG_DB}")
logger.info(f"Database {CATALOG_DB} ready")

source_table = f"{TABLE_PREFIX}_source"
logger.info(f"Checking if source table {CATALOG_DB}.{source_table} exists...")

table_exists = False
try:
    spark.sql(f"DESCRIBE TABLE {CATALOG_DB}.{source_table}")
    table_exists = True
    logger.info(f"Source table {CATALOG_DB}.{source_table} already exists")
except Exception:
    logger.info(f"Source table {CATALOG_DB}.{source_table} does not exist, creating...")

if not table_exists:
    spark.sql(f"""
        CREATE EXTERNAL TABLE {CATALOG_DB}.{source_table}
        USING PARQUET
        LOCATION '{INPUT_PATH}'
    """)
    row_count = spark.sql(f"SELECT COUNT(*) FROM {CATALOG_DB}.{source_table}").collect()[0][0]
    logger.info(f"Registered source table {CATALOG_DB}.{source_table} -> {INPUT_PATH} ({row_count} rows)")

logger.info(f"Reading validated data from catalog table {CATALOG_DB}.{source_table}")
df = spark.sql(f"SELECT * FROM {CATALOG_DB}.{source_table}")
logger.info(f"Loaded {df.count()} rows, {len(df.columns)} columns")

# Snapshot raw transformed input (DataOps schema) for future drift checks.
# Overwritten on every successful preprocess so the gate compares against
# the distribution the currently registered model was trained on.
logger.info(f"Writing drift reference snapshot to {OUTPUT_PATH}drift_reference/")
spark.read.parquet(INPUT_PATH).write.mode("overwrite").parquet(
    f"{OUTPUT_PATH}drift_reference/"
)

# -------------------------------------------------------------------------
# 2. Encode target variable
# -------------------------------------------------------------------------
logger.info("Encoding target variable...")
df = df.withColumn('y', F.when(F.col('y') == 'yes', 1.0).otherwise(0.0))

# -------------------------------------------------------------------------
# 3. Encode categorical features via StringIndexer
# -------------------------------------------------------------------------
logger.info("Encoding categorical features...")
categorical_cols = [
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'poutcome'
]

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_cols
]
pipeline = Pipeline(stages=indexers)
df = pipeline.fit(df).transform(df)

# -------------------------------------------------------------------------
# 4. Cast numeric columns
# -------------------------------------------------------------------------
numeric_cols = [
    'age', 'duration', 'campaign', 'pdays', 'previous',
    'emp_var_rate', 'cons_price_idx', 'cons_conf_idx',
    'euribor3m', 'nr_employed'
]
for col_name in numeric_cols:
    if col_name in df.columns:
        df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

# -------------------------------------------------------------------------
# 5. Select final feature set (target first for XGBoost CSV convention)
# -------------------------------------------------------------------------
indexed_cols = [f"{c}_idx" for c in categorical_cols]
feature_columns = numeric_cols + indexed_cols
output_cols = ['y'] + feature_columns

df_final = df.select([F.col(c).cast(DoubleType()) for c in output_cols])
logger.info(f"Final dataset: {df_final.count()} rows, {len(output_cols)} columns")

# -------------------------------------------------------------------------
# 6. Train / Validation / Test split
# -------------------------------------------------------------------------
logger.info(f"Splitting: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
train_df, val_df, test_df = df_final.randomSplit(
    [TRAIN_RATIO, VAL_RATIO, TEST_RATIO], seed=RANDOM_SEED
)
logger.info(f"Train: {train_df.count()}, Val: {val_df.count()}, Test: {test_df.count()}")

# -------------------------------------------------------------------------
# 7. Write to S3 as Parquet and register in Glue Data Catalog
# -------------------------------------------------------------------------
splits = {
    'train': train_df,
    'validation': val_df,
    'test': test_df,
}

# Also write CSV (no header) for SageMaker XGBoost direct consumption
train_df.write.csv(f"{OUTPUT_PATH}train/", header=False, mode='overwrite')
val_df.write.csv(f"{OUTPUT_PATH}validation/", header=False, mode='overwrite')
test_df.write.csv(f"{OUTPUT_PATH}test/", header=False, mode='overwrite')

# Write test data WITHOUT label column for batch transform inference
test_features_only = test_df.drop('y')
test_features_only.write.csv(f"{OUTPUT_PATH}test_transform/", header=False, mode='overwrite')

# Baseline copy with headers for drift monitoring
train_df.write.csv(f"{OUTPUT_PATH}baseline/", header=True, mode='overwrite')

# Register each split as a Glue Catalog table (Parquet-backed)
# Uses the project catalog database provisioned by DataZone/SMUS
for split_name, split_df in splits.items():
    table_name = f"{TABLE_PREFIX}_{split_name}"
    table_path = f"{OUTPUT_PATH}{split_name}_parquet/"

    split_df.write.parquet(table_path, mode='overwrite')

    # Drop and recreate to ensure schema is current
    spark.sql(f"DROP TABLE IF EXISTS {CATALOG_DB}.{table_name}")
    spark.sql(f"""
        CREATE EXTERNAL TABLE {CATALOG_DB}.{table_name}
        USING PARQUET
        LOCATION '{table_path}'
    """)
    row_count = spark.sql(f"SELECT COUNT(*) FROM {CATALOG_DB}.{table_name}").collect()[0][0]
    logger.info(f"Registered {CATALOG_DB}.{table_name} -> {table_path} ({row_count} rows)")

logger.info(f"All features written to {OUTPUT_PATH} and registered in {CATALOG_DB}")
job.commit()
logger.info("Bank marketing feature engineering Glue job completed successfully")
