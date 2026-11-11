#!/usr/bin/env bash
# Diagnose the latest training_pipeline failure (modelcard / clarify / train).
set -euo pipefail
export AWS_PAGER=""
REGION="${1:-us-east-2}"

echo "=== Latest processing jobs ==="
aws sagemaker list-processing-jobs --region "$REGION" --no-cli-pager \
  --sort-by CreationTime --sort-order Descending --max-results 8 \
  --query 'ProcessingJobSummaries[*].{Name:ProcessingJobName,Status:ProcessingJobStatus,Failure:FailureReason,Created:CreationTime}' \
  --output table

echo
echo "=== Latest stack-xgb training ==="
aws sagemaker list-training-jobs --region "$REGION" --no-cli-pager \
  --name-contains stack-xgb --sort-by CreationTime --sort-order Descending --max-results 3 \
  --query 'TrainingJobSummaries[*].{Name:TrainingJobName,Status:TrainingJobStatus,Created:CreationTime}' \
  --output table

dump_job_logs() {
  local JOB="$1"
  [ -z "$JOB" ] || [ "$JOB" = "None" ] && return 0
  echo
  echo "=== Logs: $JOB ==="
  local STREAMS
  STREAMS=$(aws logs describe-log-streams --region "$REGION" --no-cli-pager \
    --log-group-name /aws/sagemaker/ProcessingJobs \
    --log-stream-name-prefix "$JOB" \
    --query 'logStreams[*].logStreamName' --output text 2>/dev/null || true)
  if [ -z "$STREAMS" ] || [ "$STREAMS" = "None" ]; then
    echo "(no log streams yet)"
    return 0
  fi
  for S in $STREAMS; do
    echo "---- $S ----"
    aws logs get-log-events --region "$REGION" --no-cli-pager \
      --log-group-name /aws/sagemaker/ProcessingJobs \
      --log-stream-name "$S" --limit 100 --start-from-head \
      --query 'events[*].message' --output text | tr '\t' '\n' || true
  done
}

LATEST_MC=$(aws sagemaker list-processing-jobs --region "$REGION" --no-cli-pager \
  --name-contains stack-modelcard --sort-by CreationTime --sort-order Descending --max-results 1 \
  --query 'ProcessingJobSummaries[0].ProcessingJobName' --output text)
LATEST_CL=$(aws sagemaker list-processing-jobs --region "$REGION" --no-cli-pager \
  --name-contains stack-clarify --sort-by CreationTime --sort-order Descending --max-results 1 \
  --query 'ProcessingJobSummaries[0].ProcessingJobName' --output text)
LATEST_EV=$(aws sagemaker list-processing-jobs --region "$REGION" --no-cli-pager \
  --name-contains stack-eval --sort-by CreationTime --sort-order Descending --max-results 1 \
  --query 'ProcessingJobSummaries[0].ProcessingJobName' --output text)

dump_job_logs "$LATEST_MC"
dump_job_logs "$LATEST_CL"
dump_job_logs "$LATEST_EV"
