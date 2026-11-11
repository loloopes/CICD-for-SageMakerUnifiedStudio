#!/usr/bin/env bash
# =============================================================================
# MLOps Infrastructure Setup
# =============================================================================
# One-time setup script that provisions the infrastructure needed for the
# MLOps pipeline: event-based deploy trigger.
#
# Usage:
#   ./scripts/setup-mlops-infra.sh <dev|test|prod> <account-id> <region> <project-name>
#
# Examples:
#   ./scripts/setup-mlops-infra.sh dev 973414239152 us-east-1 stack-dev
#   ./scripts/setup-mlops-infra.sh prod 987654321098 us-west-2 stack-prod
#
# Optional env vars (override resource names; set as GitHub Actions variables
# to avoid collisions when multiple projects share an account):
#   DEPLOY_TRIGGER_FUNCTION_NAME  (default: stack-airflow-deploy-trigger)
#   DEPLOY_TRIGGER_RULE_NAME      (default: stack-model-approved)
#   DEPLOY_TRIGGER_ROLE_NAME      (default: stack-airflow-deploy-trigger-role)
#   MODEL_PACKAGE_GROUP           (default: stack-prediction-models)
#
# What it creates:
#   1. Deploy Trigger (Lambda function, EventBridge rule, IAM role)
#   2. Project role inline policy MlopsTraining (train/eval/Clarify/Model Cards/
#      endpoint + PassRole + ECR + project S3 + logs)
#
# Optional env overrides:
#   DOMAIN_ID                 DataZone domain id (else discovered by project name)
#   PROJECT_USER_ROLE_ARN     Skip discovery; attach MlopsTraining to this role
#   MLFLOW_TRACKING_SERVER_NAME  (default: smus-integration-mlflow-${REGION})
#
# Prerequisites:
#   - AWS credentials for the target account
#   - Project deployed with aws-smus-cicd-cli
# =============================================================================

set -euo pipefail
export AWS_PAGER=""

ENV="${1:?Usage: $0 <dev|test|prod> <account-id> <region> <project-name>}"
ACCOUNT_ID="${2:?Usage: $0 <dev|test|prod> <account-id> <region> <project-name>}"
REGION="${3:?Usage: $0 <dev|test|prod> <account-id> <region> <project-name>}"
PROJECT_NAME="${4:?Usage: $0 <dev|test|prod> <account-id> <region> <project-name>}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MLOps Infrastructure Setup                              ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Environment : $ENV"
echo "║  Account     : $ACCOUNT_ID"
echo "║  Region      : $REGION"
echo "║  Project     : $PROJECT_NAME"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Verify we're in the right account
CURRENT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
if [ "$CURRENT_ACCOUNT" != "$ACCOUNT_ID" ]; then
  echo "ERROR: Authenticated to account $CURRENT_ACCOUNT but expected $ACCOUNT_ID"
  exit 1
fi

# =============================================================================
# PART 1: Event-Based Deploy Trigger
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Event-Based Deploy Trigger"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

FUNCTION_NAME="${DEPLOY_TRIGGER_FUNCTION_NAME:-stack-airflow-deploy-trigger}"
RULE_NAME="${DEPLOY_TRIGGER_RULE_NAME:-stack-model-approved}"
LAMBDA_ROLE_NAME="${DEPLOY_TRIGGER_ROLE_NAME:-stack-airflow-deploy-trigger-role}"
MODEL_PACKAGE_GROUP="${MODEL_PACKAGE_GROUP:-stack-prediction-models}"

# Step 1.1: Create Lambda execution role
echo ""
echo "[1.1] Lambda execution role: $LAMBDA_ROLE_NAME..."

LAMBDA_TRUST=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF
)

if aws iam get-role --role-name "$LAMBDA_ROLE_NAME" &>/dev/null; then
  echo "  Role exists."
else
  aws iam create-role \
    --role-name "$LAMBDA_ROLE_NAME" \
    --assume-role-policy-document "$LAMBDA_TRUST" \
    --description "Lambda role for Airflow deploy trigger" > /dev/null
  echo "  Created."
fi

LAMBDA_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "Logs",
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:*"
    },
    {
      "Sid": "SecretsManager",
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": "arn:aws:secretsmanager:${REGION}:${ACCOUNT_ID}:secret:stack/github-token*"
    }
  ]
}
EOF
)

aws iam put-role-policy \
  --role-name "$LAMBDA_ROLE_NAME" \
  --policy-name "AirflowDeployTriggerPolicy" \
  --policy-document "$LAMBDA_POLICY"
echo "  Policy attached."

echo "  Waiting for role propagation..."
sleep 10

LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"

# Step 1.2: Create Lambda function
echo "[1.2] Lambda function: $FUNCTION_NAME..."

LAMBDA_DIR=$(mktemp -d)
cat > "$LAMBDA_DIR/lambda_function.py" << 'PYEOF'
"""Lambda: Trigger GitHub Actions deploy workflow on model approval.

Triggered by EventBridge when a model is approved in the registry.
Sends a repository_dispatch event to GitHub Actions which then triggers
the dev -> test -> prod promotion cascade (mlops-promote.yml).

Environment Variables:
  GITHUB_TOKEN_SECRET_ARN - Secrets Manager ARN for GitHub PAT
  GITHUB_REPO             - GitHub repo (owner/repo format)
  PROJECT_NAME            - Project name to derive stage (e.g., stack-dev → dev)
"""
import json
import logging
import os
import urllib.request
import urllib.error
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_github_token():
    """Retrieve GitHub PAT from Secrets Manager."""
    secret_arn = os.environ["GITHUB_TOKEN_SECRET_ARN"]
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_arn)
    return response["SecretString"]


def handler(event, context):
    logger.info(f"Event: {json.dumps(event)}")

    detail = event.get("detail", {})
    approval_status = detail.get("ModelApprovalStatus", "")
    updated_fields = detail.get("UpdatedModelPackageFields", []) or []

    if approval_status != "Approved":
        logger.info(f"Ignoring non-approval event: {approval_status}")
        return {"statusCode": 200, "body": "Skipped - not an approval"}

    # Loop protection. Genuine approvals include ModelApprovalStatus in
    # UpdatedModelPackageFields (observed for console/API/UI). Deploy activity
    # (CreateModel from package, CustomerMetadata stamps, additional inference
    # specs) re-emits Model Package State Change with ModelApprovalStatus still
    # "Approved" but UpdatedModelPackageFields either empty [] or listing only
    # non-approval fields. The previous check used `if updated_fields and ...`,
    # which treated [] as falsy and re-dispatched — creating an infinite
    # approve → promote → deploy → state-change → approve loop. Require an
    # explicit ModelApprovalStatus field change before dispatching.
    if "ModelApprovalStatus" not in updated_fields:
        logger.info(
            "Ignoring update that did not change approval status "
            f"(UpdatedModelPackageFields={updated_fields}). Likely a deploy "
            "side-effect or metadata stamp, not a new approval."
        )
        return {"statusCode": 200, "body": "Skipped - approval status unchanged"}

    # Configuration
    github_repo = os.environ.get("GITHUB_REPO", "")
    project_name = os.environ.get("PROJECT_NAME", "stack-dev")

    # Derive stage from project name
    stage = project_name.rsplit("-", 1)[-1]

    model_package_arn = detail.get("ModelPackageArn", "")
    model_package_group = detail.get("ModelPackageGroupName", "stack-prediction-models")

    logger.info(f"Model approved: {model_package_arn}")
    logger.info(f"Triggering deploy for stage: {stage}")

    # Get GitHub token from Secrets Manager
    github_token = get_github_token()

    # Trigger GitHub Actions repository_dispatch
    url = f"https://api.github.com/repos/{github_repo}/dispatches"
    payload = {
        "event_type": "model-approved",
        "client_payload": {
            "stage": stage,
            "model_package_arn": model_package_arn,
            "model_package_group": model_package_group,
            "approval_status": "Approved",
        },
    }

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req) as response:
            status = response.status
            logger.info(f"GitHub dispatch triggered successfully (HTTP {status})")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        logger.error(f"GitHub API error: HTTP {e.code} - {body}")
        raise RuntimeError(f"Failed to trigger GitHub Actions: HTTP {e.code} - {body}")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Deploy workflow triggered",
            "stage": stage,
            "model_package_arn": model_package_arn,
            "github_repo": github_repo,
        }),
    }
PYEOF

(cd "$LAMBDA_DIR" && zip -q function.zip lambda_function.py)

# GITHUB_REPO is required so the deploy trigger dispatches model-approved
# events to the correct repository. Fail fast with a clear message if it's
# missing or not in owner/repo format.
if [[ -z "${GITHUB_REPO:-}" ]]; then
  echo "ERROR: GITHUB_REPO is not set. Export it as 'owner/repo' before running this script." >&2
  exit 1
fi
if [[ "$GITHUB_REPO" != */* ]]; then
  echo "ERROR: GITHUB_REPO must be in 'owner/repo' format (got: '$GITHUB_REPO')." >&2
  exit 1
fi
GITHUB_TOKEN_SECRET_ARN="arn:aws:secretsmanager:${REGION}:${ACCOUNT_ID}:secret:stack/github-token"

if aws lambda get-function --function-name "$FUNCTION_NAME" &>/dev/null; then
  aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --zip-file "fileb://${LAMBDA_DIR}/function.zip" > /dev/null
  echo "  Updated function code."

  # Wait for code update to complete before updating configuration
  aws lambda wait function-updated --function-name "$FUNCTION_NAME" 2>/dev/null || sleep 5

  aws lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --environment "Variables={GITHUB_REPO=${GITHUB_REPO},GITHUB_TOKEN_SECRET_ARN=${GITHUB_TOKEN_SECRET_ARN},PROJECT_NAME=${PROJECT_NAME}}" \
    --timeout 30 \
    --memory-size 128 > /dev/null
  echo "  Updated configuration."
else
  aws lambda create-function \
    --function-name "$FUNCTION_NAME" \
    --runtime python3.11 \
    --role "$LAMBDA_ROLE_ARN" \
    --handler lambda_function.handler \
    --zip-file "fileb://${LAMBDA_DIR}/function.zip" \
    --timeout 30 \
    --memory-size 128 \
    --environment "Variables={GITHUB_REPO=${GITHUB_REPO},GITHUB_TOKEN_SECRET_ARN=${GITHUB_TOKEN_SECRET_ARN},PROJECT_NAME=${PROJECT_NAME}}" > /dev/null
  echo "  Created."
fi

rm -rf "$LAMBDA_DIR"

LAMBDA_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"

# Step 1.3: EventBridge rule
echo "[1.3] EventBridge rule: $RULE_NAME..."

EVENT_PATTERN=$(cat <<EOF
{
  "source": ["aws.sagemaker"],
  "detail-type": ["SageMaker Model Package State Change"],
  "detail": {
    "ModelPackageGroupName": ["${MODEL_PACKAGE_GROUP}"],
    "ModelApprovalStatus": ["Approved"]
  }
}
EOF
)
# Note: previously this also required previousModelApprovalStatus != "Approved"
# at the rule level. SageMaker omits previousModelApprovalStatus from some
# state-change events (notably API-driven update-model-package calls), and
# EventBridge's content filter rejects events where the required field is
# missing - so user approvals were silently dropped.
#
# The Lambda handler still protects against the deploy-stamp loop on its own:
# it dispatches ONLY when UpdatedModelPackageFields contains
# ModelApprovalStatus (genuine approval). Empty [] or metadata-only updates
# are skipped. The rule is intentionally permissive: match every Approved
# package event, let the Lambda decide whether to dispatch.

aws events put-rule \
  --name "$RULE_NAME" \
  --event-pattern "$EVENT_PATTERN" \
  --description "Triggers Airflow deploy DAG when model is approved" \
  --state ENABLED > /dev/null
echo "  Rule created."

aws events put-targets \
  --rule "$RULE_NAME" \
  --targets "Id=airflow-deploy-trigger,Arn=${LAMBDA_ARN}" > /dev/null
echo "  Target added."

aws lambda add-permission \
  --function-name "$FUNCTION_NAME" \
  --statement-id "EventBridgeInvoke" \
  --action "lambda:InvokeFunction" \
  --principal "events.amazonaws.com" \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}" 2>/dev/null || true
echo "  Lambda permission granted."

# =============================================================================
# PART 2: Project execution role — train / Clarify / Model Cards
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project role: Clarify + Model Cards (MlopsTraining)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

MLFLOW_TRACKING_SERVER_NAME="${MLFLOW_TRACKING_SERVER_NAME:-smus-integration-mlflow-${REGION}}"
PROJECT_ROLE_ARN="${PROJECT_USER_ROLE_ARN:-}"
PROJECT_ID=""
S3_BUCKET=""
S3_PROJECT_PREFIX=""

if [[ -z "$PROJECT_ROLE_ARN" ]]; then
  echo "[2.1] Resolving DataZone project '$PROJECT_NAME' userRoleArn..."
  RESOLVE_OUT=$(python3 - "$REGION" "$PROJECT_NAME" "${DOMAIN_ID:-}" <<'PY'
import json, sys
import boto3

region, project_name, domain_id = sys.argv[1], sys.argv[2], sys.argv[3] or None
dz = boto3.client("datazone", region_name=region)

def find_project(did: str):
    token = None
    while True:
        kw = {"domainIdentifier": did, "maxResults": 50}
        if token:
            kw["nextToken"] = token
        resp = dz.list_projects(**kw)
        for p in resp.get("items") or []:
            if p.get("name") == project_name:
                return p.get("id")
        token = resp.get("nextToken")
        if not token:
            return None

if not domain_id:
    for d in dz.list_domains().get("items") or []:
        pid = find_project(d["id"])
        if pid:
            domain_id, project_id = d["id"], pid
            break
    else:
        raise SystemExit(f"Project '{project_name}' not found in any domain in {region}")
else:
    project_id = find_project(domain_id)
    if not project_id:
        raise SystemExit(f"Project '{project_name}' not found in domain {domain_id}")

role_arn = bucket = prefix = None
for env in dz.list_environments(domainIdentifier=domain_id, projectIdentifier=project_id).get("items") or []:
    detail = dz.get_environment(domainIdentifier=domain_id, identifier=env["id"])
    resources = {r.get("name"): r.get("value") for r in detail.get("provisionedResources") or []}
    role_arn = role_arn or resources.get("userRoleArn")
    shared = resources.get("nonGitProjectRepositoryLocation") or resources.get("s3BucketPath")
    if shared and shared.startswith("s3://") and not bucket:
        # s3://bucket/dzd-…/projectId[/stage]  → scope policy to projectId/*
        path = shared[len("s3://"):]
        bucket, _, rest = path.partition("/")
        parts = [p for p in rest.split("/") if p]
        if len(parts) >= 2:
            prefix = f"{parts[0]}/{parts[1]}"

if not role_arn:
    raise SystemExit(f"No userRoleArn for project '{project_name}' ({project_id})")
if not bucket or not prefix:
    raise SystemExit(f"Could not resolve project S3 location for '{project_name}'")

print(json.dumps({
    "domain_id": domain_id,
    "project_id": project_id,
    "role_arn": role_arn,
    "s3_bucket": bucket,
    "s3_prefix": prefix,
}))
PY
)
  PROJECT_ROLE_ARN=$(echo "$RESOLVE_OUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['role_arn'])")
  PROJECT_ID=$(echo "$RESOLVE_OUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['project_id'])")
  S3_BUCKET=$(echo "$RESOLVE_OUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['s3_bucket'])")
  S3_PROJECT_PREFIX=$(echo "$RESOLVE_OUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['s3_prefix'])")
  echo "  Domain/project : resolved ($PROJECT_ID)"
  echo "  Role           : $PROJECT_ROLE_ARN"
  echo "  S3             : s3://${S3_BUCKET}/${S3_PROJECT_PREFIX}/*"
else
  echo "[2.1] Using PROJECT_USER_ROLE_ARN=$PROJECT_ROLE_ARN"
  if [[ -z "${PROJECT_S3_BUCKET:-}" || -z "${PROJECT_S3_PREFIX:-}" ]]; then
    echo "ERROR: When PROJECT_USER_ROLE_ARN is set, also set PROJECT_S3_BUCKET and PROJECT_S3_PREFIX" >&2
    exit 1
  fi
  S3_BUCKET="$PROJECT_S3_BUCKET"
  S3_PROJECT_PREFIX="${PROJECT_S3_PREFIX%/}"
fi

PROJECT_ROLE_NAME="${PROJECT_ROLE_ARN##*/}"

echo "[2.2] Attaching inline policy MlopsTraining to $PROJECT_ROLE_NAME..."
MLOPS_TRAINING_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SageMakerTrainEvalRegister",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:StopTrainingJob",
        "sagemaker:ListTrainingJobs",
        "sagemaker:CreateProcessingJob",
        "sagemaker:DescribeProcessingJob",
        "sagemaker:StopProcessingJob",
        "sagemaker:ListProcessingJobs",
        "sagemaker:CreateModel",
        "sagemaker:DescribeModel",
        "sagemaker:DeleteModel",
        "sagemaker:ListModels",
        "sagemaker:CreateTransformJob",
        "sagemaker:DescribeTransformJob",
        "sagemaker:StopTransformJob",
        "sagemaker:CreateModelPackage",
        "sagemaker:DescribeModelPackage",
        "sagemaker:UpdateModelPackage",
        "sagemaker:ListModelPackages",
        "sagemaker:CreateModelPackageGroup",
        "sagemaker:DescribeModelPackageGroup",
        "sagemaker:ListModelPackageGroups",
        "sagemaker:CreateModelCard",
        "sagemaker:UpdateModelCard",
        "sagemaker:DescribeModelCard",
        "sagemaker:DeleteModelCard",
        "sagemaker:ListModelCards",
        "sagemaker:ListModelCardVersions",
        "sagemaker:CreateModelCardExportJob",
        "sagemaker:DescribeModelCardExportJob",
        "sagemaker:ListModelCardExportJobs",
        "sagemaker:AddTags",
        "sagemaker:ListTags",
        "sagemaker:DeleteTags",
        "sagemaker:CreatePresignedMlflowTrackingServerUrl",
        "sagemaker:CreateEndpoint",
        "sagemaker:DescribeEndpoint",
        "sagemaker:UpdateEndpoint",
        "sagemaker:DeleteEndpoint",
        "sagemaker:CreateEndpointConfig",
        "sagemaker:DescribeEndpointConfig",
        "sagemaker:DeleteEndpointConfig",
        "sagemaker:ListEndpoints",
        "sagemaker:ListEndpointConfigs",
        "sagemaker:InvokeEndpoint"
      ],
      "Resource": "*"
    },
    {
      "Sid": "MlflowTracking",
      "Effect": "Allow",
      "Action": "sagemaker-mlflow:*",
      "Resource": "arn:aws:sagemaker:${REGION}:${ACCOUNT_ID}:mlflow-tracking-server/${MLFLOW_TRACKING_SERVER_NAME}"
    },
    {
      "Sid": "ProjectBucket",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::${S3_BUCKET}/${S3_PROJECT_PREFIX}/*"
    },
    {
      "Sid": "ProjectBucketList",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::${S3_BUCKET}",
      "Condition": {
        "StringLike": {"s3:prefix": "${S3_PROJECT_PREFIX}/*"}
      }
    },
    {
      "Sid": "PullTrainingAndClarifyImages",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SageMakerLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/sagemaker/*"
    },
    {
      "Sid": "PassRoleToSageMaker",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "${PROJECT_ROLE_ARN}",
      "Condition": {
        "StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}
      }
    }
  ]
}
EOF
)

aws iam put-role-policy \
  --role-name "$PROJECT_ROLE_NAME" \
  --policy-name "MlopsTraining" \
  --policy-document "$MLOPS_TRAINING_POLICY"
echo "  MlopsTraining attached (train/eval/Clarify/Model Cards/PassRole/ECR)."

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Setup Complete                                          ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║  Deploy Trigger                                          ║"
echo "║    Rule:      $RULE_NAME"
echo "║    Lambda:    $FUNCTION_NAME"
echo "║    Role:      $LAMBDA_ROLE_NAME"
echo "║    Model grp: $MODEL_PACKAGE_GROUP"
echo "║    GitHub:    $GITHUB_REPO"
echo "║                                                          ║"
echo "║  Project role (MlopsTraining)                            ║"
echo "║    Role:      $PROJECT_ROLE_NAME"
echo "║    Includes:  Clarify processing + Model Cards           ║"
echo "║                                                          ║"
echo "║  Flow:                                                   ║"
echo "║    Model approved → EventBridge → Lambda                 ║"
echo "║    → GitHub Actions (mlops-promote.yml)                  ║"
echo "║    → dev → test → prod cascade with manual gates         ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Prerequisites:"
echo "  1. Store a GitHub PAT in Secrets Manager:"
echo "     aws secretsmanager create-secret \\"
echo "       --name stack/github-token \\"
echo "       --secret-string 'ghp_your_token_here'"
echo ""
echo "Next steps:"
echo "  cd examples/end-to-end-data-ml-pipeline/examples/mlops-pipeline"
echo "  aws-smus-cicd-cli deploy --manifest manifest.yaml --targets $ENV"
