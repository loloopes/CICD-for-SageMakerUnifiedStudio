#!/usr/bin/env bash
# =============================================================================
# Setup GitHub Actions OIDC Role for smus-cli deployments
# =============================================================================
# Creates the OIDC provider and an IAM role that can:
#   1. Assume the specified deployment / project role (smus-cli deploys)
#   2. Provision the MLOps deploy trigger (Lambda + EventBridge + its IAM role)
#      used by setup-mlops-infra.sh in the e2e-mlops-pipeline setup-infra job
#
# Usage:
#   ./scripts/setup-github-oidc.sh <env> <deployment-role-name> <account-id> <github-repo>
#
# Examples:
#   ./scripts/setup-github-oidc.sh dev dataops-admin 476621446285 myorg/my-repo
#   ./scripts/setup-github-oidc.sh test dataops-admin 997712747839 myorg/my-repo
#   ./scripts/setup-github-oidc.sh prod dataops-admin 973414239152 myorg/my-repo
# =============================================================================

set -euo pipefail
export AWS_PAGER=""

# ── Configuration ────────────────────────────────────────────────────────────
ENV="${1:?Usage: $0 <dev|test|prod> <deployment-role-name> <account-id> <github-repo>}"
DEPLOYMENT_ROLE="${2:?Usage: $0 <dev|test|prod> <deployment-role-name> <account-id> <github-repo>}"
ACCOUNT_ID="${3:?Usage: $0 <dev|test|prod> <deployment-role-name> <account-id> <github-repo>}"
GITHUB_REPO="${4:?Usage: $0 <dev|test|prod> <deployment-role-name> <account-id> <github-repo>}"
ROLE_NAME="GitHubActionsRole-${ENV}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
DEPLOYMENT_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${DEPLOYMENT_ROLE}"

echo "=== Setting up GitHub OIDC for $ENV ==="
echo "  Account:      $ACCOUNT_ID"
echo "  OIDC Role:    $ROLE_NAME"
echo "  Deployment Role: $DEPLOYMENT_ROLE"

# ── Verify account ───────────────────────────────────────────────────────────
CURRENT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
if [ "$CURRENT_ACCOUNT" != "$ACCOUNT_ID" ]; then
  echo "ERROR: Authenticated to $CURRENT_ACCOUNT, expected $ACCOUNT_ID"
  exit 1
fi

# ── Step 1: OIDC provider ───────────────────────────────────────────────────
echo "Step 1: OIDC provider..."
OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" &>/dev/null; then
  echo "  Already exists."
else
  aws iam create-open-id-connect-provider \
    --url https://token.actions.githubusercontent.com \
    --client-id-list sts.amazonaws.com \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
  echo "  Created."
fi

# ── Step 2: IAM role with OIDC trust ────────────────────────────────────────
echo "Step 2: IAM role..."
# Repos created after 2026-07-15 (and any repo that opted in) send an immutable
# subject: repo:OWNER@OWNER_ID/REPO@REPO_ID:environment:ENV. The name-only
# subject does not match that token, and STS then denies AssumeRoleWithWebIdentity.
SUB_CLAIMS=("repo:${GITHUB_REPO}:environment:${ENV}")
if command -v gh >/dev/null 2>&1; then
  IMMUTABLE_PREFIX=$(gh api "repos/${GITHUB_REPO}/actions/oidc/customization/sub" \
    --jq 'if .use_immutable_subject == true then .sub_claim_prefix else empty end' 2>/dev/null || true)
  if [ -n "${IMMUTABLE_PREFIX}" ]; then
    SUB_CLAIMS+=("${IMMUTABLE_PREFIX}:environment:${ENV}")
    echo "  Immutable OIDC subject enabled: ${IMMUTABLE_PREFIX}:environment:${ENV}"
  fi
fi
SUB_JSON=$(printf '%s\n' "${SUB_CLAIMS[@]}" | jq -R . | jq -s .)
TRUST_POLICY=$(jq -n \
  --arg oidc "$OIDC_ARN" \
  --argjson subs "$SUB_JSON" \
  '{
    Version: "2012-10-17",
    Statement: [{
      Effect: "Allow",
      Principal: { Federated: $oidc },
      Action: "sts:AssumeRoleWithWebIdentity",
      Condition: {
        StringEquals: { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
        StringLike: { "token.actions.githubusercontent.com:sub": $subs }
      }
    }]
  }')

if aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
  aws iam update-assume-role-policy --role-name "$ROLE_NAME" --policy-document "$TRUST_POLICY"
  echo "  Updated trust policy."
else
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST_POLICY" \
    --description "GitHub Actions OIDC role for smus-cli ($ENV)"
  echo "  Created."
fi

# ── Step 3: Inline policy (AssumeRole only) ─────────────────────────────────
echo "Step 3: Inline policy..."
# Include sts:TagSession so role-chaining via configure-aws-credentials works
# (that action tags sessions unless role-skip-session-tagging is set).
ASSUME_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "AssumeProjectRole",
    "Effect": "Allow",
    "Action": ["sts:AssumeRole", "sts:TagSession"],
    "Resource": [
      "arn:aws:iam::${ACCOUNT_ID}:role/datazone*",
      "arn:aws:iam::${ACCOUNT_ID}:role/${DEPLOYMENT_ROLE}",
      "arn:aws:iam::${ACCOUNT_ID}:role/service-role/${DEPLOYMENT_ROLE}"
    ]
  }]
}
EOF
)

aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "AssumeProjectRole" \
  --policy-document "$ASSUME_POLICY"
echo "  Applied."

# ── Step 3b: Inline policy for setup-mlops-infra.sh (EventBridge + Lambda) ───
# The setup-infra CI job assumes this OIDC role directly (no secondary hop) and
# runs setup-mlops-infra.sh, which creates the deploy-trigger Lambda, its
# execution role, and the EventBridge rule. Scope is limited to the stack-*
# resources that script provisions.
echo "Step 3b: MLOps infra setup policy..."
MLOPS_INFRA_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "LambdaDeployTrigger",
      "Effect": "Allow",
      "Action": [
        "lambda:CreateFunction",
        "lambda:GetFunction",
        "lambda:UpdateFunctionCode",
        "lambda:UpdateFunctionConfiguration",
        "lambda:AddPermission",
        "lambda:RemovePermission"
      ],
      "Resource": "arn:aws:lambda:*:${ACCOUNT_ID}:function:stack-airflow-deploy-trigger"
    },
    {
      "Sid": "LambdaExecutionRole",
      "Effect": "Allow",
      "Action": [
        "iam:GetRole",
        "iam:CreateRole",
        "iam:PutRolePolicy",
        "iam:PassRole"
      ],
      "Resource": "arn:aws:iam::${ACCOUNT_ID}:role/stack-airflow-deploy-trigger-role"
    },
    {
      "Sid": "EventBridgeDeployTrigger",
      "Effect": "Allow",
      "Action": [
        "events:PutRule",
        "events:PutTargets",
        "events:DescribeRule"
      ],
      "Resource": "arn:aws:events:*:${ACCOUNT_ID}:rule/stack-model-approved"
    }
  ]
}
EOF
)

aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "MlopsInfraSetup" \
  --policy-document "$MLOPS_INFRA_POLICY"
echo "  Applied."

# ── Step 4: Trust this OIDC role on deployment + optional DataZone user roles ─
# Batch inference / role-chaining needs the target role's trust policy to allow
# GitHubActionsRole-*. setup only updated DEPLOYMENT_ROLE before; DataZone
# project user roles (datazone_usr_role_*) must trust OIDC too.
echo "Step 4: Updating trust policies to allow $ROLE_NAME..."

# Wait for IAM role propagation before referencing it in another policy
echo "  Waiting for IAM role propagation..."
sleep 10

add_oidc_trust() {
  local target_role="$1"
  if ! aws iam get-role --role-name "$target_role" &>/dev/null; then
    echo "  WARNING: role $target_role not found — skip"
    return 0
  fi
  local existing updated
  existing=$(aws iam get-role --role-name "$target_role" \
    --query 'Role.AssumeRolePolicyDocument' --output json)
  if echo "$existing" | grep -q "$ROLE_NAME"; then
    echo "  $target_role already trusts $ROLE_NAME"
    return 0
  fi
  updated=$(echo "$existing" | jq \
    --arg arn "$ROLE_ARN" \
    '.Statement += [{"Effect":"Allow","Principal":{"AWS":$arn},"Action":["sts:AssumeRole","sts:TagSession"]}]')
  aws iam update-assume-role-policy \
    --role-name "$target_role" \
    --policy-document "$updated"
  echo "  Added $ROLE_NAME as trusted principal on $target_role"
}

add_oidc_trust "$DEPLOYMENT_ROLE"

# Optional: comma-separated DataZone user role names (or leave unset to discover).
# Example:
#   DATAZONE_USER_ROLES=datazone_usr_role_xxx_yyy ./setup-github-oidc.sh ...
if [[ -n "${DATAZONE_USER_ROLES:-}" ]]; then
  IFS=',' read -ra DZ_ROLES <<< "$DATAZONE_USER_ROLES"
  for dz in "${DZ_ROLES[@]}"; do
    dz_trimmed="$(echo "$dz" | xargs)"
    [[ -n "$dz_trimmed" ]] && add_oidc_trust "$dz_trimmed"
  done
else
  echo "  Discovering datazone_usr_role_* roles in account..."
  mapfile -t DZ_DISCOVERED < <(aws iam list-roles \
    --query "Roles[?starts_with(RoleName, 'datazone_usr_role_')].RoleName" \
    --output text | tr '\t' '\n' | sed '/^$/d')
  if ((${#DZ_DISCOVERED[@]} == 0)); then
    echo "  No datazone_usr_role_* found. Set DATAZONE_USER_ROLES if batch role-chaining is needed."
  else
    for dz in "${DZ_DISCOVERED[@]}"; do
      add_oidc_trust "$dz"
    done
  fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Done ==="
echo "  OIDC Role:    $ROLE_ARN"
echo "  Deployment Role: $DEPLOYMENT_ROLE_ARN"
echo ""
echo "Next steps:"
echo "  gh variable set AWS_ROLE_ARN --env $ENV --body \"$ROLE_ARN\""
echo "  gh variable set AWS_ACCOUNT_ID --env $ENV --body \"$ACCOUNT_ID\""
echo "  gh variable set DEPLOYMENT_ROLE_NAME --env $ENV --body \"$DEPLOYMENT_ROLE\""
