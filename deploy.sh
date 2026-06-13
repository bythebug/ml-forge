#!/usr/bin/env bash
# deploy.sh — Build, push, and deploy ml-forge to AWS ECS
#
# Prerequisites:
#   aws-cli v2 configured (aws configure)
#   docker installed and running
#   jq installed (brew install jq)
#
# Usage:
#   ./deploy.sh                        # deploy to prod
#   ENVIRONMENT=staging ./deploy.sh    # deploy to staging
#   DRY_RUN=1 ./deploy.sh              # validate without deploying

set -euo pipefail

# ── configuration ─────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ENVIRONMENT="${ENVIRONMENT:-prod}"
APP_NAME="ml-forge"
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${APP_NAME}"
ECS_CLUSTER="${APP_NAME}-${ENVIRONMENT}"
ECS_SERVICE="${APP_NAME}-api-${ENVIRONMENT}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
DRY_RUN="${DRY_RUN:-0}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ── validate prerequisites ────────────────────────────────────────────────────
log "Validating prerequisites..."
command -v aws   >/dev/null || die "aws-cli not found"
command -v docker >/dev/null || die "docker not found"
command -v jq    >/dev/null || die "jq not found"

[[ -f Dockerfile ]] || die "Dockerfile not found — run from project root"

log "Deploying ${APP_NAME}:${IMAGE_TAG} → ${ENVIRONMENT} (region=${AWS_REGION})"

if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1 — validation passed, skipping deploy"
    exit 0
fi

# ── Step 1: Authenticate Docker with ECR ─────────────────────────────────────
log "Step 1/5  Authenticating with ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$ECR_REPO"

# ── Step 2: Build Docker image ────────────────────────────────────────────────
log "Step 2/5  Building Docker image..."
docker build \
    --tag "${APP_NAME}:${IMAGE_TAG}" \
    --tag "${APP_NAME}:latest" \
    --platform linux/amd64 \
    .

# ── Step 3: Push to ECR ───────────────────────────────────────────────────────
log "Step 3/5  Pushing to ECR..."
docker tag "${APP_NAME}:${IMAGE_TAG}" "${ECR_REPO}:${IMAGE_TAG}"
docker tag "${APP_NAME}:latest"        "${ECR_REPO}:latest"
docker push "${ECR_REPO}:${IMAGE_TAG}"
docker push "${ECR_REPO}:latest"
log "  pushed ${ECR_REPO}:${IMAGE_TAG}"

# ── Step 4: Update ECS task definition ───────────────────────────────────────
log "Step 4/5  Updating ECS task definition..."
CURRENT_TASK_DEF=$(aws ecs describe-task-definition \
    --task-definition "${APP_NAME}-${ENVIRONMENT}" \
    --region "$AWS_REGION" \
    --query 'taskDefinition' \
    --output json)

NEW_TASK_DEF=$(echo "$CURRENT_TASK_DEF" \
    | jq --arg IMAGE "${ECR_REPO}:${IMAGE_TAG}" \
         '.containerDefinitions[0].image = $IMAGE
          | del(.taskDefinitionArn, .revision, .status,
                .requiresAttributes, .compatibilities,
                .registeredAt, .registeredBy)')

NEW_TASK_ARN=$(aws ecs register-task-definition \
    --cli-input-json "$NEW_TASK_DEF" \
    --region "$AWS_REGION" \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)

log "  registered task: $NEW_TASK_ARN"

# ── Step 5: Deploy to ECS service ─────────────────────────────────────────────
log "Step 5/5  Deploying to ECS service (rolling update)..."
aws ecs update-service \
    --cluster "$ECS_CLUSTER" \
    --service "$ECS_SERVICE" \
    --task-definition "$NEW_TASK_ARN" \
    --force-new-deployment \
    --region "$AWS_REGION" \
    --output json | jq -r '.service.deployments[0] | "  status=\(.status) desired=\(.desiredCount) running=\(.runningCount)"'

log "Waiting for service to stabilise (timeout 5m)..."
aws ecs wait services-stable \
    --cluster "$ECS_CLUSTER" \
    --services "$ECS_SERVICE" \
    --region "$AWS_REGION"

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Deploy complete: ${APP_NAME}:${IMAGE_TAG} → ${ENVIRONMENT}"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
