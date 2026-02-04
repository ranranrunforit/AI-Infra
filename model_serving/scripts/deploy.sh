#!/bin/bash
###############################################################################
# Deployment Script for High-Performance Model Serving
#
# Automates deployment to Kubernetes cluster with:
# - Docker image building and pushing
# - Kubernetes resource deployment
# - Health check validation
# - Rollback on failure
#
# Usage: ./scripts/deploy.sh [--environment prod|staging] [--dry-run]
###############################################################################

set -e

# Configuration
DOCKER_REGISTRY="${DOCKER_REGISTRY:-registry.example.com}"
PROJECT_NAME="model-serving"
ENVIRONMENT="${1:-staging}"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment|-e)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "Deploying to $ENVIRONMENT environment..."

# Build Docker image
echo "Building Docker image..."
docker build -t ${PROJECT_NAME}:latest -f docker/Dockerfile .
docker tag ${PROJECT_NAME}:latest ${DOCKER_REGISTRY}/${PROJECT_NAME}:${ENVIRONMENT}

# Push to registry
if [[ "$DRY_RUN" == false ]]; then
    echo "Pushing to registry..."
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}:${ENVIRONMENT}
fi

# Deploy to Kubernetes
echo "Deploying to Kubernetes..."
kubectl apply -f kubernetes/${ENVIRONMENT}/ ${DRY_RUN:+--dry-run=client}

# Wait for rollout
kubectl rollout status deployment/${PROJECT_NAME} -n ${ENVIRONMENT}

# Health check
echo "Performing health check..."
sleep 10
kubectl port-forward service/${PROJECT_NAME} 8000:8000 &
PF_PID=$!
sleep 5

if curl -f http://localhost:8000/health; then
    echo "Deployment successful!"
    kill $PF_PID
else
    echo "Health check failed, rolling back..."
    kubectl rollout undo deployment/${PROJECT_NAME} -n ${ENVIRONMENT}
    kill $PF_PID
    exit 1
fi

echo "Deployment complete!"
