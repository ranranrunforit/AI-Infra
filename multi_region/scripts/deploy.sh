#!/bin/bash
# Multi-Region ML Platform Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Multi-Region ML Platform Deployment ===${NC}"

# Check prerequisites
echo "Checking prerequisites..."
command -v terraform >/dev/null 2>&1 || { echo -e "${RED}terraform is required but not installed${NC}"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo -e "${RED}kubectl is required but not installed${NC}"; exit 1; }
command -v kustomize >/dev/null 2>&1 || { echo -e "${RED}kustomize is required but not installed${NC}"; exit 1; }

# Parse arguments
ENVIRONMENT=${1:-dev}
SKIP_TERRAFORM=${2:-false}

echo "Environment: $ENVIRONMENT"
echo ""

# Deploy infrastructure with Terraform
if [ "$SKIP_TERRAFORM" != "true" ]; then
    echo -e "${YELLOW}Step 1: Deploying infrastructure with Terraform${NC}"
    cd terraform
    terraform init
    terraform workspace select $ENVIRONMENT || terraform workspace new $ENVIRONMENT
    terraform plan -out=tfplan
    terraform apply tfplan
    cd ..
    echo -e "${GREEN}Infrastructure deployed${NC}\n"
else
    echo -e "${YELLOW}Skipping Terraform deployment${NC}\n"
fi

# Get cluster credentials
echo -e "${YELLOW}Step 2: Configuring kubectl${NC}"

# AWS
echo "Configuring AWS EKS..."
aws eks update-kubeconfig --region us-west-2 --name ml-platform-us-west-2

# GCP
echo "Configuring GCP GKE..."
gcloud container clusters get-credentials ml-platform-eu-west-1 --region europe-west1

# Azure
echo "Configuring Azure AKS..."
az aks get-credentials --resource-group ml-platform-ap-south-1-rg --name ml-platform-ap-south-1

echo -e "${GREEN}Kubectl configured${NC}\n"

# Deploy Kubernetes resources
echo -e "${YELLOW}Step 3: Deploying Kubernetes resources${NC}"

REGIONS=("us-west-2" "eu-west-1" "ap-south-1")

for region in "${REGIONS[@]}"; do
    echo "Deploying to $region..."
    kubectl config use-context $region
    kubectl apply -k kubernetes/overlays/$region
    echo "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/ml-serving -n ml-platform
    echo -e "${GREEN}$region deployed${NC}"
done

echo ""

# Verify deployments
echo -e "${YELLOW}Step 4: Verifying deployments${NC}"

for region in "${REGIONS[@]}"; do
    echo "Checking $region..."
    kubectl config use-context $region
    kubectl get pods -n ml-platform
    kubectl get svc -n ml-platform
done

echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Verify DNS records: ./scripts/verify_dns.sh"
echo "2. Run health checks: ./scripts/health_check.sh"
echo "3. Check monitoring: ./scripts/check_monitoring.sh"
