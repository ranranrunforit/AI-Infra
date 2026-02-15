# Deployment Guide

This guide covers the deployment of the Multi-Region ML Platform using Terraform and Kubernetes (kubectl/helm).

## Prerequisites

1.  **Cloud Provider Accounts**: AWS, GCP, Azure subscriptions.
2.  **Domain Name**: A domain managed by AWS Route53 (e.g., `example.com`).
3.  **Tools**:
    *   Terraform >= 1.5.0
    *   kubectl
    *   AWS CLI, gcloud CLI, Azure CLI
    *   jq (for JSON processing)

## Phase 1: Infrastructure Provisioning (Terraform)

We use Terraform to provision the physical infrastructure (Clusters, Networks, Storage).

### 1. Initialize
Navigate to the terraform directory and initialize modules.

```bash
cd terraform
terraform init
```

### 2. Configure Variables
Create a `terraform.tfvars` file.

```hcl
project_name = "ml-platform-prod"
environment  = "prod"

# Cloud Configs
aws_region     = "us-west-2"
gcp_project_id = "my-gcp-project-id"
gcp_region     = "europe-west1"
azure_location = "centralindia"

# DNS
domain_name = "ml.example.com"
```

### 3. Plan and Apply

```bash
terraform plan -out=tfplan
terraform apply tfplan
```
*Note: This process may take 20-30 minutes as it spins up managed Kubernetes clusters.*

### 4. Output Retrieval
Save important outputs for the application phase.

```bash
terraform output -json > cloud_outputs.json
```

## Phase 2: Kubernetes Configuration

Connect `kubectl` to your new clusters.

### 1. Configure Contexts

**AWS EKS:**
```bash
aws eks update-kubeconfig --region us-west-2 --name ml-platform-prod-eks --alias aws-cluster
```

**GCP GKE:**
```bash
gcloud container clusters get-credentials ml-platform-prod-gke --region europe-west1 --alias gcp-cluster
```

**Azure AKS:**
```bash
az aks get-credentials --resource-group ml-platform-prod-rg --name ml-platform-prod-aks --alias azure-cluster
```

### 2. Deploy Base Resources (All Clusters)

Loop through all contexts to deploy common resources.

```bash
for ctx in aws-cluster gcp-cluster azure-cluster; do
    echo "Deploying to $ctx..."
    kubectl config use-context $ctx
    
    # Create Namespaces
    kubectl apply -f kubernetes/base/namespace.yaml
    
    # Deploy ConfigMaps and Secrets
    kubectl apply -f kubernetes/base/configmap.yaml
    kubectl apply -f kubernetes/base/secrets.yaml # Ensure this file is generated safely
    
    # Deploy Prometheus Operator (optional if using cloud managed)
    kubectl apply -f kubernetes/base/monitoring/
done
```

## Phase 3: Application Deployment

### 1. Build & Push Images
Build the main application image.

```bash
docker build -t gcr.io/my-project/ml-platform:v1 .
docker push gcr.io/my-project/ml-platform:v1
```

### 2. Deploy Regional Services

**AWS (Primary):**
```bash
kubectl config use-context aws-cluster
kubectl apply -k kubernetes/overlays/us-west-2/
```

**GCP (Secondary):**
```bash
kubectl config use-context gcp-cluster
kubectl apply -k kubernetes/overlays/eu-west-1/
```

**Azure (Secondary):**
```bash
kubectl config use-context azure-cluster
kubectl apply -k kubernetes/overlays/ap-south-1/
```

## Phase 4: DNS & Failover Setup

1.  **Verify Route53**: Go to AWS Console -> Route53 to ensure records created by Terraform exist.
2.  **Start Controller**: The `FailoverController` running in the primary cluster will now start updating health checks.

## Verification

Run the health check script to confirm global availability.

```bash
./scripts/health_check.sh
```

Expected Output:
```text
[AWS] us-west-2: HEALTHY (200 OK) - Latency: 45ms
[GCP] eu-west-1: HEALTHY (200 OK) - Latency: 120ms
[AZURE] ap-south-1: HEALTHY (200 OK) - Latency: 250ms
Global Endpoint (ml.example.com): Resolves to AWS (Primary)
```
