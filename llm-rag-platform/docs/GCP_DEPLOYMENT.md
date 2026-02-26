# GCP Deployment Guide
# Project 303: Enterprise LLM Platform with RAG

## Prerequisites

```bash
# Install required tools
# 1. Google Cloud CLI: https://cloud.google.com/sdk/docs/install
gcloud --version

# 2. Terraform >= 1.5.0: https://developer.hashicorp.com/terraform/downloads
terraform --version

# 3. Docker: https://docs.docker.com/get-docker/
docker --version
```

## Step 1: GCP Project Setup

```bash
# Create a new project (or use existing)
gcloud projects create my-llm-rag-platform --name="LLM RAG Platform"
gcloud config set project my-llm-rag-platform

# Enable billing (required for Cloud Run, Artifact Registry)
# Do this in the GCP Console: https://console.cloud.google.com/billing

# Authenticate
gcloud auth login
gcloud auth application-default login
```

## Step 2: Get Your Gemini API Key

1. Go to https://aistudio.google.com/app/apikey
2. Click **Create API Key** → Select your project
3. **With Google Pro account**, you get access to:
   - `gemini-2.0-flash` (fast, recommended)
   - `gemini-2.0-pro-exp-02-05` (most capable)
   - `gemini-1.5-pro` (stable previous gen)

> **Note on Gemini 3.0**: As of February 2026, Google has not released a public
> "Gemini 3.0". The latest models are in the Gemini 2.0 family. Gemini 2.0 Pro
> Experimental (`gemini-2.0-pro-exp-02-05`) is the most capable available model.

## Step 3: Deploy Infrastructure with Terraform

```bash
cd reference-implementation/terraform/gcp

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
project_id    = "my-llm-rag-platform"
region        = "us-central1"
environment   = "production"
gemini_model  = "gemini-2.0-flash"
llm_backend   = "gemini"

# Optional: Enable Qdrant VM (~$30/month for e2-medium)
enable_qdrant_vm = false

# Optional: Enable vLLM Spot VM (Mistral 7B on L4 GPU, ~$100/month spot)
enable_vllm_vm = false
EOF

# Initialize and apply
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

## Step 4: Store API Keys in Secret Manager

```bash
# Store Gemini API key
echo -n "YOUR_GEMINI_API_KEY" | \
  gcloud secrets versions add rag-google-api-key --data-file=-

# Store service API key (for protecting your RAG API)
echo -n "$(openssl rand -base64 32)" | \
  gcloud secrets versions add rag-service-api-key --data-file=-
```

## Step 5: Build and Push Docker Image

```bash
# Authenticate Docker with Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build image (from reference-implementation/python/)
cd reference-implementation/python
docker build -t rag-api:latest .

# Tag and push
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1

docker tag rag-api:latest \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/rag-platform/rag-api:latest

docker push \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/rag-platform/rag-api:latest
```

## Step 6: Deploy to Cloud Run

After pushing the image, update Cloud Run to use it:

```bash
gcloud run services update rag-api \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/rag-platform/rag-api:latest \
  --region=${REGION}

# Get the service URL
gcloud run services describe rag-api --region=${REGION} \
  --format="value(status.url)"
```

## Step 7: Test the Deployment

```bash
SERVICE_URL=$(gcloud run services describe rag-api --region=us-central1 --format="value(status.url)")
API_KEY=$(gcloud secrets versions access latest --secret=rag-service-api-key)

# Health check
curl ${SERVICE_URL}/health

# Expected: {"status":"ok","services":{"qdrant":"ok","llm":"ok (gemini)"},...}

# Index a test document
curl -X POST ${SERVICE_URL}/v1/documents \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "documents": [{
      "id": "test-doc-1",
      "text": "Our refund policy allows returns within 30 days of purchase.",
      "metadata": {"source": "Policy Manual", "version": "2.1"}
    }]
  }'

# Query the RAG system
curl -X POST ${SERVICE_URL}/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "query": "What is the refund policy?",
    "user_id": "test-user"
  }'
```

## Step 8 (Optional): Enable vLLM Self-Hosted Backend

If you want to use vLLM (Llama 3, Mistral) instead of or alongside Gemini:

```bash
# Enable the vLLM Spot VM in Terraform
cat >> terraform.tfvars <<EOF
enable_vllm_vm = true
vllm_model     = "mistralai/Mistral-7B-Instruct-v0.3"
EOF

terraform apply

# After VM starts (~5 minutes for model download), get its internal IP
VLLM_IP=$(terraform output -raw qdrant_internal_ip)

# Update Cloud Run to use vLLM
gcloud run services update rag-api \
  --region=${REGION} \
  --set-env-vars="LLM_BACKEND=vllm,VLLM_ENDPOINT=http://${VLLM_IP}:8000/v1"
```

## Cost Summary

| Component | Setup | Monthly Cost |
|-----------|-------|-------------|
| Cloud Run (RAG API) | Free tier | $0 (2M req/month free) |
| Gemini API | Google Pro account | Included in Pro |
| GCS (documents) | Always free | $0 (5GB free) |
| Qdrant in-memory | In Cloud Run | $0 |
| Qdrant VM (optional) | e2-medium | ~$30/month |
| vLLM Spot VM, L4 GPU (optional) | g2-standard-4 | ~$100/month |
| **Total (minimal)** | | **~$0/month** |
| **Total (with vLLM)** | | **~$130/month** |

## Google Colab Alternative (Free Tier)

For development and testing without any cloud costs, see:
- `reference-implementation/colab_quickstart.ipynb`

The Colab notebook runs the complete RAG pipeline:
- In-memory Qdrant (no server needed)
- HuggingFace bge-small embeddings (CPU, <500MB)  
- Gemini Pro for generation
- Safety guardrails

## Troubleshooting

### Cloud Run 503 on startup
Model download takes ~60 seconds. The startup probe has a 30s delay + 10 retries.
Check logs: `gcloud run services logs tail rag-api --region=us-central1`

### Gemini API quota errors
- Free tier: 15 requests/minute, 1M tokens/day
- Google Pro: Higher limits, check your quota in GCP Console
- Switch to `gemini-2.0-flash` (higher quota than Pro models)

### Qdrant connection errors (in-memory mode)
When Qdrant host is `:memory:`, documents are lost on restart.
Enable `enable_qdrant_vm=true` for persistent storage.

### vLLM Spot VM preempted
Spot VMs can be preempted by GCP. Add a startup script restart policy or use
on-demand if preemption is too frequent for your workload.
