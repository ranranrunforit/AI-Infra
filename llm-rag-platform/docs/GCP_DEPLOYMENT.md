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
# Link your billing account to project "my-llm-rag-platform"
# If your project isn't linked, click Manage Billing Accounts, find your project in the list, click the three-dot menu next to it, and select Change billing. Choose your billing account from the drop-down.

# Step A — Authenticate your gcloud CLI
gcloud auth login

# Step B — Set up Application Default Credentials (ADC) for Terraform
# Cloud Shell runs on a GCE VM, so credentials are saved to a /tmp directory,
# NOT to ~/.config/gcloud/. Run the login, then copy the file to the standard location.
gcloud auth application-default login
# When prompted "You are running on a GCE VM... continue?" → type y
# Complete the browser sign-in and paste the verification code back here.

# After login completes, copy credentials to the standard location Terraform expects:
mkdir -p ~/.config/gcloud
cp /tmp/tmp.*/application_default_credentials.json \
  ~/.config/gcloud/application_default_credentials.json

# Unset this env var if you set it manually in a previous attempt (it overrides ADC)
unset GOOGLE_APPLICATION_CREDENTIALS

# Confirm the credentials file exists
cat ~/.config/gcloud/application_default_credentials.json | python3 -m json.tool | head -3
```

## Step 2: Get Your Gemini API Key

1. Go to https://aistudio.google.com/app/apikey
2. Click **Create API Key** → Select your project


## Step 3: Deploy Infrastructure with Terraform

> **Before running Terraform**: Make sure you completed the ADC credential setup in Step 1 (the `cp /tmp/tmp.*` command). Terraform will fail with auth errors if `~/.config/gcloud/application_default_credentials.json` doesn't exist.

> **Order note**: The first `terraform apply` creates everything **except** Cloud Run (which needs a Docker image). That's expected — complete Steps 4–5 first, then re-run `terraform apply` in Step 6.

```bash
cd ~/reference-implementation/terraform/gcp

# Create terraform.tfvars — run this EXACTLY (don't mix with other commands)
cat > terraform.tfvars <<'EOF'
project_id    = "my-llm-rag-platform"
region        = "us-central1"
environment   = "production"
gemini_model  = "gemini-3.0-flash"
llm_backend   = "gemini"

# Enable Qdrant VM for persistent vector storage (~$30/month for e2-medium)
enable_qdrant_vm = true

# Optional: Enable vLLM Spot VM (Mistral 7B on L4 GPU, ~$100/month spot)
enable_vllm_vm = false
EOF

# Verify — should show ONLY key=value lines, no shell commands
cat terraform.tfvars

# Initialize and apply
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

### Step 3b: Set Up Qdrant VM (if enabled)

The Qdrant VM's startup script may fail due to line-ending issues. **Always verify and manually install if needed:**

```bash
# Wait ~2 min for VM to boot, then check if Docker is running
gcloud compute ssh qdrant-vector-db --zone=us-central1-a -- "sudo docker ps"

# If you see "docker: command not found", install manually:
gcloud compute ssh qdrant-vector-db --zone=us-central1-a

# Inside the VM, run:
sudo apt-get update -y
sudo apt-get install -y docker.io
sudo mkdir -p /data/qdrant
sudo docker run -d --name qdrant --restart unless-stopped \
  -p 6333:6333 -p 6334:6334 \
  -v /data/qdrant:/qdrant/storage qdrant/qdrant:v1.9.0

# Verify Qdrant is running
sudo docker ps
curl -s http://localhost:6333/collections
exit
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

Use **Cloud Build** to build and push (recommended — avoids Docker networking issues in Cloud Shell):

```bash
cd ~/reference-implementation/python

# Build and push via Cloud Build (builds on GCP servers, no local Docker push needed)
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/my-llm-rag-platform/rag-platform/rag-api:latest .

# If prompted to enable Cloud Build API, type 'y'
# If you get PERMISSION_DENIED, grant yourself the required roles:
#   gcloud projects add-iam-policy-binding my-llm-rag-platform \
#     --member="user:YOUR_EMAIL" --role="roles/cloudbuild.builds.editor"
#   gcloud projects add-iam-policy-binding my-llm-rag-platform \
#     --member="user:YOUR_EMAIL" --role="roles/storage.admin"
#   sleep 30 && retry the gcloud builds submit command

# Build takes ~20 minutes (downloads PyTorch + embedding models)
# Verify the image is in Artifact Registry:
gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/my-llm-rag-platform/rag-platform
```

## Step 6: Deploy Cloud Run

Now that the Docker image is pushed, re-run Terraform to deploy Cloud Run:

```bash
cd ~/reference-implementation/terraform/gcp
terraform plan -out=tfplan && terraform apply tfplan

# Get the service URL
gcloud run services describe rag-api --region=us-central1 \
  --format="value(status.url)"
```

> **Note**: If Cloud Run fails with startup probe errors, check the logs:
> `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rag-api" --limit=15 --format="value(textPayload)"`
> Common issues: Qdrant VM not ready yet (wait and retry), or missing secrets (complete Step 4).

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

---

## Cleanup: Stop All Charges Completely

Follow these steps **in order** to ensure nothing keeps billing.

### Option A: Destroy only the Qdrant VM (keep the rest running)

```bash
# From reference-implementation/terraform/gcp/
cd reference-implementation/terraform/gcp

# Disable the Qdrant VM and re-apply
cat >> terraform.tfvars <<'EOF'
enable_qdrant_vm = false
EOF

terraform apply
```

> This stops the VM and its ~$30/month charge. Cloud Run and GCS remain (free tier).

### Option B: Destroy ALL infrastructure (Terraform-managed resources)

```bash
# From reference-implementation/terraform/gcp/
cd reference-implementation/terraform/gcp

# Tear down everything Terraform created (Cloud Run, VM, Artifact Registry, etc.)
terraform destroy
```

Type `yes` when prompted. This deletes all VMs, Cloud Run services, and Artifact Registry images.

### Option C: Delete the entire GCP project (hardest stop — removes everything)

```bash
# This permanently deletes the project and ALL resources inside it.
# Billing stops completely after deletion (usually within minutes).
gcloud projects delete my-llm-rag-platform
```

You will be asked to confirm by typing the project ID. After deletion:
- All VMs, Cloud Run services, storage buckets, secrets, and images are gone.
- The project ID `my-llm-rag-platform` cannot be reused for 30 days.
- Go to https://console.cloud.google.com/billing to confirm $0 charges.

> **Tip**: If you only want a temporary pause, use Option A or B. Use Option C only if you are done with the project entirely.
