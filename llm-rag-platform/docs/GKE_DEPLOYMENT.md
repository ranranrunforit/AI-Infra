# GKE (Kubernetes) Deployment Guide
# Project 13: Enterprise LLM Platform with RAG

Running the full RAG platform on **Google Kubernetes Engine (GKE)** gives you
production-grade autoscaling, rolling updates, and centralized observability —
the same architecture as the original EKS design, but on GCP.

---

## Architecture on GKE

```
Internet ──► GKE Ingress (L7 Load Balancer + TLS)
                    │
              ┌─────▼──────────────────┐  (llm-inference namespace)
              │   rag-api (2+ pods)    │───► Gemini Pro API
              │   HPA: 2–10 replicas   │───► vLLM VM (optional)
              └─────────────────────────┘
                    │
              ┌─────▼──────────────────┐
              │   qdrant (1 pod)       │  PVC: 20Gi SSD
              └─────────────────────────┘
```

---

## Prerequisites

```bash
# Install tools
gcloud --version      # Google Cloud CLI
kubectl version       # Kubernetes CLI (comes with gcloud)
docker --version      # Docker

# Authenticate
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

---

## Step 1: Create the GKE Cluster

```bash
PROJECT_ID=your-project-id
REGION=us-central1

# Enable required APIs
gcloud services enable container.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com

# Create GKE Autopilot cluster (recommended - Google manages nodes)
# Autopilot is pay-per-pod, not pay-per-node; cheapest for variable workloads
gcloud container clusters create-auto rag-platform-cluster \
    --region=$REGION \
    --project=$PROJECT_ID

# OR: Standard cluster (more control, you manage nodes)
# gcloud container clusters create rag-platform-cluster \
#     --region=$REGION \
#     --machine-type=e2-standard-4 \
#     --num-nodes=2 \
#     --enable-autoscaling --min-nodes=1 --max-nodes=5 \
#     --workload-pool=${PROJECT_ID}.svc.id.goog

# Get credentials (sets up kubectl)
gcloud container clusters get-credentials rag-platform-cluster \
    --region=$REGION

# Verify
kubectl get nodes
```

---

## Step 2: Set Up Artifact Registry (Docker Images)

```bash
# Create registry
gcloud artifacts repositories create rag-platform \
    --repository-format=docker \
    --location=$REGION \
    --description="Project 303 RAG Platform images"

# Authenticate Docker
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build and push the RAG API image
cd reference-implementation/python
docker build -t rag-api:latest .

docker tag rag-api:latest \
    ${REGION}-docker.pkg.dev/${PROJECT_ID}/rag-platform/rag-api:latest

docker push \
    ${REGION}-docker.pkg.dev/${PROJECT_ID}/rag-platform/rag-api:latest
```

---

## Step 3: Update Image Reference in Deployment

Edit `kubernetes/gcp/rag-service/deployment.yaml` and replace:

```yaml
image: us-central1-docker.pkg.dev/YOUR_PROJECT_ID/rag-platform/rag-api:latest
```

with your actual project ID:

```bash
sed -i "s/YOUR_PROJECT_ID/${PROJECT_ID}/g" \
    kubernetes/gcp/rag-service/deployment.yaml
```

---

## Step 4: Create the Namespace and Secrets

```bash
# Create namespace
kubectl create namespace llm-inference

# Store your Gemini API key (get from https://aistudio.google.com/app/apikey)
kubectl create secret generic rag-api-secrets \
    --from-literal=GOOGLE_API_KEY=YOUR_GEMINI_API_KEY \
    --from-literal=API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))") \
    -n llm-inference

# Verify
kubectl get secret rag-api-secrets -n llm-inference
```

---

## Step 5: Deploy Qdrant (Vector Database)

```bash
kubectl apply -f kubernetes/gcp/qdrant/deployment.yaml

# Wait for Qdrant to be ready
kubectl rollout status deployment/qdrant -n llm-inference

# Test Qdrant is healthy
kubectl port-forward svc/qdrant-service 6333:6333 -n llm-inference &
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vector search engine"}
kill %1
```

---

## Step 6: Deploy the RAG API

```bash
kubectl apply -f kubernetes/gcp/rag-service/deployment.yaml

# Watch pods start up (model loading takes ~60s)
kubectl get pods -n llm-inference -w

# Check logs
kubectl logs -l app=rag-api -n llm-inference --follow

# Verify health (port-forward for testing)
kubectl port-forward svc/rag-api-service 8080:80 -n llm-inference &
curl http://localhost:8080/health
# Expected: {"status":"ok","services":{"qdrant":"ok","llm":"ok (gemini)"},...}
kill %1
```

---

## Step 7: Test the RAG API

```bash
# Index a document
kubectl port-forward svc/rag-api-service 8080:80 -n llm-inference &

RAG_API_KEY=$(kubectl get secret rag-api-secrets -n llm-inference \
    -o jsonpath='{.data.API_KEY}' | base64 -d)

curl -X POST http://localhost:8080/v1/documents \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${RAG_API_KEY}" \
    -d '{
      "documents": [{
        "id": "policy-1",
        "text": "Our refund policy allows returns within 30 days of purchase with original receipt.",
        "metadata": {"source": "Policy Manual", "version": "2.1"}
      }]
    }'

# Query it
curl -X POST http://localhost:8080/v1/chat \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${RAG_API_KEY}" \
    -d '{"query": "What is the refund policy?", "user_id": "test-user"}'

kill %1
```

---

## Step 8: Expose Publicly via GKE Ingress (HTTPS)

### Option A — Simple (no custom domain, use GCP's IP)

```bash
# Change Service type from ClusterIP to LoadBalancer in the deployment.yaml
kubectl patch svc rag-api-service -n llm-inference \
    -p '{"spec":{"type":"LoadBalancer"}}'

# Get external IP (takes ~2 minutes)
kubectl get svc rag-api-service -n llm-inference
# EXTERNAL-IP column will show your IP once provisioned

# Test
EXTERNAL_IP=$(kubectl get svc rag-api-service -n llm-inference \
    -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://${EXTERNAL_IP}/health
```

### Option B — Production (custom domain + managed HTTPS)

```bash
# 1. Reserve a global static IP
gcloud compute addresses create rag-api-ip --global
STATIC_IP=$(gcloud compute addresses describe rag-api-ip --global --format='value(address)')
echo "Point your DNS A record to: ${STATIC_IP}"

# 2. Create a managed SSL certificate (replace YOUR_DOMAIN)
gcloud compute ssl-certificates create rag-api-cert \
    --domains=api.YOUR_DOMAIN.com --global

# 3. Apply the Ingress (edit YOUR_DOMAIN in the YAML first)
sed -i "s/YOUR_DOMAIN/yourdomain.com/g" \
    kubernetes/gcp/rag-service/deployment.yaml

kubectl apply -f kubernetes/gcp/rag-service/deployment.yaml

# 4. Wait for cert provisioning (~10-15 min after DNS propagates)
kubectl describe ingress rag-api-ingress -n llm-inference
```

---

## Step 9: Enable Prometheus Monitoring

```bash
# Apply existing Prometheus rules
kubectl create configmap prometheus-rules \
    --from-file=monitoring/prometheus/llm-rules.yaml \
    -n llm-inference

# Install kube-prometheus-stack (Prometheus + Grafana)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install kube-prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --create-namespace \
    --set grafana.adminPassword=admin123

# Access Grafana
kubectl port-forward svc/kube-prometheus-grafana 3000:80 -n monitoring
# Open: http://localhost:3000 (admin / admin123)
```

---

## Step 10 (Optional): Add vLLM on GPU Nodes

For self-hosted Llama 3 / Mistral alongside Gemini:

```bash
# Add a GPU node pool (L4 GPU, spot instances)
gcloud container node-pools create gpu-pool \
    --cluster=rag-platform-cluster \
    --region=$REGION \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --num-nodes=1 \
    --spot \
    --node-labels=workload-type=llm-inference,gpu-type=l4

# Install NVIDIA GPU drivers on the nodes
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# The original vLLM deployment (from reference-implementation) can now run:
# Just update the node selector and image in:
# kubernetes/vllm/llama-3-70b-deployment.yaml
# Change: nodeSelector: gpu-type: a100 → gpu-type: l4
# Change: resources.limits.nvidia.com/gpu: 8 → 1 (1x L4 = ~24GB VRAM, use Mistral 7B)
```

---

## Useful kubectl Commands

```bash
# Check all resources in the namespace
kubectl get all -n llm-inference

# View RAG API logs
kubectl logs -l app=rag-api -n llm-inference --tail=100

# Check HPA scaling status
kubectl get hpa -n llm-inference

# Describe a pod (for debugging)
kubectl describe pod -l app=rag-api -n llm-inference

# Scale manually (for testing)
kubectl scale deployment/rag-api --replicas=3 -n llm-inference

# Rolling restart (e.g., after updating .env / secrets)
kubectl rollout restart deployment/rag-api -n llm-inference
kubectl rollout status deployment/rag-api -n llm-inference

# Delete everything (clean up)
kubectl delete namespace llm-inference
```

---

## Cost Estimate (GKE)

| Component | Config | Monthly Cost |
|-----------|--------|-------------|
| GKE Autopilot cluster | Control plane | ~$74/month |
| RAG API pods | 2x e2-standard-4 equivalent | ~$60/month |
| Qdrant pod | e2-medium equivalent | ~$15/month |
| PD-SSD (Qdrant storage) | 20Gi | ~$3/month |
| GKE Standard (alt.) | 2x e2-standard-4 nodes | ~$95/month |
| vLLM GPU pool (optional) | 1x L4 Spot | ~$100/month |
| **Total (no GPU)** | | **~$150/month** |
| **Total (with vLLM)** | | **~$250/month** |

> **Tip**: For dev/testing, use a single-node Standard cluster with e2-medium
> nodes (~$30/month) and scale down replicas to 1. The Colab notebook is
> completely free for quick experimentation.
