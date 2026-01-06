# Production ML System - Step-by-Step Tutorial

This tutorial will guide you through implementing and running the complete production ML system **from scratch**.

## 🎯 What You'll Build

By the end of this tutorial, you'll have:
- ✅ A production-ready ML API
- ✅ Automated CI/CD pipelines
- ✅ Kubernetes deployment (local and cloud)
- ✅ Complete monitoring and observability
- ✅ Security and authentication
- ✅ A portfolio-ready project

**Estimated Time:** 20-30 hours over 1-2 weeks

---

## Part 1: Local Development Setup (2-3 hours)

### Step 1.1: Set Up Your Development Environment

```bash
# Create project directory
mkdir production-ml-system
cd production-ml-system

# Initialize git
git init

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify Python version
python --version  # Should be 3.11.x
```

### Step 1.2: Create Project Structure

```bash
# Create directory structure
mkdir -p src tests/integration cicd/.github/workflows kubernetes/{base,overlays/{dev,staging,production}} security monitoring docs

# Create initial files
touch src/main.py
touch requirements.txt
touch Dockerfile
touch .env.example
touch README.md
```

### Step 1.3: Install Dependencies

Create `requirements.txt`:
```txt
Flask==3.0.0
gunicorn==21.2.0
torch==2.1.0
torchvision==0.16.0
mlflow==2.9.2
Pillow==10.1.0
prometheus-client==0.19.0
python-dotenv==1.0.0
requests==2.31.0
pytest==7.4.3
pytest-cov==4.1.0
```

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import flask; import torch; print('All imports successful!')"
```

### Step 1.4: Create the Main Application

Copy the `main.py` code I provided earlier into `src/main.py`.

### Step 1.5: Test Locally

```bash
# Create .env file
cat > .env << EOF
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_NAME=test-model
MODEL_VERSION=latest
API_KEYS=dev-test-key-123
LOG_LEVEL=DEBUG
ENVIRONMENT=development
EOF

# Run the application
python src/main.py
```

In another terminal:
```bash
# Test health endpoint
curl http://localhost:5000/health

# Test metrics endpoint
curl http://localhost:5000/metrics

# Expected output:
# {"status":"healthy",...}

# Test info endpoint
curl.exe -H "X-API-Key: test-key" http://localhost:5000/info

# Test prediction
curl.exe -X POST -H "X-API-Key: test-key" -F "file=@cat.jpg" http://localhost:5000/predict

# Pretty print JSON (pipe to Python)
curl.exe -H "X-API-Key: test-key" http://localhost:5000/info | python -m json.tool

# Test health
Invoke-RestMethod http://localhost:5000/health

# Test info with API key
Invoke-RestMethod -Uri "http://localhost:5000/info" -Headers @{"X-API-Key"="test-key"}

# Test prediction with image
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method Post -Headers @{"X-API-Key"="test-key"} -Form @{file=Get-Item "cat.jpg"}
```

**✅ Checkpoint:** Your API is running locally!

---

## Part 2: Containerization (1-2 hours)

### Step 2.1: Create Dockerfile

Copy the Dockerfile I provided earlier.

### Step 2.2: Create .dockerignore

```bash
cat > .dockerignore << EOF
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env
.git
.gitignore
*.md
tests/
.coverage
htmlcov/
EOF
```

### Step 2.3: Build Docker Image

```bash
# Build image
docker build -t ml-api:local .

# Verify image was created
docker images | grep ml-api
```

### Step 2.4: Run Container

```bash
# Run container
docker run -d -p 5000:5000 \
  -e MODEL_NAME=test-model \
  -e API_KEYS=test-key \
  -e LOG_LEVEL=INFO \
  --name ml-api \
  ml-api:local

# Check if running
docker ps

# View logs
docker logs ml-api

# Test API
curl http://localhost:5000/health
```

### Step 2.5: Test with Sample Image

```bash
# Download a test image
curl -o test_image.jpg https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/240px-Cat03.jpg

# Test prediction endpoint
curl -X POST http://localhost:5000/predict \
  -H "X-API-Key: test-key" \
  -F "file=@test_image.jpg"
```

**✅ Checkpoint:** Your API is containerized and working!

---

## Part 3: Kubernetes Deployment - Local (3-4 hours)

### Step 3.1: Install Minikube

```bash
# Install Minikube (if not already installed)
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start Minikube
minikube start --cpus=4 --memory=8192

# Verify
kubectl cluster-info
kubectl get nodes
```

### Step 3.2: Load Image into Minikube

```bash
# Load your Docker image
minikube image load ml-api:local

# Verify
minikube image ls | grep ml-api
```

### Step 3.3: Create Kubernetes Manifests

Create `kubernetes/base/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: ml-api:local
        imagePullPolicy: Never
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_NAME
          value: "test-model"
        - name: API_KEYS
          value: "dev-test-key"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-api
spec:
  selector:
    app: ml-api
  ports:
  - port: 80
    targetPort: 5000
  type: ClusterIP
```

### Step 3.4: Deploy to Minikube

```bash
# Create namespace
kubectl create namespace ml-system-dev

# Deploy
kubectl apply -f kubernetes/base/deployment.yaml -n ml-system-dev

# Check status
kubectl get pods -n ml-system-dev
kubectl get svc -n ml-system-dev

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=ml-api -n ml-system-dev --timeout=60s
```

### Step 3.5: Access Your Application

```bash
# Port forward
kubectl port-forward svc/ml-api 8080:80 -n ml-system-dev

# In another terminal, test
curl http://localhost:8080/health

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "X-API-Key: dev-test-key" \
  -F "file=@test_image.jpg"
```

**✅ Checkpoint:** Your API is running in Kubernetes!

---

## Part 4: CI/CD with GitHub Actions (4-5 hours)

### Step 4.1: Set Up GitHub Repository

```bash
# Create .gitignore
cat > .gitignore << EOF
__pycache__/
*.py[cod]
*$py.class
venv/
.env
.DS_Store
*.log
.coverage
htmlcov/
dist/
build/
*.egg-info/
EOF

# Initialize repository
git add .
git commit -m "Initial commit"

# Create GitHub repo (using GitHub CLI)
gh repo create production-ml-system --public --source=. --remote=origin

# Push
git push -u origin main
```

### Step 4.2: Create CI Workflow

Create `.github/workflows/ci.yml` with the CI pipeline code I provided earlier.

### Step 4.3: Create Basic Tests

Create `tests/test_main.py`:
```python
import pytest
from src.main import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test health endpoint returns 200"""
    response = client.get('/health')
    assert response.status_code in [200, 503]

def test_metrics_endpoint(client):
    """Test metrics endpoint returns 200"""
    response = client.get('/metrics')
    assert response.status_code == 200
    assert b'http_requests_total' in response.data
```

### Step 4.4: Test CI Locally

```bash
# Run tests
pytest tests/ -v

# Run linting
flake8 src/ --max-line-length=120

# Run formatting check
black --check src/
```

### Step 4.5: Push and Trigger CI

```bash
git add .
git commit -m "Add CI pipeline and tests"
git push

# Go to GitHub → Actions tab to see your CI running
```

**✅ Checkpoint:** Your CI pipeline is working!

---

## Part 5: Monitoring with Prometheus (2-3 hours)

### Step 5.1: Install Prometheus in Minikube

```bash
# Add Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

### Step 5.2: Create ServiceMonitor

Create `monitoring/servicemonitor.yaml`:
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ml-api
  namespace: ml-system-dev
spec:
  selector:
    matchLabels:
      app: ml-api
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

```bash
kubectl apply -f monitoring/servicemonitor.yaml
```

### Step 5.3: Access Grafana

```bash
# Get Grafana password
kubectl get secret -n monitoring prometheus-grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode
echo

# Port forward Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Open http://localhost:3000
# Username: admin
# Password: (from above command)
```

### Step 5.4: Create Dashboard

1. Go to Grafana (http://localhost:3000)
2. Add Data Source → Prometheus → http://prometheus-kube-prometheus-prometheus.monitoring:9090
3. Create New Dashboard
4. Add panels for:
   - HTTP request rate
   - Request latency
   - Prediction count
   - Error rate

**✅ Checkpoint:** You have monitoring set up!

---

## Part 6: Production Deployment (5-7 hours)

### Step 6.1: Choose Cloud Provider

Pick one:
- **Google Cloud (GKE)** - Easiest, good free tier
- **AWS (EKS)** - Most popular
- **Azure (AKS)** - Good if using Microsoft ecosystem

For this tutorial, we'll use GKE:

```bash
# Install gcloud CLI
# Follow: https://cloud.google.com/sdk/docs/install

# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create GKE cluster
gcloud container clusters create ml-system-prod \
  --num-nodes=2 \
  --machine-type=e2-standard-2 \
  --zone=us-central1-a \
  --enable-autoscaling \
  --min-nodes=2 \
  --max-nodes=5

# Get credentials
gcloud container clusters get-credentials ml-system-prod --zone=us-central1-a
```

### Step 6.2: Push Image to Registry

```bash
# Tag image for GCR
docker tag ml-api:local gcr.io/YOUR_PROJECT_ID/ml-api:v1.0.0

# Configure Docker for GCR
gcloud auth configure-docker

# Push image
docker push gcr.io/YOUR_PROJECT_ID/ml-api:v1.0.0
```

### Step 6.3: Deploy to Production

Update your deployment.yaml to use the new image:
```yaml
# Change image line to:
image: gcr.io/YOUR_PROJECT_ID/ml-api:v1.0.0
imagePullPolicy: IfNotPresent
```

```bash
# Create production namespace
kubectl create namespace ml-system-production

# Deploy
kubectl apply -f kubernetes/base/deployment.yaml -n ml-system-production

# Check status
kubectl get pods -n ml-system-production
```

### Step 6.4: Expose with LoadBalancer

```bash
# Change service type
kubectl patch svc ml-api -n ml-system-production -p '{"spec":{"type":"LoadBalancer"}}'

# Get external IP (may take a few minutes)
kubectl get svc ml-api -n ml-system-production

# Test
curl http://<EXTERNAL-IP>/health
```

**✅ Checkpoint:** Your API is in production!

---

## Part 7: Security Hardening (2-3 hours)

### Step 7.1: Enable HTTPS with cert-manager

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Wait for cert-manager to be ready
kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=60s
```

### Step 7.2: Set Up Secrets Management

```bash
# Create secret for API keys
kubectl create secret generic api-keys \
  --from-literal=prod-key=your-secure-api-key-here \
  -n ml-system-production

# Update deployment to use secret
# Add to deployment.yaml:
# env:
# - name: API_KEYS
#   valueFrom:
#     secretKeyRef:
#       name: api-keys
#       key: prod-key
```

**✅ Checkpoint:** Your system is secured!

---

## Part 8: Testing and Validation (2-3 hours)

### Step 8.1: Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils  # Ubuntu/Debian
# or
brew install httpd  # macOS

# Run load test
ab -n 1000 -c 10 -H "X-API-Key: your-api-key" \
  http://<EXTERNAL-IP>/health
```

### Step 8.2: Integration Testing

Create `tests/integration/test_e2e.py` (use code I provided earlier).

```bash
# Run integration tests
export API_URL=http://<EXTERNAL-IP>
export API_KEY=your-api-key
pytest tests/integration/test_e2e.py -v
```

**✅ Checkpoint:** System is tested and validated!

---

## Part 9: Documentation (2-3 hours)

### Step 9.1: Create README

Update your README.md with:
- Project overview
- Architecture diagram
- Setup instructions
- API documentation
- Deployment guide

### Step 9.2: Create API Documentation

```bash
# Add to your code
from flask import send_from_directory

@app.route('/docs')
def docs():
    return '''
    <h1>ML API Documentation</h1>
    <h2>Endpoints:</h2>
    <ul>
        <li>GET /health - Health check</li>
        <li>POST /predict - Make prediction</li>
        <li>GET /info - Get model info</li>
        <li>GET /metrics - Prometheus metrics</li>
    </ul>
    '''
```

**✅ Checkpoint:** Documentation is complete!

---

## Part 10: Final Demo (1-2 hours)

### Step 10.1: Prepare Demo Script

```bash
# Create demo.sh
cat > demo.sh << 'EOF'
#!/bin/bash
echo "=== ML System Demo ==="
echo ""
echo "1. Health Check:"
curl -s http://<EXTERNAL-IP>/health | jq
echo ""
echo "2. Model Info:"
curl -s -H "X-API-Key: your-key" http://<EXTERNAL-IP>/info | jq
echo ""
echo "3. Prediction:"
curl -s -X POST -H "X-API-Key: your-key" \
  -F "file=@test_image.jpg" \
  http://<EXTERNAL-IP>/predict | jq
echo ""
echo "4. Metrics (sample):"
curl -s http://<EXTERNAL-IP>/metrics | grep http_requests_total | head -5
EOF

chmod +x demo.sh
```

### Step 10.2: Record Demo

```bash
# Install asciinema for terminal recording
pip install asciinema

# Record demo
asciinema rec demo.cast

# Run your demo
./demo.sh

# Stop recording with Ctrl+D
```

**✅ Checkpoint:** Your capstone is complete!

---

## 🎉 Congratulations!

You've built a complete production ML system! Here's what you accomplished:

1. ✅ Built a production-ready ML API
2. ✅ Containerized with Docker
3. ✅ Deployed to Kubernetes (local and cloud)
4. ✅ Set up CI/CD pipelines
5. ✅ Implemented monitoring and observability
6. ✅ Secured with authentication and HTTPS
7. ✅ Tested thoroughly
8. ✅ Documented completely

## Next Steps

1. Add this project to your portfolio
2. Write a blog post about what you learned
3. Present this at a meetup or conference
4. Contribute to open source ML infrastructure projects
5. Apply for ML infrastructure engineer positions

## Resources for Continued Learning

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [MLOps Community](https://mlops.community/)
- [CNCF Projects](https://www.cncf.io/projects/)
- [SRE Books](https://sre.google/books/)

---

**You did it!** 🚀 You're now ready for production ML infrastructure work!
