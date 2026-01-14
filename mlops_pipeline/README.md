# Project 07: End-to-End MLOps Pipeline

A complete, production-ready MLOps pipeline demonstrating modern machine learning operations practices with automated training, experiment tracking, model registry, and continuous deployment.

## Overview

This project implements a comprehensive MLOps pipeline for a customer churn prediction use case, showcasing:

- **Automated Data Pipelines**: Ingestion, validation, and preprocessing with Apache Airflow
- **Experiment Tracking**: MLflow for tracking experiments, parameters, and metrics
- **Model Registry**: Versioning and lifecycle management with MLflow Model Registry
- **Data Versioning**: DVC for data and model versioning
- **Continuous Deployment**: Automated deployment to Kubernetes with validation gates
- **Monitoring & Observability**: Prometheus and Grafana for pipeline and model monitoring
- **Drift Detection**: Data and model drift detection capabilities

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MLOps Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Data Pipeline (Airflow)                                   │
│  ├── Ingestion → Validation → Preprocessing → DVC         │
│  └── Triggers Training Pipeline                            │
│                                                             │
│  Training Pipeline (Airflow + MLflow)                      │
│  ├── Load Data → Train Models → Evaluate                  │
│  ├── Select Best → Register to MLflow Registry            │
│  └── Triggers Deployment Pipeline                          │
│                                                             │
│  Deployment Pipeline (Airflow + Kubernetes)                │
│  ├── Validation → Deployment → Smoke Tests                │
│  └── Monitoring Setup                                      │
│                                                             │
│  Monitoring (Prometheus + Grafana)                         │
│  ├── Pipeline Metrics → Model Performance                 │
│  └── Drift Detection → Alerting                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Workflow Orchestration | Apache Airflow (2.7.3) |
| Experiment Tracking | MLflow (2.8.1) |
| Data Versioning | DVC (3.30.1) |
| Model Training | Scikit-learn, XGBoost |
| Object Storage | MinIO (S3-compatible) |
| Database | PostgreSQL (14) |
| Message Broker | Redis (7) |
| Container Orchestration | Kubernetes |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |

## Quick Start

### Prerequisites

- Docker (20.10+)
- Docker Compose (2.0+)
- 8GB RAM minimum
- 20GB disk space

### Local Development Setup

1. **Clone and Navigate**
   ```bash
   cd project-102-mlops-pipeline
   ```

2. **Run Setup Script**
   ```bash
   ./scripts/setup.sh
   ```

3. **Access Services**
   - Airflow UI: http://localhost:8080 (admin/admin)
   - MLflow UI: http://localhost:5000
   - MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

### Running the Pipeline

#### Option 1: Using Makefile
```bash
# Start all services
make docker-up

# Run data pipeline
make run-data-pipeline

# Run training pipeline
make run-training-pipeline

# Run deployment pipeline
make run-deployment-pipeline
```

#### Option 2: Using Airflow UI
1. Open Airflow UI at http://localhost:8080
2. Enable DAGs: `data_pipeline`, `training_pipeline`, `deployment_pipeline`
3. Trigger `data_pipeline` manually
4. Pipelines will cascade automatically

#### Option 3: Using CLI
```bash
docker-compose exec airflow-webserver airflow dags trigger data_pipeline
docker-compose exec airflow-webserver airflow dags trigger training_pipeline
docker-compose exec airflow-webserver airflow dags trigger deployment_pipeline
```

## Project Structure

```
project-102-mlops-pipeline/
├── dags/                           # Airflow DAG definitions
│   ├── data_pipeline.py           # Data ingestion & preprocessing
│   ├── training_pipeline.py       # Model training & registration
│   └── deployment_pipeline.py     # Model deployment
├── src/                            # Source code
│   ├── common/                     # Shared utilities
│   │   ├── config.py              # Configuration management
│   │   ├── logger.py              # Logging utilities
│   │   └── storage.py             # S3/MinIO client
│   ├── data/                       # Data processing modules
│   │   ├── ingestion.py           # Data ingestion
│   │   ├── validation.py          # Data quality validation
│   │   └── preprocessing.py       # Feature engineering
│   ├── training/                   # Model training modules
│   │   ├── trainer.py             # Model training
│   │   ├── evaluator.py           # Model evaluation
│   │   └── registry.py            # MLflow registry management
│   ├── deployment/                 # Deployment modules
│   │   ├── deployer.py            # Deployment orchestration
│   │   └── kubernetes_client.py   # K8s operations
│   └── monitoring/                 # Monitoring modules
│       ├── drift_detector.py      # Drift detection
│       └── metrics_collector.py   # Metrics collection
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── conftest.py                 # Pytest fixtures
├── docker/                         # Docker configurations
│   ├── Dockerfile.airflow         # Airflow image
│   └── Dockerfile.mlflow          # MLflow image
├── kubernetes/                     # Kubernetes manifests
│   ├── airflow/                   # Airflow deployment
│   ├── mlflow/                    # MLflow deployment
│   ├── postgres/                  # PostgreSQL
│   ├── redis/                     # Redis
│   └── minio/                     # MinIO
├── monitoring/                     # Monitoring configuration
│   ├── prometheus/                # Prometheus config
│   └── grafana/                   # Grafana dashboards
├── scripts/                        # Utility scripts
│   ├── setup.sh                   # Initial setup
│   ├── deploy.sh                  # Kubernetes deployment
│   ├── test-pipeline.sh           # End-to-end test
│   └── promote-model.sh           # Model promotion
├── docs/                           # Documentation
│   ├── ARCHITECTURE.md            # System architecture
│   ├── PIPELINE.md                # Pipeline documentation
│   ├── MLFLOW.md                  # MLflow guide
│   ├── DEPLOYMENT.md              # Deployment guide
│   ├── DVC.md                     # DVC guide
│   └── TROUBLESHOOTING.md         # Common issues
├── docker-compose.yml              # Local development stack
├── requirements.txt                # Python dependencies
├── requirements-dev.txt            # Development dependencies
├── Makefile                        # Common commands
├── .env.example                    # Environment template
└── README.md                       # This file
```

## Use Case: Customer Churn Prediction

The pipeline trains a binary classification model to predict customer churn:

- **Input Features**: Customer demographics, usage patterns, service subscriptions
- **Target**: Binary churn indicator (0 = retained, 1 = churned)
- **Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Sample Data

The pipeline includes synthetic data generation for testing. Features include:
- Tenure (months)
- Monthly charges
- Total charges
- Contract type
- Payment method
- Internet service type
- Add-on services (security, support, streaming)

## Key Features

### 1. Data Pipeline
- Automated data ingestion from multiple sources
- Comprehensive data validation (schema, quality, distributions)
- Feature engineering and preprocessing
- Data versioning with DVC
- S3-compatible storage (MinIO)

### 2. Training Pipeline
- Multiple model training in parallel
- Hyperparameter tuning with RandomizedSearchCV
- Cross-validation for robust evaluation
- MLflow experiment tracking
- Model registry integration
- Performance validation gates

### 3. Model Registry
- Version control for models
- Stage transitions (Staging → Production)
- Model lineage and metadata
- Performance comparison across versions
- Artifact storage in S3

### 4. Deployment Pipeline
- Automated deployment from registry
- Integration tests before deployment
- Rolling updates to Kubernetes
- Smoke tests post-deployment
- Rollback on failure

### 5. Monitoring & Drift Detection
- Real-time pipeline metrics
- Model performance tracking
- Data drift detection (KS test, JS divergence)
- Prediction drift detection
- Alerting via Prometheus
- Custom Grafana dashboards

### 6. CI/CD Integration
- Automated testing on PR
- Linting and code quality checks
- Model training in CI
- Performance validation
- Automated deployment to staging
- Manual approval for production

## Development

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# With coverage
make test-coverage
```

### Code Quality

```bash
# Lint
make lint

# Format
make format

# Type checking
mypy src
```

### Local Development

```bash
# Install dependencies
make install

# Start services
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

## Deployment

### Docker Deployment

```bash

# Start Minikube with sufficient resources
minikube start --cpus=4 --memory=8192 --driver=docker
# Or
minikube start --cpus=4 --memory=8192 --disk-size=20g

# Enable required addons
minikube addons enable metrics-server
minikube addons enable ingress

# Build and Load Images (The "Minikube Dance")

# 1. Build and Load Airflow:

docker build -f Dockerfile -t mlops-airflow:latest .
minikube image load mlops-airflow:latest

# 2. Build and Load MLflow:

docker build -f docker/Dockerfile.mlflow -t mlops-mlflow:latest .
minikube image load mlops-mlflow:latest

# 3. Build and Load Model Server:

docker build -f Dockerfile.model -t model-serving-api:latest .
minikube image load model-serving-api:latest

```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
./scripts/deploy.sh


# 1. Clean up existing deployment (if any)
# Delete the namespace (this removes everything)
kubectl delete namespace ml-serving
# Wait a moment for cleanup
Start-Sleep -Seconds 10

# 2. Create namespace
kubectl create namespace ml-serving

# 3. Deploy in the correct order
# Deploy infrastructure first (order matters!)
kubectl apply -f kubernetes/postgres/deployment.yaml
kubectl apply -f kubernetes/redis/deployment.yaml
kubectl apply -f kubernetes/minio/deployment.yaml

# Wait for infrastructure to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n ml-serving --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n ml-serving --timeout=300s
kubectl wait --for=condition=ready pod -l app=minio -n ml-serving --timeout=300s

# Deploy MLflow
kubectl apply -f kubernetes/mlflow/deployment.yaml
kubectl wait --for=condition=ready pod -l app=mlflow -n ml-serving --timeout=300s

# Deploy Airflow (optional, if you need it)
kubectl apply -f kubernetes/airflow/

# Deploy ConfigMaps
kubectl apply -f kubernetes/configmap.yaml

# Finally deploy your model
kubectl apply -f kubernetes/model-deployment.yaml
kubectl apply -f kubernetes/service.yaml

# 4. Monitor the deployment
# Watch pods come up
kubectl get pods -n ml-serving -w

# In another terminal, check logs
kubectl logs -n ml-serving -l app=mlops-model -f

# 5. Verify everything is working
# Check all resources
kubectl get all -n ml-serving

# Check if MLflow is accessible
kubectl run -n ml-serving curl-test --image=curlimages/curl --rm -it --restart=Never -- curl http://mlflow:5000/health

# Check model pod logs
kubectl logs -n ml-serving -l app=mlops-model --tail=50



# Or using kubectl
kubectl apply -f kubernetes/

# Check status
kubectl get pods -n ml-serving
kubectl get services -n ml-serving


# Verification
# Check all pods
kubectl get pods -n ml-serving

# Check MLflow logs
kubectl logs -n ml-serving -l app=mlflow --tail=50

# Check Airflow scheduler logs (this loads DAGs)
kubectl logs -n ml-serving -l component=scheduler --tail=50

# Check DAGs (should show your 3 DAGs)
kubectl port-forward -n ml-serving svc/airflow-webserver 8080:8080
# Open http://localhost:8080 (admin/admin)

# Check MLflow
kubectl port-forward -n ml-serving svc/mlflow 5000:5000
# Open http://localhost:5000

# Check MinIO
kubectl port-forward -n ml-serving svc/minio 9001:9001
# Open http://localhost:9001 (minioadmin/minioadmin)

```


### Debug

```bash
# Lint
make lint

# Format
make format

# Type checking
mypy src
```

### Local Development

```bash
# Install dependencies
make install

# Start services
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

## Deployment

### Docker Deployment

```bash

# Start Minikube with sufficient resources
minikube start --cpus=4 --memory=8192 --driver=docker
# Or
minikube start --cpus=4 --memory=8192 --disk-size=20g --driver=docker

# Enable required addons
minikube addons enable metrics-server
minikube addons enable ingress

# Build and Load Images (The "Minikube Dance")

# 1. Build and Load Airflow:

docker build -f docker/Dockerfile.airflow -t custom-airflow:2.9.0 .
minikube image load custom-airflow:2.9.0

# 2. Build and Load MLflow:

docker build -f docker/Dockerfile.mlflow -t mlops-mlflow:latest .
minikube image load mlops-mlflow:latest

# 3. Build and Load Model Server:

docker build -f docker/Dockerfile.model -t model-serving-api:latest .
minikube image load model-serving-api:latest

```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
./scripts/deploy.sh


# 1. Clean up existing deployment (if any)
# Delete the namespace (this removes everything)
kubectl delete namespace ml-serving
# Wait a moment for cleanup
Start-Sleep -Seconds 10

# 2. Create namespace
kubectl create namespace ml-serving

# 3. Deploy in the correct order
# Deploy infrastructure first (order matters!)
kubectl apply -f kubernetes/postgres/deployment.yaml
kubectl apply -f kubernetes/redis/deployment.yaml
kubectl apply -f kubernetes/minio/

# Wait for infrastructure to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n ml-serving --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n ml-serving --timeout=300s
kubectl wait --for=condition=ready pod -l app=minio -n ml-serving --timeout=300s

# Deploy MLflow
kubectl apply -f kubernetes/mlflow/deployment.yaml
kubectl wait --for=condition=ready pod -l app=mlflow -n ml-serving --timeout=300s

# Deploy Airflow (optional, if you need it)
kubectl apply -f kubernetes/airflow/

# Deploy ConfigMaps
kubectl apply -f kubernetes/configmap.yaml

# Finally deploy your model
kubectl apply -f kubernetes/model-deployment.yaml
kubectl apply -f kubernetes/service.yaml

# 4. Monitor the deployment
# Watch pods come up
kubectl get pods -n ml-serving -w

# In another terminal, check logs
kubectl logs -n ml-serving -l app=mlops-model -f

# 5. Verify everything is working
# Check all resources
kubectl get all -n ml-serving

# Check if MLflow is accessible
kubectl run -n ml-serving curl-test --image=curlimages/curl --rm -it --restart=Never -- curl http://mlflow:5000/health

# Check model pod logs
kubectl logs -n ml-serving -l app=mlops-model --tail=50



# Or using kubectl
kubectl apply -f kubernetes/

# Check status
kubectl get pods -n ml-serving
kubectl get services -n ml-serving


# Verification
# Check all pods
kubectl get pods -n ml-serving

# Check MLflow logs
kubectl logs -n ml-serving -l app=mlflow --tail=50

# Check Airflow scheduler logs (this loads DAGs)
kubectl logs -n ml-serving -l component=scheduler --tail=50

# Check DAGs (should show your 3 DAGs)
kubectl port-forward -n ml-serving svc/airflow-webserver 8080:8080
# Open http://localhost:8080 (admin/admin)

# Check MLflow
kubectl port-forward -n ml-serving svc/mlflow 5000:5000
# Open http://localhost:5000

# Check MinIO
kubectl port-forward -n ml-serving svc/minio 9001:9001
# Open http://localhost:9001 (minioadmin/minioadmin)

```


### Debug

```bash
# airflow

# Check scheduler logs
kubectl logs -n ml-serving airflow-scheduler-84d7c8545-c4x5d --tail=100

# If the pod has restarted, check previous logs
kubectl logs -n ml-serving airflow-scheduler-84d7c8545-c4x5d --previous --tail=100

# Apply the Fix
# Delete the crashing pods
kubectl delete deployment airflow-scheduler airflow-webserver -n ml-serving
# Delete everything Airflow-related
kubectl delete job airflow-copy-dags -n ml-serving

# Wait a moment
Start-Sleep -Seconds 5

# Apply the fixed deployment
kubectl apply -f kubernetes/airflow/deployment.yaml

# Watch it come up
kubectl get pods -n ml-serving -w

# Monitor the Fix
# Watch scheduler logs in real-time
kubectl logs -n ml-serving -l component=scheduler -f

# In another terminal, watch webserver
kubectl logs -n ml-serving -l component=webserver -f



# Delete the old init job:
kubectl delete job airflow-init -n ml-serving
kubectl delete job airflow-copy-dags -n ml-serving

# Ensure your image is loaded in Minikube:
docker build -f Dockerfile.airflow -t custom-airflow:2.9.0 .
minikube image load custom-airflow:2.9.0

# Apply the new configuration:
kubectl apply -f airflow_init_job.yaml
kubectl apply -f deployment.yaml

# Restart the pods (Critical):
kubectl delete pods -n ml-serving -l app=airflow
# Once the new pods come up, check with 
kubectl get pods -n ml-serving





# mlops-model-deployment

# force Kubernetes to restart the pod and pick up the new code:
kubectl rollout restart deployment mlops-model-deployment -n ml-serving
# Wait 30 seconds and check:
kubectl logs -n ml-serving -l component=model-server

# Delete the old deployment
kubectl delete deployment mlops-model-deployment -n ml-serving
# Re-apply it
kubectl apply -f kubernetes/model-deployment.yaml




# model-server

# Verify the Switch 
docker images.
# If you see a lot of "k8s.gcr.io" images, it worked.
# If you only see your own local images, it failed. Stop here.

# Build the Image (Again) 
# Since we are now "inside" Minikube, building here saves it where Kubernetes can instantly see it.
docker build --no-cache -f Dockerfile.model -t model-serving-api:latest .

# Delete the Broken Pods 
# Now that the image is definitely there, kill the stuck pods to force a restart.
kubectl delete pod -n ml-serving -l component=model-server

# Check Logs Correctly
# Since the pod names keep changing, stop copy-pasting the random names. 
# Use this "Label Selector" command instead. 
# It automatically finds the right pod for you:
kubectl logs -n ml-serving -l component=model-server -f




# rebuild and redeploy

# 1. Delete the failing pods
kubectl delete deployment churn-model -n ml-serving

# 2. Rebuild the image
docker build -f Dockerfile.model-server -t model-server:latest .

# 3. Load into Minikube
minikube image load model-server:latest

# 4. Verify the image is loaded
minikube image ls | grep model-server

# 5. Redeploy
kubectl apply -f kubernetes/model-serving/churn-model-deployment.yaml

# 6. Watch the pods start
kubectl get pods -n ml-serving -l app=churn-model -w

# 7. Check logs
kubectl logs -n ml-serving -l app=churn-model -f
```


### Model Promotion

```bash
# Promote model to production
./scripts/promote-model.sh churn-classifier 3

# This will:
# 1. Transition model to Production stage
# 2. Trigger deployment pipeline
# 3. Deploy to Kubernetes
```

## Monitoring

### Accessing Monitoring Tools

1. **Grafana Dashboards**: http://localhost:3000
   - MLOps Overview Dashboard
   - Pipeline Execution Metrics
   - Model Performance Dashboard
   - Resource Utilization

2. **Prometheus**: http://localhost:9090
   - Query metrics directly
   - View alerting rules
   - Check targets health

3. **MLflow**: http://localhost:5000
   - Experiment tracking
   - Model registry
   - Artifact browser

### Key Metrics

- `pipeline_runs_total`: Total pipeline runs by status
- `pipeline_duration_seconds`: Pipeline execution time
- `data_quality_score`: Data quality assessment
- `model_accuracy`: Model accuracy score
- `feature_drift_score`: Feature drift detection
- `prediction_drift_score`: Prediction drift detection

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# Storage
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_ENDPOINT_URL=http://minio:9000

# Model thresholds
MIN_ACCURACY=0.75
MIN_F1_SCORE=0.70

# Drift thresholds
DRIFT_THRESHOLD=0.1
```

### Customization

1. **Add New Models**: Edit `src/training/trainer.py:get_model_configs()`
2. **Add Features**: Edit `src/data/preprocessing.py:_engineer_features()`
3. **Adjust Thresholds**: Update `.env` file
4. **Custom Metrics**: Add to `src/monitoring/metrics_collector.py`

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues and solutions.

Quick fixes:
```bash
# Reset everything
make docker-down
make clean
make docker-build
make docker-up

# Check service health
make check-status

# View logs
docker-compose logs -f [service-name]
```

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and components
- [Pipeline Documentation](docs/PIPELINE.md) - Detailed pipeline workflows
- [MLflow Guide](docs/MLFLOW.md) - Experiment tracking and registry
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment


## Performance

### Benchmarks (10K samples)

| Pipeline | Duration | Resources |
|----------|----------|-----------|
| Data Pipeline | ~2 min | 1 CPU, 2GB RAM |
| Training Pipeline | ~5 min | 2 CPU, 4GB RAM |
| Deployment | ~1 min | 1 CPU, 1GB RAM |

### Scalability

- Handles datasets up to 1M rows
- Supports distributed training with Ray/Horovod (configurable)
- Kubernetes autoscaling for production loads
- Horizontal scaling of Airflow workers

## Security

- No hardcoded credentials (environment variables only)
- S3 bucket access controls
- Kubernetes RBAC
- Secrets management via K8s secrets
- Network policies for pod communication
