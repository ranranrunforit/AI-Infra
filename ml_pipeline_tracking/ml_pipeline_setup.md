# ML Pipeline Project - Complete Setup Guide

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] Docker Desktop installed and running
- [ ] Docker Compose v2.0+ installed
- [ ] Python 3.11+ installed
- [ ] Git installed
- [ ] 16GB+ RAM available
- [ ] 50GB+ free disk space
- [ ] Terminal/command line access

## 🚀 Quick Start (30 minutes)

### Step 1: Project Setup (5 minutes)

```bash
# Create project directory
mkdir -p ~/projects/ml-pipeline-project
cd ~/projects/ml-pipeline-project

# Create directory structure
mkdir -p {data/{raw,processed},src,dags,mlflow,dvc,artifacts,models,tests,docs,notebooks}

# Initialize Git
git init
```

### Step 2: Create Configuration Files (10 minutes)

#### 2.1 Docker Compose Configuration

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # PostgreSQL - Database for MLflow and Airflow
  postgres:
    image: postgres:15
    container_name: mlpipeline-postgres
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow_password
      POSTGRES_MULTIPLE_DATABASES: mlflow,airflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sh:/docker-entrypoint-initdb.d/init-db.sh
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflow"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - mlpipeline

  # MinIO - S3-compatible object storage for artifacts
  minio:
    image: minio/minio:latest
    container_name: mlpipeline-minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - mlpipeline

  # Create MinIO bucket
  minio-init:
    image: minio/mc:latest
    container_name: mlpipeline-minio-init
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
      sleep 10;
      /usr/bin/mc alias set myminio http://minio:9000 minioadmin minioadmin123;
      /usr/bin/mc mb myminio/mlflow || true;
      /usr/bin/mc mb myminio/dvc || true;
      exit 0;
      "
    networks:
      - mlpipeline

  # Redis - Message broker for Airflow
  redis:
    image: redis:7
    container_name: mlpipeline-redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - mlpipeline

  # MLflow Tracking Server
  mlflow:
    build:
      context: ./mlflow
      dockerfile: Dockerfile
    container_name: mlpipeline-mlflow
    depends_on:
      postgres:
        condition: service_healthy
      minio:
        condition: service_healthy
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:mlflow_password@postgres:5432/mlflow
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin123
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow_password@postgres:5432/mlflow
      --default-artifact-root s3://mlflow/
      --host 0.0.0.0
      --port 5000
    networks:
      - mlpipeline

  # Airflow Webserver
  airflow-webserver:
    build:
      context: ./airflow
      dockerfile: Dockerfile
    container_name: mlpipeline-airflow-webserver
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql://mlflow:mlflow_password@postgres:5432/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://redis:6379/0
      AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://mlflow:mlflow_password@postgres:5432/airflow
      AIRFLOW__CORE__FERNET_KEY: 'UKMzEm3yIuFYEq1y3-2FxPNWSVwRASpahmQ9kQfEr8E='
      AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
      AIRFLOW__API__AUTH_BACKENDS: 'airflow.api.auth.backend.basic_auth'
      AIRFLOW__WEBSERVER__SECRET_KEY: 'secret_key_12345'
    volumes:
      - ./dags:/opt/airflow/dags
      - ./src:/opt/airflow/src
      - ./data:/opt/airflow/data
      - ./artifacts:/opt/airflow/artifacts
      - ./models:/opt/airflow/models
      - ./logs:/opt/airflow/logs
    ports:
      - "8080:8080"
    command: webserver
    networks:
      - mlpipeline

  # Airflow Scheduler
  airflow-scheduler:
    build:
      context: ./airflow
      dockerfile: Dockerfile
    container_name: mlpipeline-airflow-scheduler
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql://mlflow:mlflow_password@postgres:5432/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://redis:6379/0
      AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://mlflow:mlflow_password@postgres:5432/airflow
      AIRFLOW__CORE__FERNET_KEY: 'UKMzEm3yIuFYEq1y3-2FxPNWSVwRASpahmQ9kQfEr8E='
      AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    volumes:
      - ./dags:/opt/airflow/dags
      - ./src:/opt/airflow/src
      - ./data:/opt/airflow/data
      - ./artifacts:/opt/airflow/artifacts
      - ./models:/opt/airflow/models
      - ./logs:/opt/airflow/logs
    command: scheduler
    networks:
      - mlpipeline

  # Airflow Worker
  airflow-worker:
    build:
      context: ./airflow
      dockerfile: Dockerfile
    container_name: mlpipeline-airflow-worker
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql://mlflow:mlflow_password@postgres:5432/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://redis:6379/0
      AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://mlflow:mlflow_password@postgres:5432/airflow
      AIRFLOW__CORE__FERNET_KEY: 'UKMzEm3yIuFYEq1y3-2FxPNWSVwRASpahmQ9kQfEr8E='
    volumes:
      - ./dags:/opt/airflow/dags
      - ./src:/opt/airflow/src
      - ./data:/opt/airflow/data
      - ./artifacts:/opt/airflow/artifacts
      - ./models:/opt/airflow/models
      - ./logs:/opt/airflow/logs
    command: celery worker
    networks:
      - mlpipeline

volumes:
  postgres_data:
  minio_data:

networks:
  mlpipeline:
    driver: bridge
```

#### 2.2 PostgreSQL Init Script

Create `init-db.sh`:

```bash
#!/bin/bash
set -e

# Create multiple databases
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE mlflow;
    CREATE DATABASE airflow;
EOSQL
```

Make it executable:
```bash
chmod +x init-db.sh
```

#### 2.3 MLflow Dockerfile

Create `mlflow/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /mlflow

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    mlflow==2.8.1 \
    psycopg2-binary \
    boto3

# Expose MLflow port
EXPOSE 5000

# Default command (overridden by docker-compose)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
```

#### 2.4 Airflow Dockerfile

Create `airflow/Dockerfile`:

```dockerfile
FROM apache/airflow:2.7.3-python3.11

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Set working directory
WORKDIR /opt/airflow
```

Create `airflow/requirements.txt`:

```txt
# Airflow providers
apache-airflow-providers-celery==3.3.4
apache-airflow-providers-postgres==5.7.1
apache-airflow-providers-redis==3.3.1

# ML Libraries
torch==2.1.0
torchvision==0.16.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2

# MLflow
mlflow==2.8.1
boto3==1.29.7

# Data Validation
great-expectations==0.18.3

# DVC
dvc==3.30.1
dvc-s3==2.23.0

# Utilities
requests==2.31.0
matplotlib==3.8.2
seaborn==0.13.0
joblib==1.3.2
psycopg2-binary==2.9.9
```

### Step 3: Environment Configuration (5 minutes)

Create `.env`:

```bash
# PostgreSQL
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow_password
POSTGRES_DB=mlflow

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_ENDPOINT=http://minio:9000

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin123

# Airflow
AIRFLOW_UID=50000
AIRFLOW__CORE__EXECUTOR=CeleryExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://mlflow:mlflow_password@postgres:5432/airflow
```

Create `.gitignore`:

```txt
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/*.pth
models/*.pkl
*.h5

# MLflow
mlruns/

# DVC
.dvc/cache/

# Logs
*.log
logs/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Environment
.env
```

### Step 4: Create Python Requirements (5 minutes)

Create `requirements.txt` (for local development):

```txt
# Core ML
torch==2.1.0
torchvision==0.16.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2

# MLflow
mlflow==2.8.1
boto3==1.29.7

# Data Validation
great-expectations==0.18.3

# DVC
dvc==3.30.1
dvc-s3==2.23.0

# Utilities
requests==2.31.0
matplotlib==3.8.2
seaborn==0.13.0
joblib==1.3.2
pytest==7.4.3
jupyter==1.0.0
```

### Step 5: Initialize DVC (5 minutes)

```bash
# Install DVC locally
pip install dvc dvc-s3

# Initialize DVC
dvc init

# Configure DVC remote (MinIO)
dvc remote add -d minio s3://dvc
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin123

# Commit DVC config
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

## 🏗️ Build and Start Services

### Step 1: Build Docker Images

```bash
# Build all services
docker-compose build

# This may take 10-15 minutes on first build
```

### Step 2: Initialize Airflow Database

```bash
# Initialize Airflow DB
docker-compose run --rm airflow-webserver airflow db init

# Create admin user
docker-compose run --rm airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

While your PostgreSQL database is running, the user `airflow` has not been created yet.

This happens because the `postgres` container only creates one default user on startup (defined in your `docker-compose.yaml` as `POSTGRES_USER`). Usually, this default user is named `user` or `mlflow`, not `airflow`.

You need to manually create the `airflow` user and database. Follow these steps:

### Step 1: Log in as the Superuser

Since we don't know exactly which default user you set in `docker-compose.yaml` (it's either `user` or `mlflow`), try logging in with the default `user`:

```powershell
docker-compose exec postgres psql -U user -d postgres

```

*(If that fails with "role user does not exist", try changing `-U user` to `-U mlflow`)*.

### Step 2: Create the Airflow User and Database

Once you see the `postgres=#` prompt, run these SQL commands one by one (don't forget the semicolons `;`):

1. **Create the user:**
```sql
CREATE USER airflow WITH PASSWORD 'airflow';

```

2. **Create the database:**
```sql
CREATE DATABASE airflow;

```

3. **Grant permissions:**
```sql
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;

```

*(Note: On newer Postgres versions, you might also need: `ALTER DATABASE airflow OWNER TO airflow;`)*

4. **Exit:**
```sql
\q

```

### Step 3: Verify

Now try your original command again. It should work:

```powershell
docker-compose exec postgres psql -U airflow -d airflow -c "\l"

```

### Step 3: Start All Services

```bash
# Start all services in background
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Step 4: Verify Services

Open your browser and check:

1. **MLflow UI**: http://localhost:5000
   - Should show "No experiments yet"

2. **Airflow UI**: http://localhost:8080
   - Login: admin / admin
   - Should show Airflow dashboard

3. **MinIO Console**: http://localhost:9001
   - Login: minioadmin / minioadmin123
   - Should show "mlflow" and "dvc" buckets

4. **PostgreSQL**: 
   ```bash
   docker exec -it mlpipeline-postgres psql -U mlflow -d mlflow -c "SELECT 1;"
   # Should return: 1
   ```

## 🧪 Test Basic Functionality

### Test 1: MLflow Connection

Create `test_mlflow.py`:

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Create experiment
experiment_id = mlflow.create_experiment("test_experiment")

# Start run and log
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_param("test_param", "hello")
    mlflow.log_metric("test_metric", 42)
    print("✅ MLflow test successful!")
```

Run:
```bash
python test_mlflow.py
```

Check http://localhost:5000 - you should see the test experiment!

### Test 2: DVC Connection

```bash
# Create test file
echo "test data" > data/raw/test.txt

# Track with DVC
dvc add data/raw/test.txt

# Push to MinIO
dvc push

# Should see: "1 file pushed"
```

### Test 3: Airflow DAG

Create a simple test DAG in `dags/test_dag.py`:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def hello_world():
    print("Hello from Airflow!")
    return "Success"

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

with DAG(
    'test_dag',
    default_args=default_args,
    description='Test DAG',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False
) as dag:
    
    task = PythonOperator(
        task_id='hello_task',
        python_callable=hello_world
    )
```

Check Airflow UI - DAG should appear in ~30 seconds!

## 🎯 Next Steps

Now that your infrastructure is running, you can:

1. **Implement the Python modules** (data_ingestion.py, preprocessing.py, etc.)
2. **Create the ML pipeline DAG** in Airflow
3. **Get sample data** for training
4. **Run your first experiment**

Would you like me to:
1. Show you how to implement each Python module step-by-step?
2. Help you find and prepare a sample dataset?
3. Create the complete ML pipeline DAG?
4. Set up a sample training workflow?

## 🔧 Troubleshooting

### Services won't start?
```bash
# Check logs
docker-compose logs postgres
docker-compose logs mlflow

# Restart services
docker-compose restart
```

### Port already in use?
```bash
# Change ports in docker-compose.yml
# For example, change 5000:5000 to 5001:5000
```

### Permission issues?
```bash
# Fix permissions
chmod -R 755 data/ artifacts/ models/
```

### Clean slate restart?
```bash
# Stop and remove everything
docker-compose down -v

# Remove all data
rm -rf data/raw/* data/processed/* mlruns/* .dvc/cache/*

# Start fresh
docker-compose up -d
```

## 📚 Useful Commands

```bash
# View logs
docker-compose logs -f [service_name]

# Restart single service
docker-compose restart mlflow

# Stop all
docker-compose stop

# Remove all (including volumes)
docker-compose down -v

# Execute commands in container
docker exec -it mlpipeline-airflow-webserver bash

# Check resource usage
docker stats
```
