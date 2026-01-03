# How to Run This ML Pipeline Project

## 🎯 Goal
By the end of this guide, you'll have a fully functional ML pipeline with:
- Data ingestion and preprocessing
- Model training with MLflow tracking
- Automated workflows with Airflow
- Data versioning with DVC

---

## 📦 Part 1: Initial Setup (30 minutes)

### 1.1 Install Prerequisites

```bash
# Check Docker
docker --version
# Should show: Docker version 20.x or higher

# Check Docker Compose
docker-compose --version
# Should show: Docker Compose version 2.x or higher

# Check Python
python --version
# Should show: Python 3.11 or higher
```

### 1.2 Create Project Structure

```bash
# Create project directory
mkdir ~/ml-pipeline-project
cd ~/ml-pipeline-project

# Create all directories
mkdir -p data/{raw,processed} src dags mlflow airflow artifacts models tests docs notebooks logs

# Create empty __init__.py files
touch src/__init__.py
touch dags/__init__.py
```

### 1.3 Copy Configuration Files

Copy all the Docker Compose, Dockerfile, and configuration files from the artifacts I provided above:

1. **docker-compose.yml** - Main orchestration file
2. **mlflow/Dockerfile** - MLflow container
3. **airflow/Dockerfile** - Airflow container
4. **airflow/requirements.txt** - Python dependencies
5. **init-db.sh** - PostgreSQL initialization
6. **.env** - Environment variables
7. **.gitignore** - Git ignore rules

### 1.4 Copy Python Modules

Copy these files to the `src/` directory:
1. **data_ingestion.py**
2. **preprocessing.py**
3. **training.py**
4. **evaluation.py**

---

## 🚀 Part 2: Start Infrastructure (20 minutes)

### 2.1 Build Docker Images

```bash
# Make init script executable
chmod +x init-db.sh

# Build all images (this takes 10-15 minutes first time)
docker-compose build

# You should see:
# Building mlflow...
# Building airflow-webserver...
# etc.
```

### 2.2 Initialize Airflow Database

```bash
# Initialize database
docker-compose run --rm airflow-webserver airflow db init

# Create admin user
docker-compose run --rm airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# You should see: "User admin created"
```

### 2.3 Start All Services

```bash
# Start everything in background
docker-compose up -d

# Check status (all should be "Up")
docker-compose ps

# View logs
docker-compose logs -f

# Press Ctrl+C to stop viewing logs (services keep running)
```

### 2.4 Verify Services

Open your browser and check these URLs:

1. **MLflow**: http://localhost:5000
   - You should see the MLflow UI
   - No experiments yet (that's normal)

2. **Airflow**: http://localhost:8080
   - Login: `admin` / `admin`
   - You should see the Airflow dashboard

3. **MinIO**: http://localhost:9001
   - Login: `minioadmin` / `minioadmin123`
   - You should see two buckets: `mlflow` and `dvc`

**Troubleshooting:**
```bash
# If services won't start:
docker-compose logs postgres
docker-compose logs mlflow

# If ports are in use:
# Edit docker-compose.yml and change port mappings
# Example: Change "5000:5000" to "5001:5000"

# Clean slate restart:
docker-compose down -v
docker-compose up -d
```

---

## 🧪 Part 3: Test the Pipeline (15 minutes)

### 3.1 Install Local Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install requirements
pip install -r airflow/requirements.txt
```

### 3.2 Run Quick Start Script

Copy the "Quick Start Workflow" Python script I provided to your project root as `quickstart.py`, then:

```bash
# Run the complete pipeline
python quickstart.py
```

This script will:
1. Create a sample dataset
2. Ingest the data
3. Preprocess it
4. Train a model
5. Track everything in MLflow
6. Evaluate the model

**Expected output:**
```
========================================
STEP 1: Creating Sample Dataset
========================================
✅ Created sample dataset with 1010 records

========================================
STEP 2: Data Ingestion
========================================
✅ Data ingestion complete

... (more steps)

🎉 PIPELINE COMPLETE!
```

### 3.3 Check Results

1. **MLflow UI** (http://localhost:5000):
   - Go to "Experiments"
   - Click on "sample_experiment"
   - You should see your training run with:
     - Parameters (learning_rate, batch_size, etc.)
     - Metrics (train_loss, val_accuracy, etc.)
     - Artifacts (model files, plots)

2. **Check Files Created**:
   ```bash
   # Raw data
   ls data/raw/
   # Should show: sample_dataset.csv, ingested_dataset.csv

   # Processed data
   ls data/processed/
   # Should show: train.csv, val.csv, test.csv

   # Models
   ls models/
   # Should show: best_model.pth

   # Evaluation plots
   ls evaluation_plots/
   # Should show: confusion_matrix.png, classification_report.txt
   ```

---

## 📊 Part 4: Explore MLflow (10 minutes)

### 4.1 View Experiment

1. Go to http://localhost:5000
2. Click on "sample_experiment"
3. Click on the run (it will have a timestamp)

You'll see:
- **Parameters**: All hyperparameters used
- **Metrics**: Charts showing loss and accuracy over epochs
- **Artifacts**: Model files and plots

### 4.2 Compare Runs

Run the quickstart script again with different parameters:

```python
# Edit quickstart.py, change these parameters:
params = {
    'model_name': 'mobilenet_v2',  # Changed from resnet18
    'num_epochs': 5,
    'learning_rate': 0.0001,  # Changed from 0.001
    # ... rest stays the same
}
```

Then run again:
```bash
python quickstart.py
```

Now in MLflow:
1. Go to "sample_experiment"
2. Select both runs (checkboxes)
3. Click "Compare"
4. See side-by-side comparison of parameters and metrics!

### 4.3 Download Model

1. In MLflow, go to your run
2. Click "Artifacts"
3. Click on "model/"
4. You can download the model files

---

## 🔄 Part 5: Use DVC for Data Versioning (10 minutes)

### 5.1 Initialize DVC

```bash
# Initialize Git (if not already done)
git init

# Initialize DVC
dvc init

# Configure MinIO as remote
dvc remote add -d minio s3://dvc
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin123

# Commit DVC config
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### 5.2 Track Data

```bash
# Track raw data
dvc add data/raw/sample_dataset.csv

# Commit to Git
git add data/raw/sample_dataset.csv.dvc data/raw/.gitignore
git commit -m "Track sample dataset v1.0"

# Push to MinIO
dvc push

# You should see: "1 file pushed"
```

### 5.3 Version Your Data

```bash
# Modify the dataset
python -c "
import pandas as pd
df = pd.read_csv('data/raw/sample_dataset.csv')
df = df.head(500)  # Make it smaller
df.to_csv('data/raw/sample_dataset.csv', index=False)
print('Dataset modified!')
"

# Track new version
dvc add data/raw/sample_dataset.csv
git add data/raw/sample_dataset.csv.dvc
git commit -m "Dataset v1.1 - reduced to 500 samples"
dvc push

# Tag the version
git tag -a "data-v1.1" -m "Dataset version 1.1"
```

### 5.4 Retrieve Old Version

```bash
# Go back to version 1.0
git checkout data-v1.0

# Pull the old data
dvc pull

# Check it's the old version
wc -l data/raw/sample_dataset.csv
# Should show 1010 lines (original)

# Go back to latest
git checkout main
dvc pull
```

---

## 🔧 Part 6: Understanding the Components (Educational)

### 6.1 Data Flow

```
Raw Data (CSV)
    â†"
DataIngestion.ingest_from_csv()
    â†"
Save to data/raw/
    â†"
DataPreprocessor.run_pipeline()
    ├─ clean_data()
    ├─ encode_labels()
    └─ create_train_test_split()
    â†"
Save to data/processed/
    â†"
Create DataLoaders
    â†"
ModelTrainer.train()
    ├─ Log to MLflow
    └─ Save best model
    â†"
ModelEvaluator.evaluate()
    └─ Generate plots and metrics
```

### 6.2 MLflow Tracking

Every time you train a model, MLflow tracks:

```python
# Parameters (hyperparameters)
mlflow.log_params({
    'learning_rate': 0.001,
    'batch_size': 32,
    'model_name': 'resnet18'
})

# Metrics (per epoch)
mlflow.log_metrics({
    'train_loss': 0.45,
    'val_accuracy': 85.2
}, step=epoch)

# Artifacts (files)
mlflow.log_artifact('model.pth')
mlflow.log_artifact('confusion_matrix.png')
```

### 6.3 DVC Workflow

DVC works like Git for data:

```bash
# Add data (creates .dvc file)
dvc add data/raw/dataset.csv

# Commit the .dvc file to Git
git add data/raw/dataset.csv.dvc
git commit -m "Add dataset"

# Push actual data to remote storage
dvc push

# Later: retrieve data
dvc pull
```

---

## 🎯 Part 7: Next Steps

### 7.1 Create an Airflow DAG

Now that you understand the pipeline, automate it with Airflow:

1. Look at the `ml_pipeline_dag.py` stub in the documents
2. Implement each task function
3. Place it in the `dags/` folder
4. It will appear in the Airflow UI automatically

### 7.2 Run Real Experiments

1. Get a real dataset (e.g., from Kaggle)
2. Modify the code to handle real images
3. Run multiple experiments with different hyperparameters
4. Compare results in MLflow

### 7.3 Model Registry

Register your best model:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# Register model
model_uri = "runs:/YOUR_RUN_ID/model"
mlflow.register_model(model_uri, "image_classifier")

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="image_classifier",
    version=1,
    stage="Production"
)
```

---

## 📚 Common Issues & Solutions

### Issue: Services won't start

**Solution:**
```bash
# Check logs
docker-compose logs postgres
docker-compose logs mlflow

# Restart
docker-compose restart

# Clean restart
docker-compose down -v
docker-compose up -d
```

### Issue: Port already in use

**Solution:**
Edit `docker-compose.yml` and change port mappings:
```yaml
# Change from:
ports:
  - "5000:5000"

# To:
ports:
  - "5001:5000"
```

### Issue: MLflow can't connect to MinIO

**Solution:**
```bash
# Check MinIO is running
docker-compose ps minio

# Check buckets exist
docker exec -it mlpipeline-minio mc ls minio/
# Should show: mlflow/ and dvc/

# Recreate bucket
docker exec -it mlpipeline-minio mc mb minio/mlflow
```

### Issue: Python module not found

**Solution:**
```bash
# Make sure you're in the project root
pwd

# Make sure src/ has __init__.py
touch src/__init__.py

# Add src to path in your script
import sys
sys.path.insert(0, 'src')
```

---
