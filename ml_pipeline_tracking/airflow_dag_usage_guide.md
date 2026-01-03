# How to Use the ML Pipeline Airflow DAG

## 🎯 What This DAG Does

This Airflow DAG automates your entire ML pipeline:

```
Ingest Data → Validate → Preprocess → Version (DVC) → Train → Evaluate → Register Model
```

Each step runs automatically, with error handling, retries, and XCom communication between tasks.

---

## 📝 Changes I Made to Fix the Code

### Issues Fixed:

1. ✅ **Path configuration** - Changed to `/opt/airflow/src` (correct Airflow path)
2. ✅ **Missing imports** - Added all required imports in each function
3. ✅ **XCom handling** - Fixed data passing between tasks
4. ✅ **Dataset creation** - Added automatic sample dataset creation if source missing
5. ✅ **DataLoader implementation** - Created SimpleDataset class (was missing)
6. ✅ **Model loading** - Fixed model architecture recreation for evaluation
7. ✅ **Class names** - Properly load from label_mapping.json
8. ✅ **Error handling** - Added try-catch blocks and validation checks
9. ✅ **MLflow integration** - Fixed experiment lookup and model registration
10. ✅ **DVC handling** - Made it optional (skips if not initialized)
11. ✅ **Email notifications** - Disabled by default (removed as it needs SMTP setup)
12. ✅ **Reduced epochs** - Changed to 3 epochs for faster demo

---

## 🚀 Step-by-Step Setup

### Step 1: Copy the DAG File (5 minutes)

```bash
# Navigate to your project
cd ~/ml-pipeline-project

# Copy the corrected DAG to the dags folder
# (Copy the "Complete ML Pipeline DAG" artifact I just created)

# Make sure it's named correctly
ls dags/ml_pipeline_dag.py
```

### Step 2: Copy Python Modules to Airflow (5 minutes)

Make sure your `src/` folder has all the modules:

```bash
# Check your src folder
ls src/

# You should see:
# - __init__.py
# - data_ingestion.py
# - preprocessing.py
# - training.py
# - evaluation.py
```

**Important**: The Airflow container mounts `./src` to `/opt/airflow/src`, so these files are automatically available!

### Step 3: Verify Airflow is Running (2 minutes)

```bash
# Check services
docker-compose ps

# Should show:
# - mlpipeline-airflow-webserver    (Up)
# - mlpipeline-airflow-scheduler    (Up)
# - mlpipeline-airflow-worker       (Up)
# - mlpipeline-mlflow               (Up)
# - mlpipeline-postgres             (Up)

# If not running:
docker-compose up -d
```

### Step 4: Check DAG in Airflow UI (3 minutes)

1. Open **Airflow UI**: http://localhost:8080
2. Login: `admin` / `admin`
3. Wait 30-60 seconds (Airflow scans for DAGs every 30 seconds)
4. You should see **"ml_training_pipeline"** in the DAG list

**Troubleshooting:**
```bash
# If DAG doesn't appear, check logs:
docker-compose logs airflow-scheduler

# Look for errors like:
# - Import errors
# - Syntax errors
# - Module not found

# Manually trigger DAG scan:
docker exec -it mlpipeline-airflow-scheduler airflow dags list
```

---

## 🎮 Running the DAG

### Method 1: Manual Trigger (Recommended for First Run)

1. **Go to Airflow UI**: http://localhost:8080
2. **Find your DAG**: Look for `ml_training_pipeline`
3. **Click the DAG name** to open the DAG detail view
4. **Click the "Play" button** (▶️) in the top right
5. **Click "Trigger DAG"**

You'll see:
- The DAG starts running
- Tasks turn from gray → yellow (running) → green (success) or red (failed)
- A graph view shows the pipeline flow

### Method 2: Command Line Trigger

```bash
# Trigger the DAG manually
docker exec -it mlpipeline-airflow-scheduler \
    airflow dags trigger ml_training_pipeline

# You should see:
# Created <DagRun ml_training_pipeline @ 2025-01-03 ...>
```

### Method 3: Let It Run on Schedule

The DAG is scheduled to run **weekly on Sundays at midnight**. You can change this:

```python
# In ml_pipeline_dag.py, change schedule_interval:

# Every day at midnight
schedule_interval='@daily'

# Every hour
schedule_interval='@hourly'

# Custom cron (every Monday at 9 AM)
schedule_interval='0 9 * * 1'

# Never (only manual)
schedule_interval=None
```

---

## 📊 Monitoring DAG Execution

### Watch in Real-Time

1. **Graph View** (Default):
   - Click on DAG name → Graph
   - See task dependencies
   - Click tasks to see logs

2. **Grid View**:
   - Click on DAG name → Grid
   - See all historical runs
   - Quickly identify failed runs

3. **Gantt Chart**:
   - Click on DAG name → Gantt
   - See task durations
   - Identify bottlenecks

### Check Task Logs

1. Click on any **task box** (colored square)
2. Click **"Log"** button
3. See real-time task output

**Example log output:**
```
[2025-01-03 10:15:23] INFO - Starting data ingestion...
[2025-01-03 10:15:24] INFO - Loaded 1000 records for preprocessing
[2025-01-03 10:15:25] INFO - ✅ Data ingestion complete
```

### View Task Results

Click on task → **"XCom"** to see data passed between tasks:

```json
{
  "raw_data_path": "/opt/airflow/data/raw/raw_dataset.csv",
  "train_size": 700,
  "val_size": 150,
  "test_size": 150,
  "num_classes": 4,
  "best_val_acc": 85.2
}
```

---

## 🔍 Understanding Each Task

### Task 1: `ingest_data`

**What it does:**
- Checks for source data at `/opt/airflow/data/source/dataset.csv`
- If not found, creates sample dataset
- Loads and saves raw data
- Passes file path to next task via XCom

**Check results:**
```bash
# Inside Airflow container
docker exec -it mlpipeline-airflow-webserver bash

# Check raw data
ls /opt/airflow/data/raw/
# Should show: raw_dataset.csv, raw_dataset.csv.meta.json

# View first few lines
head /opt/airflow/data/raw/raw_dataset.csv
```

### Task 2: `validate_data`

**What it does:**
- Loads raw data
- Checks for required columns
- Validates minimum row count
- Checks for null values
- Raises error if validation fails

**What to check:**
- If this task fails, look at logs to see what failed
- Common issues: missing columns, too few rows, null values

### Task 3: `preprocess_data`

**What it does:**
- Cleans data (removes duplicates, null values)
- Encodes labels to integers
- Splits into train/val/test (70/15/15)
- Saves processed data

**Check results:**
```bash
# Check processed data
ls /opt/airflow/data/processed/
# Should show: train.csv, val.csv, test.csv + metadata

# Check sizes
wc -l /opt/airflow/data/processed/*.csv
```

### Task 4: `version_data_dvc`

**What it does:**
- Versions processed data with DVC
- Pushes to remote storage (MinIO)
- **Note**: This task is optional and will skip if DVC not initialized

**To enable DVC:**
```bash
# Inside Airflow container
docker exec -it mlpipeline-airflow-webserver bash

cd /opt/airflow
dvc init
dvc remote add -d minio s3://dvc
dvc remote modify minio endpointurl http://minio:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin123

git add .dvc
```

### Task 5: `train_model`

**What it does:**
- Creates DataLoaders
- Initializes MLflow tracker
- Trains model (ResNet18)
- Logs everything to MLflow
- Saves best model

**Check results:**

1. **MLflow UI**: http://localhost:5000
   - Go to experiment "airflow_ml_pipeline"
   - See training run with parameters and metrics

2. **Check model file:**
```bash
ls /opt/airflow/models/
# Should show: best_model.pth
```

### Task 6: `evaluate_model`

**What it does:**
- Loads test data
- Loads best model
- Computes metrics (accuracy, precision, recall, F1)
- Generates confusion matrix
- Saves evaluation plots

**Check results:**
```bash
# Check evaluation outputs
ls /opt/airflow/evaluation_plots/
# Should show: confusion_matrix.png, classification_report.txt, test_metrics.json

# View metrics
cat /opt/airflow/evaluation_plots/test_metrics.json
```

### Task 7: `register_model`

**What it does:**
- Checks if model meets accuracy threshold (70%)
- If yes, registers model in MLflow Model Registry
- Transitions to "Staging" stage

**Check results:**

Go to **MLflow UI** → **Models** tab:
- You should see "image_classifier"
- Click it to see version 1 in "Staging" stage

---

## 🐛 Troubleshooting

### Issue 1: DAG Not Appearing

**Symptoms:** DAG doesn't show up in Airflow UI

**Solution:**
```bash
# Check for import errors
docker exec -it mlpipeline-airflow-scheduler \
    airflow dags list-import-errors

# Check scheduler logs
docker-compose logs airflow-scheduler | grep -i error

# Manually parse DAG
docker exec -it mlpipeline-airflow-scheduler \
    python /opt/airflow/dags/ml_pipeline_dag.py
```

### Issue 2: Task Failed with Import Error

**Symptoms:** Task fails with "ModuleNotFoundError: No module named 'data_ingestion'"

**Solution:**
```bash
# Check src files are mounted
docker exec -it mlpipeline-airflow-webserver ls /opt/airflow/src

# Should show:
# __init__.py  data_ingestion.py  preprocessing.py  training.py  evaluation.py

# If not, restart Airflow:
docker-compose restart airflow-webserver airflow-scheduler airflow-worker
```

### Issue 3: MLflow Connection Failed

**Symptoms:** Task fails with "ConnectionError: MLflow tracking server not accessible"

**Solution:**
```bash
# Check MLflow is running
docker-compose ps mlflow

# Test connection from Airflow
docker exec -it mlpipeline-airflow-webserver \
    curl http://mlflow:5000/health

# Should return: {"status": "ok"}
```

### Issue 4: Task Stuck in "Running"

**Symptoms:** Task stays yellow (running) for a long time

**Solution:**
1. Check task logs (click task → Log)
2. Look for what it's waiting on
3. Check resource usage:

```bash
# Check Docker resources
docker stats

# If high CPU/memory, might just be slow
# If 0%, might be stuck - restart worker:
docker-compose restart airflow-worker
```

### Issue 5: All Tasks Failed

**Symptoms:** All tasks turn red

**Solution:**
1. **Click first failed task** (usually ingest_data)
2. **Read the logs carefully**
3. Common causes:
   - Missing files/directories
   - Permission errors
   - Python syntax errors in your code

```bash
# Check Airflow logs
docker-compose logs airflow-worker

# Restart everything if needed
docker-compose restart
```

---

## 🎨 Customizing the DAG

### Change Schedule

```python
# In ml_pipeline_dag.py:

# Run daily at 2 AM
schedule_interval='0 2 * * *'

# Run every Monday at 9 AM
schedule_interval='0 9 * * 1'

# Run every hour
schedule_interval='@hourly'

# Never (manual only)
schedule_interval=None
```

### Change Training Parameters

```python
# In train_model() function, modify:

params = {
    'model_name': 'mobilenet_v2',  # Try different model
    'num_epochs': 10,               # More epochs
    'batch_size': 64,               # Larger batches
    'learning_rate': 0.0001,        # Lower learning rate
    'optimizer': 'sgd',             # Different optimizer
}
```

### Add Email Notifications

1. Configure SMTP in `docker-compose.yml`:

```yaml
airflow-webserver:
  environment:
    AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
    AIRFLOW__SMTP__SMTP_PORT: 587
    AIRFLOW__SMTP__SMTP_USER: your-email@gmail.com
    AIRFLOW__SMTP__SMTP_PASSWORD: your-app-password
    AIRFLOW__SMTP__SMTP_MAIL_FROM: your-email@gmail.com
```

2. Enable in DAG:

```python
default_args = {
    'email': ['your-email@example.com'],
    'email_on_failure': True,  # Change to True
    'email_on_retry': False,
}
```

### Add More Tasks

Example: Add a data augmentation task:

```python
def augment_data(**context):
    """Augment training data."""
    logger.info("Starting data augmentation...")
    # Your augmentation code here
    return "Augmentation successful"

# In the DAG:
task_augment = PythonOperator(
    task_id='augment_data',
    python_callable=augment_data,
)

# Update dependencies:
task_preprocess >> task_augment >> task_dvc
```

---

## 📊 Viewing Results

### 1. Check MLflow Experiments

http://localhost:5000

- **Experiments** → "airflow_ml_pipeline"
- See all parameters logged
- See metrics charts (loss, accuracy over epochs)
- Download model artifacts

### 2. Check Model Registry

http://localhost:5000 → **Models** tab

- See "image_classifier" model
- View versions and stages (None/Staging/Production)
- Compare model versions

### 3. Check Evaluation Plots

```bash
# From your local machine:
ls evaluation_plots/

# Or inside container:
docker exec -it mlpipeline-airflow-webserver \
    ls /opt/airflow/evaluation_plots/

# Copy plots to your machine:
docker cp mlpipeline-airflow-webserver:/opt/airflow/evaluation_plots ./
```

---

## 🎯 Next Steps

### 1. Run with Real Data

Replace the sample dataset with real images:

```bash
# Create source directory
mkdir -p data/source

# Add your dataset
cp your-dataset.csv data/source/dataset.csv

# Make sure it has these columns:
# - image_path
# - label
```

### 2. Improve the Model

- Try different architectures (in `train_model()`)
- Tune hyperparameters
- Run multiple experiments
- Compare in MLflow

### 3. Set Up Production Deployment

- Promote model from Staging → Production in MLflow
- Create deployment DAG (separate from training)
- Set up monitoring

### 4. Add Advanced Features

- **A/B testing**: Train two models, compare
- **Model drift detection**: Monitor performance over time
- **Feature engineering**: Add feature engineering task
- **Hyperparameter tuning**: Add Optuna task

---

## ✅ Quick Reference Commands

```bash
# List all DAGs
docker exec -it mlpipeline-airflow-scheduler airflow dags list

# Trigger DAG manually
docker exec -it mlpipeline-airflow-scheduler \
    airflow dags trigger ml_training_pipeline

# Pause DAG (stop automatic runs)
docker exec -it mlpipeline-airflow-scheduler \
    airflow dags pause ml_training_pipeline

# Unpause DAG
docker exec -it mlpipeline-airflow-scheduler \
    airflow dags unpause ml_training_pipeline

# View task logs
docker exec -it mlpipeline-airflow-scheduler \
    airflow tasks logs ml_training_pipeline ingest_data 2025-01-03

# Clear failed task (to rerun)
docker exec -it mlpipeline-airflow-scheduler \
    airflow tasks clear ml_training_pipeline --task-regex train_model

# Check DAG dependencies
docker exec -it mlpipeline-airflow-scheduler \
    airflow dags show ml_training_pipeline
```

---

## 🎓 Summary

You now have a **production-grade automated ML pipeline**! 

The DAG:
- ✅ Runs automatically on schedule
- ✅ Tracks all experiments in MLflow
- ✅ Versions data with DVC
- ✅ Handles errors with retries
- ✅ Passes data between tasks via XCom
- ✅ Registers successful models
- ✅ Can be monitored in real-time

**Your pipeline is now:**
1. **Reproducible** - Same input → Same output
2. **Automated** - No manual steps
3. **Monitored** - Real-time visibility
4. **Production-ready** - Error handling, retries, logging

🎉 **Congratulations! You've built a complete MLOps pipeline!**
