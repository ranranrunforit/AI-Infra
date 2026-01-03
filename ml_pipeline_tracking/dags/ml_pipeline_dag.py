"""
Airflow DAG for ML Training Pipeline

This DAG orchestrates the complete ML pipeline from data ingestion to model registration.

Learning Objectives:
- Design Airflow DAGs with proper task dependencies
- Implement PythonOperators for ML tasks
- Use XCom for inter-task communication
- Handle errors and retries
- Schedule pipelines

TODO: Complete all sections marked with TODO
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
from pathlib import Path
import logging



# TODO: Add project source to Python path
sys.path.insert(0, '/opt/airflow/src')


logger = logging.getLogger(__name__)


from data_ingestion import DataIngestion
from preprocessing import DataPreprocessor
from evaluation import ModelEvaluator
from training import MLflowTracker, ModelTrainer

import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ============================================================================
# DAG Configuration
# ============================================================================

# TODO: Define default_args for all tasks
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['cz78@illinois.edu'],  # TODO: Update with your email
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,  # TODO: Set appropriate retry count
    'retry_delay': timedelta(minutes=5),  # TODO: Set retry delay
    # TODO: Add execution_timeout
    'execution_timeout': timedelta(hours=2),
}

# TODO: Define pipeline configuration
PIPELINE_CONFIG = {
    'raw_data_path': '/opt/airflow/data/raw',
    'processed_data_path': '/opt/airflow/data/processed',
    'model_save_path': '/opt/airflow/models',
    'artifacts_path': '/opt/airflow/artifacts',
    'mlflow_tracking_uri': 'http://mlflow:5000',
    'plots_dir': '/opt/airflow/evaluation_plots',
    'experiment_name': 'airflow_ml_pipeline',
    'required_columns': ['image_path', 'label'],
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
}


# ============================================================================
# Task Functions
# ============================================================================

def ingest_data(**context):
    """
    Task: Ingest data from source.

    TODO:
    1. Import DataIngestion class
    2. Initialize with configuration
    3. Ingest data from CSV (or API, database)
    4. Save raw data
    5. Push data path to XCom for next task
    6. Return success message
    """
    logger.info("Starting data ingestion...")

    # TODO: Import DataIngestion
    
    # Initialize ingestion
    ingestion = DataIngestion(PIPELINE_CONFIG)
    
    # Check if source dataset exists, if not create sample data
    source_path = Path('/opt/airflow/data/source/dataset.csv')
    
    if not source_path.exists():
        logger.info("Source dataset not found. Creating sample dataset...")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create sample data
        
        np.random.seed(42)
        classes = ['cat', 'dog', 'bird', 'fish']
        n_samples = 1000
        
        data = {
            'image_id': [f'img_{i:04d}' for i in range(n_samples)],
            'image_path': [f'images/{cls}/img_{i:04d}.jpg' 
                          for i, cls in enumerate(np.random.choice(classes, n_samples))],
            'label': np.random.choice(classes, n_samples),
            'width': np.random.randint(200, 500, n_samples),
            'height': np.random.randint(200, 500, n_samples),
        }
        
        sample_df = pd.DataFrame(data)
        sample_df.to_csv(source_path, index=False)
        logger.info(f"Created sample dataset with {len(sample_df)} records")
    
    # Ingest data
    df = ingestion.ingest_from_csv(str(source_path))
    
    # Save raw data
    output_path = ingestion.save_raw_data(df, 'raw_dataset.csv')
    
    # Push path to XCom
    context['task_instance'].xcom_push(key='raw_data_path', value=str(output_path))
    
    logger.info(f"Data ingestion complete. Ingested {len(df)} records")
    return f"Data ingestion successful: {len(df)} records"


def validate_data(**context):
    """
    Task: Validate data quality with Great Expectations.

    TODO:
    1. Pull raw data path from XCom
    2. Load data
    3. Initialize DataValidator
    4. Create expectation suite
    5. Run validation
    6. Raise error if validation fails
    7. Return validation result
    """
    logger.info("Starting data validation...")

    # TODO: Pull data path from previous task
    raw_data_path = context['task_instance'].xcom_pull(
        task_ids='ingest_data',
        key='raw_data_path'
    )

    if not raw_data_path:
        raise ValueError("No raw data path found from previous task")
    
    # Load data
    df = pd.read_csv(raw_data_path)
    logger.info(f"Loaded {len(df)} records for validation")
    
    # Basic validation checks
    validation_passed = True
    issues = []
    
    # Check 1: Required columns present
    required_cols = PIPELINE_CONFIG['required_columns']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        validation_passed = False
    
    # Check 2: Minimum row count
    if len(df) < 100:
        issues.append(f"Too few records: {len(df)} < 100")
        validation_passed = False
    
    # Check 3: Check for null values in required columns
    if validation_passed:
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
    
    # Check 4: Label values are valid
    if 'label' in df.columns:
        unique_labels = df['label'].dropna().unique()
        logger.info(f"Found labels: {list(unique_labels)}")
        if len(unique_labels) < 2:
            issues.append(f"Too few unique labels: {len(unique_labels)}")
            validation_passed = False
    
    # Log validation results
    if validation_passed:
        logger.info("✅ Data validation passed all checks")
        return "Data validation successful"
    else:
        error_msg = f"Data validation failed: {'; '.join(issues)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def preprocess_data(**context):
    """
    Task: Preprocess data (clean, encode, split).

    TODO:
    1. Pull raw data path from XCom
    2. Load data
    3. Initialize DataPreprocessor
    4. Run preprocessing pipeline
    5. Push completion status to XCom
    6. Return success message
    """
    logger.info("Starting data preprocessing...")

    # TODO: Pull data path
    raw_data_path = context['task_instance'].xcom_pull(
        task_ids='ingest_data',
        key='raw_data_path'
    )

    # Load data
    df = pd.read_csv(raw_data_path)
    logger.info(f"Loaded {len(df)} records for preprocessing")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(PIPELINE_CONFIG)
    
    # Run pipeline
    train, val, test = preprocessor.run_pipeline(df, label_column='label')
    
    # Push status to XCom
    context['task_instance'].xcom_push(key='preprocessing_complete', value=True)
    context['task_instance'].xcom_push(key='train_size', value=len(train))
    context['task_instance'].xcom_push(key='val_size', value=len(val))
    context['task_instance'].xcom_push(key='test_size', value=len(test))
    
    # Get number of classes
    num_classes = len(train['label_encoded'].unique())
    context['task_instance'].xcom_push(key='num_classes', value=num_classes)
    
    logger.info(f"Preprocessing complete: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return f"Preprocessing successful: {num_classes} classes"


def version_data_dvc(**context):
    """
    Task: Version processed data with DVC.

    TODO:
    1. Run dvc add on processed data directory
    2. Commit DVC file to git
    3. Push to DVC remote
    4. Tag with version
    5. Return success message

    Note: This requires DVC and Git to be set up in the Airflow container
    """
    logger.info("Versioning data with DVC...")

    # Check if DVC is initialized
    dvc_dir = Path('/opt/airflow/.dvc')
    if not dvc_dir.exists():
        logger.warning("DVC not initialized. Skipping versioning.")
        logger.info("To enable DVC: run 'dvc init' in the Airflow container")
        return "DVC versioning skipped (not initialized)"
    
    try:
        # Add processed data to DVC
        subprocess.run(
            ['dvc', 'add', 'data/processed'],
            cwd='/opt/airflow',
            check=True,
            capture_output=True,
            text=True
        )
        
        # Push to remote (if configured)
        result = subprocess.run(
            ['dvc', 'push'],
            cwd='/opt/airflow',
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("DVC push successful")
        else:
            logger.warning(f"DVC push failed: {result.stderr}")
        
        logger.info("Data versioning complete")
        return "DVC versioning successful"
        
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC versioning failed: {e}")
        logger.warning("Continuing pipeline despite DVC failure")
        return "DVC versioning failed (non-critical)"


def train_model(**context):
    """
    Task: Train ML model with MLflow tracking.

    TODO:
    1. Initialize MLflowTracker
    2. Load preprocessed data
    3. Create data loaders
    4. Define training parameters
    5. Initialize ModelTrainer
    6. Run training
    7. Push best validation accuracy to XCom
    8. Return success message
    """
    logger.info("Starting model training...")

    # TODO: Import required classes
    from src.training import MLflowTracker, ModelTrainer
    import pandas as pd

    # Simple Dataset class for demo
    class SimpleDataset(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)
            self.transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            label = int(self.df.iloc[idx]['label_encoded'])
            # Create dummy image (in real project: load actual image)
            image = torch.randn(3, 224, 224)
            image = self.transform(image)
            return image, label
    
    # Initialize MLflow tracker
    tracker = MLflowTracker(
        tracking_uri=PIPELINE_CONFIG['mlflow_tracking_uri'],
        experiment_name=PIPELINE_CONFIG['experiment_name']
    )
    
    # Load processed data
    train_df = pd.read_csv(f"{PIPELINE_CONFIG['processed_data_path']}/train.csv")
    val_df = pd.read_csv(f"{PIPELINE_CONFIG['processed_data_path']}/val.csv")
    
    logger.info(f"Loaded train data: {len(train_df)} samples")
    logger.info(f"Loaded val data: {len(val_df)} samples")
    
    # Create datasets
    train_dataset = SimpleDataset(train_df)
    val_dataset = SimpleDataset(val_df)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Get number of classes from XCom
    num_classes = context['task_instance'].xcom_pull(
        task_ids='preprocess_data',
        key='num_classes'
    )
    
    # Define training parameters
    params = {
        'model_name': 'resnet18',
        'num_epochs': 3,  # Small for demo
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'lr_step_size': 2,
        'lr_gamma': 0.1,
        'early_stopping_patience': 2
    }
    
    logger.info(f"Training with parameters: {params}")
    
    # Initialize trainer
    trainer = ModelTrainer(PIPELINE_CONFIG, tracker)
    
    # Run training
    model, best_val_acc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        params=params
    )
    
    # Push metrics to XCom
    context['task_instance'].xcom_push(key='best_val_acc', value=best_val_acc)
    
    logger.info(f"Model training complete. Best val accuracy: {best_val_acc:.2f}%")
    return f"Training successful: {best_val_acc:.2f}% accuracy"


def evaluate_model(**context):
    """
    Task: Evaluate model on test set.

    TODO:
    1. Load test data
    2. Load best model
    3. Initialize ModelEvaluator
    4. Run evaluation
    5. Push test metrics to XCom
    6. Return success message
    """
    logger.info("Starting model evaluation...")

    # Simple Dataset class
    class SimpleDataset(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)
            self.transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            label = int(self.df.iloc[idx]['label_encoded'])
            image = torch.randn(3, 224, 224)
            image = self.transform(image)
            return image, label
    
    # Load test data
    test_df = pd.read_csv(f"{PIPELINE_CONFIG['processed_data_path']}/test.csv")
    logger.info(f"Loaded test data: {len(test_df)} samples")
    
    # Create test loader
    test_dataset = SimpleDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Load best model
    model_path = f"{PIPELINE_CONFIG['model_save_path']}/best_model.pth"
    
    # Get model architecture
    num_classes = context['task_instance'].xcom_pull(
        task_ids='preprocess_data',
        key='num_classes'
    )
    
    # Recreate model architecture
    import torchvision.models as models
    import torch.nn as nn
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    logger.info(f"Loaded model from {model_path}")
    
    # Get class names from label mapping
    import json
    label_mapping_path = Path(PIPELINE_CONFIG['artifacts_path']) / 'label_mapping.json'
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    
    class_names = sorted(label_mapping.keys(), key=lambda x: label_mapping[x])
    logger.info(f"Class names: {class_names}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(PIPELINE_CONFIG, class_names)
    
    # Run evaluation
    metrics = evaluator.evaluate(model, test_loader)
    
    # Push metrics to XCom
    context['task_instance'].xcom_push(key='test_metrics', value=metrics)
    
    logger.info(f"Model evaluation complete. Test accuracy: {metrics['test_accuracy']:.4f}")
    return f"Evaluation successful: {metrics['test_accuracy']:.4f} accuracy"



def register_model(**context):
    """
    Task: Register model in MLflow Model Registry if it meets criteria.

    TODO:
    1. Pull test metrics from XCom
    2. Check if model meets production criteria (e.g., accuracy >= 85%)
    3. If yes, register model in MLflow
    4. Transition to Staging stage
    5. Return registration result
    """
    logger.info("Starting model registration...")

    import mlflow
    
    # Set tracking URI
    mlflow.set_tracking_uri(PIPELINE_CONFIG['mlflow_tracking_uri'])
    
    # Pull test metrics
    test_metrics = context['task_instance'].xcom_pull(
        task_ids='evaluate_model',
        key='test_metrics'
    )
    
    if not test_metrics:
        logger.error("No test metrics found from evaluation task")
        return "Model registration failed: no metrics"
    
    test_accuracy = test_metrics.get('test_accuracy', 0)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    
    # Check production criteria
    accuracy_threshold = 0.70  # Lower threshold for demo
    
    if test_accuracy >= accuracy_threshold:
        logger.info(f"Model meets criteria ({test_accuracy:.4f} >= {accuracy_threshold})")
        
        # Get latest run ID
        experiment = mlflow.get_experiment_by_name(PIPELINE_CONFIG['experiment_name'])
        
        if not experiment:
            logger.error(f"Experiment '{PIPELINE_CONFIG['experiment_name']}' not found")
            return "Model registration failed: experiment not found"
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if len(runs) == 0:
            logger.error("No runs found in experiment")
            return "Model registration failed: no runs found"
        
        run_id = runs.iloc[0]['run_id']
        logger.info(f"Found run ID: {run_id}")
        
        # Register model
        model_uri = f"runs:/{run_id}/model"
        
        try:
            result = mlflow.register_model(
                model_uri=model_uri,
                name="image_classifier"
            )
            
            logger.info(f"Model registered: version {result.version}")
            
            # Transition to Staging
            client = mlflow.tracking.MlflowClient(PIPELINE_CONFIG['mlflow_tracking_uri'])
            client.transition_model_version_stage(
                name="image_classifier",
                version=result.version,
                stage="Staging"
            )
            
            logger.info(f"Model version {result.version} transitioned to Staging")
            
            return f"Model registered: version {result.version}"
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return f"Model registration failed: {str(e)}"
    else:
        logger.info(f"Model did not meet criteria ({test_accuracy:.4f} < {accuracy_threshold})")
        return f"Model not registered: accuracy {test_accuracy:.4f} < {accuracy_threshold}"



# ============================================================================
# DAG Definition
# ============================================================================

# TODO: Create the DAG
dag = DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='End-to-end ML training pipeline with MLflow tracking',
    # TODO: Set schedule (weekly on Sundays at midnight)
    schedule_interval='@weekly',
    start_date=days_ago(1),
    catchup=False,  # Don't run for past dates
    max_active_runs=1,  # Only one run at a time
    tags=['ml', 'training', 'production', 'mlops'],
)

# TODO: Define tasks
with dag:
    # Task 1: Ingest Data
    task_ingest = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data,
        # TODO: Add provide_context=True if using Airflow < 2.0
    )

    # Task 2: Validate Data
    task_validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
    )

    # Task 3: Preprocess Data
    task_preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    # Task 4: Version Data with DVC
    task_dvc = PythonOperator(
        task_id='version_data_dvc',
        python_callable=version_data_dvc,
    )

    # Task 5: Train Model
    task_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    # Task 6: Evaluate Model
    task_evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    # Task 7: Register Model
    task_register = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
    )

    # Task 8: Send Success Email
    task_notify = EmailOperator(
        task_id='send_success_email',
        to='mlops@example.com',  # TODO: Update email
        subject='[SUCCESS] ML Training Pipeline - {{ ds }}',
        html_content="""
        <h3>ML Training Pipeline Completed Successfully</h3>
        <p><strong>Execution Date:</strong> {{ ds }}</p>
        <p><strong>Status:</strong> SUCCESS</p>
        <p>View results in MLflow: <a href="http://mlflow:5000">MLflow UI</a></p>
        <p>View pipeline: <a href="http://airflow:8080/dags/ml_training_pipeline/grid">Airflow DAG</a></p>
        """,
    )

    # TODO: Define task dependencies
    # The pipeline should flow as:
    # ingest → validate → preprocess → version → train → evaluate → register → notify

    task_ingest >> task_validate >> task_preprocess >> task_dvc
    task_dvc >> task_train >> task_evaluate >> task_register >> task_notify


# ============================================================================
# DAG Testing (for local development)
# ============================================================================

if __name__ == "__main__":
    """
    Test the DAG structure without running tasks.

    TODO:
    1. Print DAG information
    2. Verify task dependencies
    3. Check for cycles
    """
    print(f"DAG: {dag.dag_id}")
    print(f"Schedule: {dag.schedule_interval}")
    print(f"Tasks: {len(dag.tasks)}")
    print("\nTask Dependencies:")
    for task in dag.tasks:
        print(f"  {task.task_id}:")
        print(f"    upstream: {task.upstream_task_ids}")
        print(f"    downstream: {task.downstream_task_ids}")