"""Model training pipeline with MLflow tracking."""

from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import pandas as pd
import time

# Add src to path
import sys
sys.path.insert(0, '/opt/airflow')

from src.training.trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator
from src.training.registry import ModelRegistry
from src.common.config import config
from src.common.logger import get_logger
from src.monitoring.metrics_collector import MetricsCollector

logger = get_logger(__name__)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


@dag(
    dag_id='training_pipeline',
    default_args=default_args,
    description='Model training pipeline with MLflow tracking',
    schedule_interval=None,  # Triggered by data pipeline
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['training', 'mlops'],
)
def training_pipeline():
    """Training pipeline DAG."""

    @task()
    def load_data() -> dict:
        """
        Load processed data.

        Returns:
            Dictionary with data info
        """
        logger.info("Loading processed data")

        try:
            from src.data.preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()

            # Load processed data
            X_train = pd.read_csv(preprocessor.processed_data_path / 'X_train.csv')
            X_test = pd.read_csv(preprocessor.processed_data_path / 'X_test.csv')
            y_train = pd.read_csv(preprocessor.processed_data_path / 'y_train.csv')['Churn']
            y_test = pd.read_csv(preprocessor.processed_data_path / 'y_test.csv')['Churn']

            logger.info(
                f"Loaded data - Train: {len(X_train)} rows x {X_train.shape[1]} features, "
                f"Test: {len(X_test)} rows"
            )
            logger.info(f"Train data shape: {X_train.shape}")
            logger.info(f"Test data shape: {X_test.shape}")
            logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")

            return {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': X_train.shape[1]
            }

        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            raise

    @task()
    def train_baseline_models(data_info: dict) -> dict:
        """
        Train baseline models.

        Args:
            data_info: Information about the data

        Returns:
            Dictionary with training results
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 60)
        logger.info(f"Data info: {data_info}")
        start_time = time.time()

        try:
            from src.data.preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()

            # Load data
            logger.info("Loading training and test data...")
            X_train = pd.read_csv(preprocessor.processed_data_path / 'X_train.csv')
            X_test = pd.read_csv(preprocessor.processed_data_path / 'X_test.csv')
            y_train = pd.read_csv(preprocessor.processed_data_path / 'y_train.csv')['Churn']
            y_test = pd.read_csv(preprocessor.processed_data_path / 'y_test.csv')['Churn']

            logger.info(f"Data loaded successfully")
            logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

            # Train models
            logger.info("Initializing ModelTrainer...")
            trainer = ModelTrainer()
            
            logger.info("Starting model training for all baseline models...")
            results = trainer.train_all_models(X_train, y_train, X_test, y_test)

            logger.info(f"Training completed. Number of models trained: {len(results)}")
            logger.info(f"Model names: {list(results.keys())}")

            if not results:
                error_msg = "No models were trained successfully. Check ModelTrainer implementation."
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Collect results
            training_results = {}
            logger.info("Evaluating trained models...")
            
            for model_name, (model, run_id) in results.items():
                logger.info(f"Evaluating model: {model_name} (run_id: {run_id})")
                
                evaluator = ModelEvaluator()
                metrics = evaluator.evaluate_model(model, X_test, y_test, model_name)

                logger.info(f"Model {model_name} metrics: {metrics}")

                training_results[model_name] = {
                    'run_id': run_id,
                    'metrics': metrics
                }

                # Record metrics
                metrics_collector = MetricsCollector()
                metrics_collector.record_model_training(
                    model_type=model_name,
                    duration=time.time() - start_time,
                    metrics=metrics,
                    model_name=config.MODEL_NAME,
                    model_version=run_id
                )

            duration = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"TRAINING COMPLETED: {len(results)} models in {duration:.2f}s")
            logger.info(f"Results summary: {[(k, v['metrics'].get('f1_score', 0)) for k, v in training_results.items()]}")
            logger.info("=" * 60)

            return training_results

        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
            raise

    @task()
    def select_best_model(training_results: dict) -> dict:
        """
        Select best model based on metrics.

        Args:
            training_results: Training results for all models

        Returns:
            Best model information
        """
        logger.info("=" * 60)
        logger.info("SELECTING BEST MODEL")
        logger.info("=" * 60)
        logger.info(f"Training results received: {list(training_results.keys())}")

        try:
            if not training_results:
                error_msg = "No training results provided. Cannot select best model."
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Find model with highest F1 score
            best_model_name = None
            best_f1_score = 0
            best_run_id = None

            for model_name, result in training_results.items():
                metrics = result.get('metrics', {})
                f1_score = metrics.get('f1_score', 0)
                
                logger.info(f"Model: {model_name}, F1 Score: {f1_score:.4f}")
                
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_model_name = model_name
                    best_run_id = result['run_id']

            if best_model_name is None:
                error_msg = "Could not find any model with valid F1 score"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info("=" * 60)
            logger.info(f"BEST MODEL SELECTED: {best_model_name}")
            logger.info(f"F1 Score: {best_f1_score:.4f}")
            logger.info(f"Run ID: {best_run_id}")
            logger.info("=" * 60)

            return {
                'model_name': best_model_name,
                'run_id': best_run_id,
                'f1_score': best_f1_score,
                'metrics': training_results[best_model_name]['metrics']
            }

        except Exception as e:
            logger.error(f"Model selection failed: {e}", exc_info=True)
            raise

    @task()
    def register_best_model(best_model_info: dict) -> dict:
        """
        Register best model to MLflow Model Registry.

        Args:
            best_model_info: Information about best model

        Returns:
            Model registry information
        """
        logger.info("=" * 60)
        logger.info("REGISTERING BEST MODEL")
        logger.info("=" * 60)
        logger.info(f"Model info: {best_model_info}")

        try:
            registry = ModelRegistry()

            # Register model
            logger.info(f"Registering model with run_id: {best_model_info['run_id']}")
            version = registry.register_model(
                run_id=best_model_info['run_id'],
                tags={
                    'model_type': best_model_info['model_name'],
                    'f1_score': str(best_model_info['f1_score'])
                }
            )

            logger.info(f"Model registered as version {version}")

            # Transition to Staging
            logger.info("Transitioning model to Staging...")
            registry.transition_model_stage(
                model_name=config.MODEL_NAME,
                version=version,
                stage='Staging'
            )

            logger.info("=" * 60)
            logger.info(f"MODEL REGISTRATION COMPLETED")
            logger.info(f"Model: {config.MODEL_NAME}")
            logger.info(f"Version: {version}")
            logger.info(f"Stage: Staging")
            logger.info("=" * 60)

            return {
                'model_name': config.MODEL_NAME,
                'version': version,
                'stage': 'Staging',
                'run_id': best_model_info['run_id']
            }

        except Exception as e:
            logger.error(f"Model registration failed: {e}", exc_info=True)
            raise

    @task()
    def validate_model_performance(registry_info: dict) -> bool:
        """
        Validate model performance against thresholds.

        Args:
            registry_info: Model registry information

        Returns:
            True if validation passed
        """
        logger.info("=" * 60)
        logger.info("VALIDATING MODEL PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"Registry info: {registry_info}")

        try:
            from src.data.preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()
            evaluator = ModelEvaluator()
            registry = ModelRegistry()

            # Load test data
            logger.info("Loading test data...")
            X_test = pd.read_csv(preprocessor.processed_data_path / 'X_test.csv')
            y_test = pd.read_csv(preprocessor.processed_data_path / 'y_test.csv')['Churn']
            logger.info(f"Test data loaded: {X_test.shape}")

            # Load model from registry
            logger.info(f"Loading model {registry_info['model_name']} version {registry_info['version']}")
            model = registry.load_model(
                model_name=registry_info['model_name'],
                version=registry_info['version']
            )

            # Evaluate
            logger.info("Evaluating model...")
            metrics = evaluator.evaluate_model(model, X_test, y_test)
            logger.info(f"Evaluation metrics: {metrics}")

            # Check thresholds
            logger.info("Checking performance thresholds...")
            meets_thresholds, failed_metrics = evaluator.check_model_thresholds(metrics)

            if not meets_thresholds:
                logger.warning("=" * 60)
                logger.warning("MODEL FAILED PERFORMANCE THRESHOLDS")
                logger.warning(f"Failed metrics: {failed_metrics}")
                logger.warning("=" * 60)
                return False

            logger.info("=" * 60)
            logger.info("MODEL PASSED PERFORMANCE VALIDATION")
            logger.info("=" * 60)
            return True

        except Exception as e:
            logger.error(f"Model validation failed: {e}", exc_info=True)
            return False

    @task.branch()
    def decide_promotion(validation_passed: bool) -> str:
        """
        Decide whether to promote model to production.

        Args:
            validation_passed: Whether validation passed

        Returns:
            Task ID to execute next
        """
        logger.info("=" * 60)
        logger.info("DEPLOYMENT DECISION")
        logger.info("=" * 60)
        
        if validation_passed:
            logger.info("✓ Model APPROVED for production deployment")
            logger.info("=" * 60)
            return 'notify_success'
        else:
            logger.warning("✗ Model NOT approved for production")
            logger.warning("=" * 60)
            return 'skip_deployment'

    @task()
    def notify_success(registry_info: dict):
        """Notify that model is ready for deployment."""
        logger.info("=" * 60)
        logger.info("MODEL READY FOR DEPLOYMENT")
        logger.info("=" * 60)
        logger.info(f"Model: {registry_info['model_name']}")
        logger.info(f"Version: {registry_info['version']}")
        logger.info(f"Stage: {registry_info['stage']}")
        logger.info("Deployment pipeline will be triggered automatically")
        logger.info("=" * 60)
        return "ready_for_deployment"

    @task()
    def skip_deployment():
        """Skip deployment."""
        logger.info("=" * 60)
        logger.info("DEPLOYMENT SKIPPED")
        logger.info("Model did not meet performance requirements")
        logger.info("=" * 60)
        return "deployment_skipped"

    # Define task dependencies
    data_info = load_data()
    training_results = train_baseline_models(data_info)
    best_model = select_best_model(training_results)
    registry_info = register_best_model(best_model)
    validation_passed = validate_model_performance(registry_info)
    promotion_decision = decide_promotion(validation_passed)

    # Branch: either notify success or skip deployment
    success_notification = notify_success(registry_info)
    skip_notification = skip_deployment()
    
    promotion_decision >> [success_notification, skip_notification]
    
    # Trigger deployment pipeline after successful validation
    trigger_deployment = TriggerDagRunOperator(
        task_id='trigger_deployment_pipeline',
        trigger_dag_id='deployment_pipeline',
        wait_for_completion=False,
        conf={'model_info': '{{ ti.xcom_pull(task_ids="register_best_model") }}'},
    )
    
    # Connect trigger to success notification
    success_notification >> trigger_deployment


# Instantiate the DAG
training_pipeline_dag = training_pipeline()