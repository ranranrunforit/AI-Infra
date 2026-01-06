"""
Script to train and register a simple model in MLflow
Run this once to set up your model in MLflow
"""

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure MLflow
MLFLOW_TRACKING_URI = "http://mlflow-server:5001"
MODEL_NAME = "image-classifier"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def create_simple_model():
    """
    Create a simple ResNet18 model
    In production, this would be your trained model
    """
    logger.info("Creating ResNet18 model...")
    
    # Use pretrained ResNet18 (you can replace with your trained model)
    model = models.resnet18(pretrained=True)
    model.eval()
    
    logger.info("Model created successfully")
    return model

def register_model_in_mlflow():
    """
    Register the model in MLflow Model Registry
    """
    logger.info("Starting model registration process...")
    
    # Start MLflow run
    with mlflow.start_run(run_name="image-classifier-v1") as run:
        
        # Create model
        model = create_simple_model()
        
        # Log parameters
        mlflow.log_param("model_type", "resnet18")
        mlflow.log_param("pretrained", True)
        mlflow.log_param("num_classes", 1000)
        
        # Log metrics (in real scenario, these would be validation metrics)
        mlflow.log_metric("accuracy", 0.85)
        mlflow.log_metric("precision", 0.83)
        mlflow.log_metric("recall", 0.84)
        
        # Create a sample input for signature
        sample_input = torch.randn(1, 3, 224, 224)
        
        # Log the model
        logger.info("Logging model to MLflow...")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=mlflow.models.infer_signature(
                sample_input.numpy(),
                model(sample_input).detach().numpy()
            )
        )
        
        logger.info(f"Model logged with run_id: {run.info.run_id}")
    
    # Get the latest version
    client = mlflow.tracking.MlflowClient()
    
    # Wait a moment for registration to complete
    import time
    time.sleep(2)
    
    # Get all versions
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    
    if versions:
        latest_version = max(versions, key=lambda v: int(v.version))
        logger.info(f"Latest model version: {latest_version.version}")
        
        # Transition to Production stage
        logger.info("Transitioning model to Production stage...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=False
        )
        
        logger.info(f"✅ Model '{MODEL_NAME}' version {latest_version.version} is now in Production!")
        logger.info(f"Model URI: models:/{MODEL_NAME}/Production")
        
    else:
        logger.error("No model versions found!")

def verify_model():
    """
    Verify the model can be loaded
    """
    logger.info("\nVerifying model can be loaded...")
    
    try:
        # Load model from registry
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.pytorch.load_model(model_uri)
        
        # Test prediction
        sample_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(sample_input)
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"Output shape: {output.shape}")
        logger.info(f"Model is ready to use!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("MLflow Model Registration Script")
    print("=" * 70)
    print()
    print("This script will:")
    print("1. Create a ResNet18 model")
    print("2. Register it in MLflow")
    print("3. Transition it to Production stage")
    print()
    print("Make sure MLflow server is running at http://127.0.0.1:5001")
    print()
    
    # Check if MLflow is accessible
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        client.search_experiments()
        print("✅ MLflow server is accessible")
        print()
    except Exception as e:
        print(f"❌ Cannot connect to MLflow server: {e}")
        print("\nPlease start MLflow server first:")
        print("  mlflow server --host 127.0.0.1 --port 5001")
        exit(1)
    
    # Register the model
    try:
        register_model_in_mlflow()
        print()
        
        # Verify
        if verify_model():
            print()
            print("=" * 70)
            print("SUCCESS! Model is registered and ready to use")
            print("=" * 70)
            print()
            print("You can now:")
            print("1. View your model in MLflow UI: http://localhost:5001")
            print("2. Start your ML API: python src/main.py")
            print()
        
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        import traceback
        traceback.print_exc()
