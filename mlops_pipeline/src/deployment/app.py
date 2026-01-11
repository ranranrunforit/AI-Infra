"""FastAPI model serving endpoint."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
from typing import List, Dict
import logging
import os
from datetime import datetime  # <--- Add this line

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Model Serving API",
    description="ML model serving endpoint for churn prediction",
    version="1.0.0"
)

# Global model variable
model = None
model_metadata = {}


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[List[float]]
    
    class Config:
        schema_extra = {
            "example": {
                "features": [[0.5, 1.2, 0.8, 1.5, 0.3, 0.7, 1.1, 0.9, 
                             0.4, 0.6, 1.3, 0.2, 0.8, 1.0, 0.5, 0.9, 1.2, 0.7]]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[int]
    probabilities: List[List[float]] = None
    model_version: str = None


@app.on_event("startup")
async def load_model():
    """Load model from MLflow on startup."""
    global model, model_metadata
    
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    model_name = os.getenv('MODEL_NAME', 'churn-classifier')
    model_version = os.getenv('MODEL_VERSION', 'latest')
    
    logger.info(f"Loading model from MLflow...")
    logger.info(f"MLflow URI: {mlflow_uri}")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Model Version: {model_version}")
    
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Try to load model by version or stage
        if model_version.isdigit():
            model_uri = f"models:/{model_name}/{model_version}"
        elif model_version in ['Production', 'Staging', 'None']:
            model_uri = f"models:/{model_name}/{model_version}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        logger.info(f"Loading model from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        
        model_metadata = {
            'name': model_name,
            'version': model_version,
            'uri': model_uri,
            'loaded_at': str(datetime.now())
        }
        
        logger.info(f"✓ Model loaded successfully: {model_name}")
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        logger.warning("Application will start but predictions will fail until model is loaded")
        model_metadata = {
            'name': model_name,
            'version': model_version,
            'error': str(e),
            'loaded': False
        }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "model-serving-api",
        "timestamp": str(datetime.now())
    }


@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    if model is None:
        return {
            "status": "not_ready",
            "reason": "Model not loaded",
            "model_metadata": model_metadata
        }
    
    return {
        "status": "ready",
        "model_loaded": True,
        "model_metadata": model_metadata
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Model Serving API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "predict": "/predict",
            "model_info": "/model/info"
        }
    }


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    return {
        "model_metadata": model_metadata,
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions using the loaded model.
    
    Args:
        request: PredictionRequest containing features
        
    Returns:
        PredictionResponse with predictions and probabilities
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /ready endpoint for details."
        )
    
    try:
        # Convert input to numpy array
        features = np.array(request.features)
        
        # Make predictions
        predictions = model.predict(features).tolist()
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features).tolist()
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_version=model_metadata.get('version', 'unknown')
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)