# Complete Quick Start Workflow - Run This to Test Everything!

"""
This script demonstrates the complete ML pipeline workflow:
1. Create sample dataset
2. Ingest data
3. Preprocess data
4. Train model with MLflow tracking
5. Evaluate model
6. Register model in MLflow

Run this after starting all Docker services!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from data_ingestion import DataIngestion
from preprocessing import DataPreprocessor
from training import MLflowTracker, ModelTrainer
from evaluation import ModelEvaluator

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# STEP 1: Create Sample Dataset
# ============================================================================

def create_sample_dataset(n_samples=1000, output_dir='data/raw'):
    """
    Create a sample image classification dataset.
    
    This creates a CSV with synthetic data for image classification.
    In a real project, you'd have actual images.
    """
    print("=" * 70)
    print("STEP 1: Creating Sample Dataset")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Define classes
    classes = ['cat', 'dog', 'bird', 'fish']
    
    # Generate synthetic data
    data = {
        'image_id': [f'img_{i:04d}' for i in range(n_samples)],
        'image_path': [f'images/{cls}/img_{i:04d}.jpg' 
                       for i, cls in enumerate(np.random.choice(classes, n_samples))],
        'label': np.random.choice(classes, n_samples),
        'width': np.random.randint(200, 500, n_samples),
        'height': np.random.randint(200, 500, n_samples),
        'split': np.random.choice(['train', 'val', 'test'], n_samples, p=[0.7, 0.15, 0.15])
    }
    
    df = pd.DataFrame(data)
    
    # Add some duplicates (to test cleaning)
    df = pd.concat([df, df.head(10)], ignore_index=True)
    
    # Add some missing values (to test cleaning)
    df.loc[np.random.choice(df.index, 20), 'label'] = np.nan
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_path / 'sample_dataset.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Created sample dataset with {len(df)} records")
    print(f"   Saved to: {csv_path}")
    print(f"   Classes: {classes}")
    print(f"   Class distribution:\n{df['label'].value_counts()}")
    
    return csv_path


# ============================================================================
# STEP 2: Ingest Data
# ============================================================================

def ingest_data(csv_path):
    """Ingest data using DataIngestion module."""
    print("\n" + "=" * 70)
    print("STEP 2: Data Ingestion")
    print("=" * 70)
    
    config = {
        'raw_data_path': 'data/raw',
        'retry_attempts': 3,
        'retry_delay': 5
    }
    
    ingestion = DataIngestion(config)
    
    # Ingest from CSV
    df = ingestion.ingest_from_csv(str(csv_path))
    
    # Save with metadata
    output_path = ingestion.save_raw_data(df, 'ingested_dataset.csv')
    
    print(f"✅ Data ingestion complete")
    print(f"   Records ingested: {len(df)}")
    print(f"   Output path: {output_path}")
    
    return df


# ============================================================================
# STEP 3: Preprocess Data
# ============================================================================

def preprocess_data(df):
    """Preprocess data using DataPreprocessor module."""
    print("\n" + "=" * 70)
    print("STEP 3: Data Preprocessing")
    print("=" * 70)
    
    config = {
        'processed_data_path': 'data/processed',
        'artifacts_path': 'artifacts',
        'required_columns': ['image_path', 'label'],
        'test_size': 0.2,
        'val_size': 0.1,
        'random_state': 42
    }
    
    preprocessor = DataPreprocessor(config)
    
    # Run complete pipeline
    train, val, test = preprocessor.run_pipeline(df, label_column='label')
    
    print(f"✅ Preprocessing complete")
    print(f"   Train: {len(train)} samples")
    print(f"   Val: {len(val)} samples")
    print(f"   Test: {len(test)} samples")
    
    return train, val, test


# ============================================================================
# STEP 4: Create Simple Dataset and DataLoaders
# ============================================================================

class SimpleImageDataset(Dataset):
    """
    Simple dataset for demonstration.
    In a real project, you'd load actual images.
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get label (already encoded)
        label = int(self.df.iloc[idx]['label_encoded'])
        
        # Create a dummy image (random tensor)
        # In real project: image = Image.open(image_path)
        image = torch.randn(3, 224, 224)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_dataloaders(train_df, val_df, test_df, batch_size=32):
    """Create PyTorch DataLoaders."""
    print("\n" + "=" * 70)
    print("STEP 4: Creating DataLoaders")
    print("=" * 70)
    
    # Simple transforms (for demonstration with random tensors)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SimpleImageDataset(train_df, transform=transform)
    val_dataset = SimpleImageDataset(val_df, transform=transform)
    test_dataset = SimpleImageDataset(test_df, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ DataLoaders created")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# STEP 5: Train Model with MLflow Tracking
# ============================================================================

def train_model(train_loader, val_loader, num_classes=4):
    """Train model with MLflow tracking."""
    print("\n" + "=" * 70)
    print("STEP 5: Model Training with MLflow")
    print("=" * 70)
    
    # Initialize MLflow tracker
    tracker = MLflowTracker(
        tracking_uri='http://localhost:5000',
        experiment_name='sample_experiment'
    )
    
    # Training configuration
    training_config = {
        'model_save_path': 'models'
    }
    
    # Training parameters
    params = {
        'model_name': 'mobilenet_v2',  # Changed from resnet18
        'num_epochs': 3, # Small for demo
        'learning_rate': 0.0001,  # Changed from 0.001
        'optimizer': 'adam',
        'lr_step_size': 2,
        'lr_gamma': 0.1,
        'early_stopping_patience': 2
    }
    
    # Initialize trainer
    trainer = ModelTrainer(training_config, tracker)
    
    # Train
    print("\n🚀 Starting training...")
    print("   (This is a quick demo with only 3 epochs)")
    
    model, best_val_acc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        params=params
    )
    
    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   View results at: http://localhost:5000")
    
    return model, best_val_acc


# ============================================================================
# STEP 6: Evaluate Model
# ============================================================================

def evaluate_model(model, test_loader):
    """Evaluate model on test set."""
    print("\n" + "=" * 70)
    print("STEP 6: Model Evaluation")
    print("=" * 70)
    
    config = {
        'plots_dir': 'evaluation_plots'
    }
    
    class_names = ['cat', 'dog', 'bird', 'fish']
    
    evaluator = ModelEvaluator(config, class_names)
    
    # Evaluate
    metrics = evaluator.evaluate(model, test_loader)
    
    print(f"\n✅ Evaluation complete!")
    print(f"\n📊 Test Metrics:")
    print(f"   Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"   Precision: {metrics['test_precision']:.4f}")
    print(f"   Recall:    {metrics['test_recall']:.4f}")
    print(f"   F1 Score:  {metrics['test_f1']:.4f}")
    
    print(f"\n   Plots saved to: evaluation_plots/")
    
    return metrics


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Run complete ML pipeline workflow."""
    print("\n" + "=" * 70)
    print("ML PIPELINE QUICK START WORKFLOW")
    print("=" * 70)
    print("\nThis will demonstrate the complete ML pipeline:")
    print("1. Create sample dataset")
    print("2. Ingest data")
    print("3. Preprocess data")
    print("4. Create DataLoaders")
    print("5. Train model with MLflow")
    print("6. Evaluate model")
    print("\n" + "=" * 70)
    
    try:
        # Step 1: Create sample dataset
        csv_path = create_sample_dataset(n_samples=1000)
        
        # Step 2: Ingest data
        df = ingest_data(csv_path)
        
        # Step 3: Preprocess data
        train_df, val_df, test_df = preprocess_data(df)
        
        # Step 4: Create DataLoaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df, batch_size=32
        )
        
        # Step 5: Train model
        model, best_val_acc = train_model(train_loader, val_loader, num_classes=4)
        
        # Step 6: Evaluate model
        metrics = evaluate_model(model, test_loader)
        
        # Final summary
        print("\n" + "=" * 70)
        print("🎉 PIPELINE COMPLETE!")
        print("=" * 70)
        print("\n✅ All steps executed successfully!")
        print("\n📊 Results:")
        print(f"   Best validation accuracy: {best_val_acc:.2f}%")
        print(f"   Test accuracy: {metrics['test_accuracy']:.4f}")
        
        print("\n🔗 View Results:")
        print("   MLflow UI: http://localhost:5000")
        print("   Airflow UI: http://localhost:8080")
        print("   MinIO Console: http://localhost:9001")
        
        print("\n📁 Outputs:")
        print("   Raw data: data/raw/")
        print("   Processed data: data/processed/")
        print("   Models: models/")
        print("   Artifacts: artifacts/")
        print("   Evaluation plots: evaluation_plots/")
        
        print("\n🎯 Next Steps:")
        print("1. Check MLflow UI for experiment tracking")
        print("2. Review evaluation plots")
        print("3. Create Airflow DAG to automate this pipeline")
        print("4. Version your data with DVC")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 70)
        print("SUCCESS! Your ML pipeline is working! 🎉")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("Something went wrong. Check the error above.")
        print("=" * 70)
