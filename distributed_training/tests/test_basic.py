"""
Basic tests for distributed training components
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.resnet import create_resnet_model, resnet18, resnet34, resnet50
from data.data_loader import create_distributed_dataloader, DatasetFactory
from training.checkpointing import CheckpointManager
from utils.metrics import MetricsTracker, AverageMeter, accuracy
from utils.gpu_monitor import get_gpu_info


class TestModels:
    """Test model creation and forward pass"""
    
    def test_resnet18_creation(self):
        model = resnet18(num_classes=10)
        assert isinstance(model, nn.Module)
    
    def test_resnet34_creation(self):
        model = resnet34(num_classes=100)
        assert isinstance(model, nn.Module)
    
    def test_resnet50_creation(self):
        model = resnet50(num_classes=1000)
        assert isinstance(model, nn.Module)
    
    def test_model_forward_pass(self):
        model = resnet18(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_gradient_checkpointing(self):
        model = create_resnet_model(
            'resnet18',
            num_classes=10,
            use_gradient_checkpointing=True
        )
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_model_parameters(self):
        model = resnet18(num_classes=10)
        params = sum(p.numel() for p in model.parameters())
        assert params > 1_000_000  # Should have at least 1M parameters


class TestDataLoaders:
    """Test data loading functionality"""
    
    def test_cifar10_dataset(self):
        dataset, num_classes, image_shape = DatasetFactory.get_dataset(
            'cifar10', './data', train=True, download=True
        )
        assert num_classes == 10
        assert image_shape == (3, 32, 32)
        assert len(dataset) == 50000
    
    def test_dataloader_creation(self):
        loader, num_classes, image_shape = create_distributed_dataloader(
            dataset_name='cifar10',
            data_path='./data',
            batch_size=32,
            num_workers=0,  # Use 0 workers for testing
            train=True,
            world_size=1,
            rank=0
        )
        assert num_classes == 10
        assert len(loader) > 0
    
    def test_dataloader_batch(self):
        loader, _, _ = create_distributed_dataloader(
            dataset_name='cifar10',
            data_path='./data',
            batch_size=32,
            num_workers=0,
            train=True,
            world_size=1,
            rank=0
        )
        
        images, labels = next(iter(loader))
        assert images.shape == (32, 3, 32, 32)
        assert labels.shape == (32,)


class TestCheckpointing:
    """Test checkpoint management"""
    
    def test_checkpoint_manager_creation(self, tmp_path):
        manager = CheckpointManager(str(tmp_path), keep_last_n=3)
        assert manager.checkpoint_dir.exists()
    
    def test_checkpoint_save_load(self, tmp_path):
        manager = CheckpointManager(str(tmp_path), keep_last_n=3)
        
        # Save checkpoint
        state = {
            'model': {'weight': torch.randn(10)},
            'optimizer': {},
        }
        path = manager.save_checkpoint(state, step=100, metric=0.95)
        assert path.exists()
        
        # Load checkpoint
        loaded_state = manager.load_checkpoint(str(path))
        assert 'model' in loaded_state
        assert 'step' in loaded_state
        assert loaded_state['step'] == 100
    
    def test_checkpoint_cleanup(self, tmp_path):
        manager = CheckpointManager(str(tmp_path), keep_last_n=2)
        
        # Save multiple checkpoints
        for i in range(5):
            state = {'model': {}, 'step': i}
            manager.save_checkpoint(state, step=i)
        
        # Should only keep last 2
        checkpoints = list(tmp_path.glob("checkpoint_*.pt"))
        # +1 for best.pt if enabled
        assert len(checkpoints) <= 3


class TestMetrics:
    """Test metrics tracking"""
    
    def test_average_meter(self):
        meter = AverageMeter()
        meter.update(1.0)
        meter.update(2.0)
        meter.update(3.0)
        assert meter.avg == 2.0
    
    def test_accuracy_calculation(self):
        output = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        target = torch.tensor([1, 0, 1])
        acc = accuracy(output, target, topk=(1,))
        assert acc[0] == 100.0  # All predictions correct
    
    def test_metrics_tracker(self, tmp_path):
        tracker = MetricsTracker(
            log_dir=str(tmp_path),
            use_tensorboard=False
        )
        
        metrics = {'loss': 1.5, 'accuracy': 85.0}
        tracker.update(metrics, step=1)
        
        assert tracker.get_metric('loss') == 1.5
        assert tracker.get_metric('accuracy') == 85.0
        
        tracker.close()


class TestGPUMonitoring:
    """Test GPU monitoring (skips if no GPU)"""
    
    def test_get_gpu_info(self):
        gpu_info = get_gpu_info()
        # This will be empty list if no GPU, which is fine
        assert isinstance(gpu_info, list)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_gpu_monitoring_with_gpu(self):
        from utils.gpu_monitor import GPUMonitor
        monitor = GPUMonitor(device_id=0)
        metrics = monitor.get_gpu_metrics(force=True)
        
        if metrics:
            assert 'memory_allocated_mb' in metrics
            assert metrics['memory_allocated_mb'] >= 0


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield
    # Cleanup GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
