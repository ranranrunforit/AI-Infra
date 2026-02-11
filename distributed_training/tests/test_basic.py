"""
Comprehensive tests for the distributed training project.

Tests all core modules:
- Models (ResNet, ViT)
- Data loading (DatasetFactory, transforms)
- Checkpointing (save/load/cleanup)
- Metrics (AverageMeter, accuracy, MetricsTracker)
- GPU monitoring
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.resnet import create_resnet_model, resnet18, resnet34, resnet50
from data.data_loader import create_distributed_dataloader, DatasetFactory
from training.checkpointing import CheckpointManager
from utils.metrics import MetricsTracker, AverageMeter, accuracy
from utils.gpu_monitor import get_gpu_info, GPUMonitor


# ─────────────────────────────────────────────────────────────
#  Model Tests
# ─────────────────────────────────────────────────────────────

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
    
    def test_model_forward_pass_224(self):
        """Test forward pass with ImageNet-sized input"""
        model = resnet18(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_model_forward_pass_32(self):
        """Test forward pass with CIFAR-sized input"""
        model = resnet18(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_model_parameters(self):
        model = resnet18(num_classes=10)
        params = sum(p.numel() for p in model.parameters())
        assert params > 1_000_000  # ResNet18 has ~11M params

    def test_resnet50_parameters(self):
        model = resnet50(num_classes=10)
        params = sum(p.numel() for p in model.parameters())
        assert params > 20_000_000  # ResNet50 has ~25M params
    
    def test_pretrained_model(self):
        """Test pretrained model loads without error"""
        model = create_resnet_model("resnet18", num_classes=10, pretrained=True)
        assert isinstance(model, nn.Module)
    
    def test_dropout_model(self):
        """Test model with dropout"""
        model = create_resnet_model("resnet18", num_classes=10, dropout_rate=0.5)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 10)


# ─────────────────────────────────────────────────────────────
#  Data Loader Tests
# ─────────────────────────────────────────────────────────────

class TestDataLoaders:
    """Test data loading functionality"""

    def test_cifar10_dataset_factory(self):
        dataset, num_classes, image_shape = DatasetFactory.get_dataset(
            'cifar10', './data', train=True, download=True
        )
        assert num_classes == 10
        assert image_shape == (3, 32, 32)
        assert len(dataset) == 50000

    def test_cifar100_dataset_factory(self):
        dataset, num_classes, image_shape = DatasetFactory.get_dataset(
            'cifar100', './data', train=True, download=True
        )
        assert num_classes == 100
        assert image_shape == (3, 32, 32)

    def test_dataloader_creation(self):
        loader, num_classes, image_shape = create_distributed_dataloader(
            dataset_name='cifar10',
            data_path='./data',
            batch_size=32,
            num_workers=0,
            is_train=True,
            world_size=1,
            rank=0,
        )
        assert num_classes == 10
        assert len(loader) > 0

    def test_dataloader_batch_shape(self):
        loader, _, _ = create_distributed_dataloader(
            dataset_name='cifar10',
            data_path='./data',
            batch_size=32,
            num_workers=0,
            is_train=True,
            world_size=1,
            rank=0,
        )
        images, labels = next(iter(loader))
        assert images.shape == (32, 3, 32, 32)
        assert labels.shape == (32,)

    def test_val_loader(self):
        loader, num_classes, _ = create_distributed_dataloader(
            dataset_name='cifar10',
            data_path='./data',
            batch_size=16,
            num_workers=0,
            is_train=False,
            world_size=1,
            rank=0,
        )
        assert num_classes == 10
        images, labels = next(iter(loader))
        assert images.shape[0] == 16


# ─────────────────────────────────────────────────────────────
#  Checkpointing Tests
# ─────────────────────────────────────────────────────────────

class TestCheckpointing:
    """Test checkpoint management"""
    
    def test_checkpoint_manager_creation(self, tmp_path):
        manager = CheckpointManager(str(tmp_path), keep_last_n=3)
        assert manager.checkpoint_dir.exists()
    
    def test_checkpoint_save_load(self, tmp_path):
        manager = CheckpointManager(str(tmp_path), keep_last_n=3)
        
        state = {
            'model': {'weight': torch.randn(10)},
            'optimizer': {},
        }
        path = manager.save_checkpoint(state, step=100, metric=0.95)
        assert path.exists()
        
        loaded = manager.load_checkpoint(str(path))
        assert 'model' in loaded
        assert 'step' in loaded
        assert loaded['step'] == 100
    
    def test_checkpoint_save_api(self, tmp_path):
        """Test the backward-compatible save() API"""
        manager = CheckpointManager(str(tmp_path))
        
        state = {
            'model_state_dict': {'w': torch.randn(5)},
            'epoch': 3,
        }
        path = manager.save(state, epoch=3, step=500, is_best=True)
        assert path.exists()
        
        best_path = tmp_path / "checkpoint_best.pt"
        assert best_path.exists()
    
    def test_checkpoint_cleanup(self, tmp_path):
        manager = CheckpointManager(str(tmp_path), keep_last_n=2)
        
        for i in range(5):
            state = {'model': {}, 'step': i}
            manager.save_checkpoint(state, step=i)
        
        # Should only keep last 2 step checkpoints + best + latest
        step_checkpoints = list(tmp_path.glob("checkpoint_step*.pt"))
        assert len(step_checkpoints) <= 2
    
    def test_load_latest(self, tmp_path):
        manager = CheckpointManager(str(tmp_path))
        
        state1 = {'data': 'first', 'step': 1}
        manager.save_checkpoint(state1, step=1)
        
        state2 = {'data': 'second', 'step': 2}
        manager.save_checkpoint(state2, step=2)
        
        loaded = manager.load_checkpoint()  # Should load latest
        assert loaded['step'] == 2
    
    def test_has_checkpoint(self, tmp_path):
        manager = CheckpointManager(str(tmp_path))
        assert not manager.has_checkpoint()
        
        manager.save_checkpoint({'model': {}}, step=0)
        assert manager.has_checkpoint()


# ─────────────────────────────────────────────────────────────
#  Metrics Tests
# ─────────────────────────────────────────────────────────────

class TestMetrics:
    """Test metrics tracking"""
    
    def test_average_meter(self):
        meter = AverageMeter()
        meter.update(1.0)
        meter.update(2.0)
        meter.update(3.0)
        assert meter.avg == 2.0
        assert meter.min == 1.0
        assert meter.max == 3.0
        assert meter.count == 3
    
    def test_average_meter_weighted(self):
        meter = AverageMeter()
        meter.update(2.0, n=3)  # 3 samples with value 2.0
        meter.update(4.0, n=1)  # 1 sample with value 4.0
        expected_avg = (2.0 * 3 + 4.0 * 1) / 4
        assert abs(meter.avg - expected_avg) < 1e-6
    
    def test_accuracy_calculation(self):
        output = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        target = torch.tensor([1, 0, 1])
        acc = accuracy(output, target, topk=(1,))
        assert acc[0] == 100.0  # All predictions correct
    
    def test_accuracy_partial(self):
        output = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
        target = torch.tensor([1, 0, 1])
        acc = accuracy(output, target, topk=(1,))
        # pred = [0, 0, 1], target = [1, 0, 1] -> 2/3 correct
        assert abs(acc[0] - 66.67) < 1.0
    
    def test_accuracy_topk(self):
        output = torch.tensor([
            [0.3, 0.5, 0.2],
            [0.1, 0.2, 0.7],
        ])
        target = torch.tensor([0, 2])
        top1, top2 = accuracy(output, target, topk=(1, 2))
        assert top2 >= top1
    
    def test_metrics_tracker(self, tmp_path):
        tracker = MetricsTracker(
            log_dir=str(tmp_path),
            use_tensorboard=False,
        )
        
        metrics = {'loss': 1.5, 'accuracy': 85.0}
        tracker.update(metrics, step=1)
        
        assert tracker.get_metric('loss') == 1.5
        assert tracker.get_metric('accuracy') == 85.0
        
        tracker.close()
    
    def test_metrics_tracker_jsonl(self, tmp_path):
        tracker = MetricsTracker(log_dir=str(tmp_path), use_tensorboard=False)
        tracker.update({'loss': 0.5}, step=1)
        tracker.update({'loss': 0.3}, step=2)
        
        metrics_file = tmp_path / "metrics.jsonl"
        assert metrics_file.exists()
        
        lines = metrics_file.read_text().strip().split('\n')
        assert len(lines) == 2
        
        data = json.loads(lines[0])
        assert data['loss'] == 0.5
        assert data['step'] == 1
        
        tracker.close()
    
    def test_metrics_tracker_summary(self, tmp_path):
        tracker = MetricsTracker(log_dir=str(tmp_path), use_tensorboard=False)
        tracker.update({'loss': 1.0}, step=1)
        tracker.update({'loss': 0.5}, step=2)
        tracker.update({'loss': 0.2}, step=3)
        
        summary = tracker.get_summary()
        assert 'loss' in summary
        assert abs(summary['loss']['average'] - 0.5667) < 0.01
        
        tracker.close()


# ─────────────────────────────────────────────────────────────
#  GPU Monitoring Tests
# ─────────────────────────────────────────────────────────────

class TestGPUMonitoring:
    """Test GPU monitoring (skips if no GPU)"""
    
    def test_get_gpu_info(self):
        gpu_info = get_gpu_info()
        assert isinstance(gpu_info, list)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_gpu_info_with_gpu(self):
        gpu_info = get_gpu_info()
        assert len(gpu_info) > 0
        assert 'name' in gpu_info[0]
        assert 'total_memory_mb' in gpu_info[0]
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_gpu_monitoring_with_gpu(self):
        monitor = GPUMonitor(device_id=0)
        metrics = monitor.get_gpu_metrics(force=True)
        
        assert 'memory_allocated_mb' in metrics
        assert metrics['memory_allocated_mb'] >= 0
        assert 'memory_total_mb' in metrics
    
    def test_gpu_monitor_no_gpu(self):
        """GPUMonitor should return empty dict gracefully when no GPU"""
        if torch.cuda.is_available():
            pytest.skip("GPU is available, testing no-GPU path not possible")
        monitor = GPUMonitor(device_id=0)
        metrics = monitor.get_gpu_metrics()
        assert metrics == {}
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_gpu_peak_metrics(self):
        monitor = GPUMonitor(device_id=0)
        monitor.reset_peak_metrics()
        
        # Allocate some GPU memory
        t = torch.randn(1000, 1000, device='cuda')
        
        metrics = monitor.get_gpu_metrics(force=True)
        peak = monitor.get_peak_metrics()
        
        assert peak['peak_memory_allocated_mb'] > 0
        
        del t
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
#  Profiler Tests
# ─────────────────────────────────────────────────────────────

class TestProfiler:
    """Test profiling utilities"""
    
    def test_profiler_record(self):
        from utils.profiler import Profiler
        
        profiler = Profiler()
        profiler.record("test_op", 0.5)
        profiler.record("test_op", 1.0)
        profiler.record("test_op", 0.75)
        
        summary = profiler.summary()
        assert "test_op" in summary
        assert abs(summary["test_op"]["mean"] - 0.75) < 1e-6

    def test_profiler_measure_context(self):
        from utils.profiler import Profiler
        import time
        
        profiler = Profiler()
        with profiler.measure("sleep_test"):
            time.sleep(0.1)
        
        assert "sleep_test" in profiler.timings
        assert profiler.timings["sleep_test"][0] >= 0.09
    
    def test_throughput_calculator(self):
        from utils.profiler import ThroughputCalculator
        import time
        
        calc = ThroughputCalculator()
        calc.start()
        
        time.sleep(0.1)
        calc.update(100)
        
        throughput = calc.get_throughput()
        assert throughput > 0
    
    def test_memory_snapshot(self):
        from utils.profiler import get_memory_snapshot
        
        snapshot = get_memory_snapshot()
        if torch.cuda.is_available():
            assert 'allocated_mb' in snapshot
        else:
            assert snapshot == {}


# ─────────────────────────────────────────────────────────────
#  Transformer Model Tests
# ─────────────────────────────────────────────────────────────

class TestTransformerModels:
    """Test Vision Transformer models"""
    
    def test_vit_creation(self):
        from models.transformer import create_transformer_model
        model = create_transformer_model("vit_b_16", num_classes=10)
        assert isinstance(model, nn.Module)
    
    def test_vit_forward_pass(self):
        from models.transformer import create_transformer_model
        model = create_transformer_model("vit_b_16", num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_vit_parameter_count(self):
        from models.transformer import create_transformer_model
        model = create_transformer_model("vit_b_16", num_classes=10)
        params = sum(p.numel() for p in model.parameters())
        assert params > 80_000_000  # ViT-B has ~86M params


# ─────────────────────────────────────────────────────────────
#  Integration Tests
# ─────────────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_model_with_cifar10_data(self):
        """Test a model forward pass with real CIFAR-10 data"""
        model = resnet18(num_classes=10)
        model.eval()
        
        loader, num_classes, shape = create_distributed_dataloader(
            dataset_name='cifar10',
            data_path='./data',
            batch_size=4,
            num_workers=0,
            is_train=False,
            world_size=1,
            rank=0,
        )
        
        images, labels = next(iter(loader))
        with torch.no_grad():
            output = model(images)
        
        assert output.shape == (4, 10)
        assert num_classes == 10
    
    def test_training_step_simulation(self):
        """Simulate a single training step without GPU"""
        model = resnet18(num_classes=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))
        
        # Forward
        output = model(x)
        loss = criterion(output, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
    
    def test_checkpoint_roundtrip(self, tmp_path):
        """Test saving and loading a model checkpoint"""
        model = resnet18(num_classes=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        manager = CheckpointManager(str(tmp_path))
        
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        manager.save_checkpoint(state, step=100, metric=0.85)
        
        loaded = manager.load_checkpoint()
        
        model2 = resnet18(num_classes=10)
        model2.load_state_dict(loaded['model'])
        
        # Verify same parameters
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Parameters differ at {n1}"


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
