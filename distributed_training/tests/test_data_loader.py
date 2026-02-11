"""
Data loader tests — tests DatasetFactory, transforms, and distributed loader.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loader import (
    create_distributed_dataloader,
    DatasetFactory,
    get_transforms,
    DATASET_INFO,
)


class TestDatasetFactory:
    """Test the DatasetFactory"""
    
    def test_cifar10(self):
        dataset, num_classes, shape = DatasetFactory.get_dataset(
            'cifar10', './data', train=True, download=True
        )
        assert num_classes == 10
        assert shape == (3, 32, 32)
        assert len(dataset) == 50000
    
    def test_cifar10_test(self):
        dataset, num_classes, shape = DatasetFactory.get_dataset(
            'cifar10', './data', train=False, download=True
        )
        assert num_classes == 10
        assert len(dataset) == 10000
    
    def test_cifar100(self):
        dataset, num_classes, shape = DatasetFactory.get_dataset(
            'cifar100', './data', train=True, download=True
        )
        assert num_classes == 100
        assert shape == (3, 32, 32)
    
    def test_dataset_info_registry(self):
        assert "cifar10" in DATASET_INFO
        assert "cifar100" in DATASET_INFO
        assert "imagenet" in DATASET_INFO


class TestTransforms:
    """Test data transforms"""
    
    def test_cifar_train_transforms(self):
        t = get_transforms("cifar10", is_train=True)
        assert t is not None
    
    def test_cifar_val_transforms(self):
        t = get_transforms("cifar10", is_train=False)
        assert t is not None
    
    def test_imagenet_train_transforms(self):
        t = get_transforms("imagenet", is_train=True)
        assert t is not None
    
    def test_imagenet_val_transforms(self):
        t = get_transforms("imagenet", is_train=False)
        assert t is not None


class TestDistributedDataloader:
    """Test the distributed dataloader"""
    
    def test_create_loader(self):
        loader, num_classes, shape = create_distributed_dataloader(
            dataset_name='cifar10',
            data_path='./data',
            batch_size=16,
            num_workers=0,
            is_train=True,
            world_size=1,
            rank=0,
        )
        assert num_classes == 10
        assert shape == (3, 32, 32)
    
    def test_loader_batch_shapes(self):
        loader, _, _ = create_distributed_dataloader(
            dataset_name='cifar10',
            data_path='./data',
            batch_size=8,
            num_workers=0,
            is_train=True,
            world_size=1,
            rank=0,
        )
        images, labels = next(iter(loader))
        assert images.shape == (8, 3, 32, 32)
        assert labels.shape == (8,)
    
    def test_val_loader(self):
        loader, _, _ = create_distributed_dataloader(
            dataset_name='cifar10',
            data_path='./data',
            batch_size=16,
            num_workers=0,
            is_train=False,
            world_size=1,
            rank=0,
        )
        images, labels = next(iter(loader))
        assert images.shape[0] == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
