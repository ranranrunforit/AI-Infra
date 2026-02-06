import pytest
from src.data.data_loader import create_distributed_dataloader

def test_cifar10_loader():
    loader = create_distributed_dataloader(
        dataset_name="cifar10",
        data_path="/tmp/data",
        batch_size=32,
        is_train=True,
        num_workers=0
    )
    assert loader is not None
