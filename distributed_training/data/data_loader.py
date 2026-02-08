"""
Distributed Data Loading

Handles data loading for distributed training with proper sharding and augmentations.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Optional
import ray.train.torch


def get_transforms(dataset_name: str, is_train: bool):
    """Get data transforms for dataset"""
    
    if dataset_name.lower() in ["cifar10", "cifar100"]:
        if is_train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])
    else:  # ImageNet-style
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    return transform


def create_distributed_dataloader(
    dataset_name: str,
    data_path: str,
    batch_size: int,
    is_train: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True
) -> DataLoader:
    """Create distributed data loader"""
    
    transform = get_transforms(dataset_name, is_train)
    
    # Load dataset
    if dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_path,
            train=is_train,
            download=True,
            transform=transform
        )
    elif dataset_name.lower() == "cifar100":
        dataset = datasets.CIFAR100(
            root=data_path,
            train=is_train,
            download=True,
            transform=transform
        )
    elif dataset_name.lower() == "imagenet":
        split = 'train' if is_train else 'val'
        dataset = datasets.ImageNet(
            root=data_path,
            split=split,
            transform=transform
        )
    else:
        import os
        data_dir = os.path.join(data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=is_train
    )
    
    # Prepare with Ray Train
    try:
        dataloader = ray.train.torch.prepare_data_loader(dataloader)
    except:
        pass
    
    return dataloader
