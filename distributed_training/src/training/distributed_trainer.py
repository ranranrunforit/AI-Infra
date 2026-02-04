"""
Distributed Training with Ray Train and PyTorch DDP

This module implements the main distributed training loop using Ray Train
for orchestration and PyTorch DistributedDataParallel for multi-GPU training.

Key Features:
- Multi-node, multi-GPU training with linear scaling
- Fault tolerance with automatic checkpointing
- Mixed precision training (FP16/BF16)
- Gradient accumulation for large effective batch sizes
- MLflow integration for experiment tracking
- Comprehensive metrics and monitoring
"""

import os
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import ray
from ray import train
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer

import mlflow

from ..models.resnet import create_resnet_model
from ..models.transformer import create_transformer_model
from ..data.data_loader import create_distributed_dataloader
from .checkpointing import CheckpointManager
from ..utils.metrics import MetricsTracker
from ..utils.gpu_monitor import GPUMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration with all hyperparameters"""

    # Model configuration
    model: str = "resnet50"
    num_classes: int = 1000
    pretrained: bool = False

    # Dataset configuration
    dataset: str = "imagenet"
    data_path: str = "/mnt/data"
    num_workers: int = 8
    prefetch_factor: int = 2

    # Training hyperparameters
    epochs: int = 90
    batch_size: int = 256
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Distributed training
    num_gpu_workers: int = 4
    gpus_per_worker: int = 2
    use_gpu: bool = True
    backend: str = "nccl"

    # Optimization
    mixed_precision: str = "fp16"  # fp16, bf16, or none
    gradient_accumulation_steps: int = 1
    gradient_clip_val: Optional[float] = None
    use_gradient_checkpointing: bool = False

    # Checkpointing
    checkpoint_dir: str = "/mnt/checkpoints"
    checkpoint_freq: int = 1000  # steps
    keep_last_n: int = 3
    resume_from: Optional[str] = None

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment: str = "distributed-training"

    # Logging
    log_freq: int = 10  # steps
    eval_freq: int = 1000  # steps

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


class DistributedTrainer:
    """Main distributed training class"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.checkpoint_manager = None
        self.metrics_tracker = None
        self.gpu_monitor = None

        self.global_step = 0
        self.current_epoch = 0
        self.best_accuracy = 0.0

    def setup(self, world_rank: int, world_size: int):
        """Setup training components for distributed training"""

        # Set device
        if self.config.use_gpu:
            local_rank = train.get_context().get_local_rank()
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        logger.info(f"Rank {world_rank}/{world_size} using device: {self.device}")

        # Create model
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        # Wrap model with DDP
        self.model = DDP(
            self.model,
            device_ids=[self.device] if self.config.use_gpu else None,
            find_unused_parameters=False,
            broadcast_buffers=True,
            gradient_as_bucket_view=True  # Performance optimization
        )

        # Create optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            nesterov=True
        )

        # Create learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=0
        )

        # Mixed precision scaler
        if self.config.mixed_precision in ["fp16", "bf16"]:
            self.scaler = GradScaler(enabled=True)

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            keep_last_n=self.config.keep_last_n
        )

        # Metrics tracker
        self.metrics_tracker = MetricsTracker(
            log_dir=os.path.join(self.config.checkpoint_dir, "logs")
        )

        # GPU monitor
        if self.config.use_gpu:
            self.gpu_monitor = GPUMonitor(device_id=local_rank)

        # Resume from checkpoint if specified
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

        # MLflow setup (only on rank 0)
        if world_rank == 0:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment)
            mlflow.start_run()
            mlflow.log_params(self.config.to_dict())

    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        if "resnet" in self.config.model:
            model = create_resnet_model(
                model_name=self.config.model,
                num_classes=self.config.num_classes,
                pretrained=self.config.pretrained
            )
        elif "transformer" in self.config.model or "vit" in self.config.model:
            model = create_transformer_model(
                model_name=self.config.model,
                num_classes=self.config.num_classes,
                pretrained=self.config.pretrained
            )
        else:
            raise ValueError(f"Unknown model: {self.config.model}")

        # Apply gradient checkpointing if enabled
        if self.config.use_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        return model

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""

        self.model.train()
        epoch_metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "throughput": 0.0
        }

        num_batches = len(train_loader)
        epoch_start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_start_time = time.time()

            # Move data to device
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with autocast(
                enabled=self.config.mixed_precision in ["fp16", "bf16"],
                dtype=torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
            ):
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)

            # Backward pass
            if self.config.mixed_precision in ["fp16", "bf16"]:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_val:
                    if self.config.mixed_precision in ["fp16", "bf16"]:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )

                # Optimizer step
                if self.config.mixed_precision in ["fp16", "bf16"]:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            # Calculate metrics
            with torch.no_grad():
                _, predicted = outputs.max(1)
                accuracy = predicted.eq(targets).float().mean().item()

            batch_time = time.time() - batch_start_time
            throughput = inputs.size(0) / batch_time

            # Update epoch metrics
            epoch_metrics["loss"] += loss.item()
            epoch_metrics["accuracy"] += accuracy
            epoch_metrics["throughput"] += throughput

            # Log metrics
            if batch_idx % self.config.log_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                gpu_util = self.gpu_monitor.get_utilization() if self.gpu_monitor else 0.0

                log_dict = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "loss": loss.item(),
                    "accuracy": accuracy * 100,
                    "throughput": throughput,
                    "lr": current_lr,
                    "gpu_util": gpu_util,
                    "step": self.global_step
                }

                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {accuracy*100:.2f}% "
                    f"Throughput: {throughput:.1f} samples/sec "
                    f"GPU: {gpu_util:.1f}%"
                )

                self.metrics_tracker.log_step(log_dict)

                # MLflow logging (rank 0 only)
                if train.get_context().get_world_rank() == 0:
                    mlflow.log_metrics(log_dict, step=self.global_step)

            # Checkpointing
            if self.global_step % self.config.checkpoint_freq == 0:
                self._save_checkpoint(epoch, batch_idx)

        # Calculate average metrics
        epoch_metrics["loss"] /= num_batches
        epoch_metrics["accuracy"] /= num_batches
        epoch_metrics["throughput"] /= num_batches

        epoch_time = time.time() - epoch_start_time
        epoch_metrics["epoch_time"] = epoch_time

        logger.info(
            f"Epoch {epoch} completed in {epoch_time:.2f}s | "
            f"Avg Loss: {epoch_metrics['loss']:.4f} | "
            f"Avg Acc: {epoch_metrics['accuracy']*100:.2f}% | "
            f"Avg Throughput: {epoch_metrics['throughput']:.1f} samples/sec"
        )

        return epoch_metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Validate model"""

        self.model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        num_batches = len(val_loader)

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)

            _, predicted = outputs.max(1)
            accuracy = predicted.eq(targets).float().mean().item()

            val_loss += loss.item()
            val_accuracy += accuracy

        val_loss /= num_batches
        val_accuracy /= num_batches

        metrics = {
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }

        logger.info(
            f"Validation | Loss: {val_loss:.4f} | "
            f"Acc: {val_accuracy*100:.2f}%"
        )

        # MLflow logging (rank 0 only)
        if train.get_context().get_world_rank() == 0:
            mlflow.log_metrics(metrics, step=self.global_step)

        # Update best accuracy
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self._save_checkpoint(epoch, is_best=True)

        return metrics

    def _save_checkpoint(
        self,
        epoch: int,
        batch_idx: int = 0,
        is_best: bool = False
    ):
        """Save training checkpoint"""

        checkpoint_data = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "global_step": self.global_step,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_accuracy": self.best_accuracy,
            "config": self.config.to_dict()
        }

        if self.scaler:
            checkpoint_data["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = self.checkpoint_manager.save(
            checkpoint_data,
            epoch=epoch,
            step=self.global_step,
            is_best=is_best
        )

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Report checkpoint to Ray Train
        checkpoint = Checkpoint.from_directory(str(Path(checkpoint_path).parent))
        train.report(
            metrics={"epoch": epoch, "step": self.global_step},
            checkpoint=checkpoint
        )

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)

        self.current_epoch = checkpoint_data["epoch"]
        self.global_step = checkpoint_data["global_step"]
        self.best_accuracy = checkpoint_data["best_accuracy"]

        self.model.module.load_state_dict(checkpoint_data["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

        if self.scaler and "scaler_state_dict" in checkpoint_data:
            self.scaler.load_state_dict(checkpoint_data["scaler_state_dict"])

        logger.info(
            f"Checkpoint loaded | Epoch: {self.current_epoch} | "
            f"Step: {self.global_step} | Best Acc: {self.best_accuracy*100:.2f}%"
        )

    def cleanup(self):
        """Cleanup resources"""
        if train.get_context().get_world_rank() == 0:
            mlflow.end_run()


def train_func(config: Dict[str, Any]):
    """
    Main training function for Ray Train

    This function is executed on each worker in the Ray cluster.
    Ray Train handles the distributed setup and coordination.
    """

    # Parse config
    training_config = TrainingConfig.from_dict(config)

    # Get distributed context
    world_rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()

    logger.info(
        f"Starting training on rank {world_rank}/{world_size} | "
        f"Model: {training_config.model} | "
        f"Dataset: {training_config.dataset}"
    )

    # Create trainer
    trainer = DistributedTrainer(training_config)
    trainer.setup(world_rank, world_size)

    # Create data loaders
    train_loader = create_distributed_dataloader(
        dataset_name=training_config.dataset,
        data_path=training_config.data_path,
        batch_size=training_config.batch_size,
        is_train=True,
        num_workers=training_config.num_workers,
        prefetch_factor=training_config.prefetch_factor
    )

    val_loader = create_distributed_dataloader(
        dataset_name=training_config.dataset,
        data_path=training_config.data_path,
        batch_size=training_config.batch_size,
        is_train=False,
        num_workers=training_config.num_workers,
        prefetch_factor=training_config.prefetch_factor
    )

    # Training loop
    for epoch in range(trainer.current_epoch, training_config.epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Validate
        if (epoch + 1) % 5 == 0 or epoch == training_config.epochs - 1:
            val_metrics = trainer.validate(val_loader, epoch)
        else:
            val_metrics = {}

        # Step scheduler
        trainer.scheduler.step()

        # Report to Ray Train
        metrics = {**train_metrics, **val_metrics}
        train.report(metrics)

    # Cleanup
    trainer.cleanup()

    return {"final_accuracy": trainer.best_accuracy}


def main():
    """Main entry point for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Distributed Training with Ray")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-path", type=str, default="/mnt/data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpus-per-worker", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", type=str, default="/mnt/checkpoints")
    parser.add_argument("--ray-address", type=str, default=None,
                       help="Ray cluster address (None for local)")

    args = parser.parse_args()

    # Initialize Ray
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    # Create config
    config = TrainingConfig(
        model=args.model,
        dataset=args.dataset,
        data_path=args.data_path,
        num_gpu_workers=args.num_workers,
        gpus_per_worker=args.gpus_per_worker,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )

    # Create trainer
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=True,
        resources_per_worker={"GPU": args.gpus_per_worker, "CPU": 8}
    )

    trainer = TorchTrainer(
        train_func,
        train_loop_config=config.to_dict(),
        scaling_config=scaling_config
    )

    # Train
    result = trainer.fit()

    print("Training completed!")
    print(f"Final metrics: {result.metrics}")
    print(f"Checkpoint: {result.checkpoint}")

    ray.shutdown()


if __name__ == "__main__":
    main()
