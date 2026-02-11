"""
Distributed Training with Ray Train and PyTorch DDP

This module implements the main distributed training loop using Ray Train
for orchestration and PyTorch DistributedDataParallel for multi-GPU training.

Optimized for:
- Single GPU (RTX 5070) or Multi-GPU setups
- Fault tolerance with automatic checkpointing
- Mixed precision training (FP16/BF16)
- Gradient accumulation for large effective batch sizes
- Comprehensive metrics and monitoring
"""

import os
import time
import logging
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import ray
from ray import train
from ray.train import Checkpoint, ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer, TorchConfig

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.resnet import create_resnet_model
from models.transformer import create_transformer_model
from data.data_loader import create_distributed_dataloader
from training.checkpointing import CheckpointManager
from utils.metrics import MetricsTracker
from utils.gpu_monitor import GPUMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose Ray internal warnings during cleanup
logging.getLogger('ray.train').setLevel(logging.ERROR)
logging.getLogger('ray.util').setLevel(logging.ERROR)
logging.getLogger('ray._private').setLevel(logging.ERROR)

# Filter out specific noisy warnings
import warnings
warnings.filterwarnings('ignore', message='.*PlacementGroupCleaner.*')
warnings.filterwarnings('ignore', message='.*GCS.*')


@dataclass
class TrainingConfig:
    """Training configuration with all hyperparameters"""
    
    # Model configuration
    model: str = "resnet50"
    num_classes: int = 10  # CIFAR-10 default
    pretrained: bool = False
    
    # Dataset configuration
    dataset: str = "cifar10"
    data_path: str = "/mnt/data"
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Training hyperparameters
    epochs: int = 90
    batch_size: int = 256
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    
    # Distributed training
    num_gpu_workers: int = 1  # For RTX 5070: 1 worker
    gpus_per_worker: int = 1
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
    
    # Logging
    log_freq: int = 10  # steps
    eval_freq: int = 500  # steps
    
    # Ray specific
    ray_address: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


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
        
        logger.info(f"Initialized trainer with config: {config.model}, {config.dataset}")
    
    def setup(self, world_rank: int, world_size: int):
        """Setup training components for distributed training"""
        
        # Set device
        if self.config.use_gpu and torch.cuda.is_available():
            local_rank = train.get_context().get_local_rank()
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
            logger.info(f"Rank {world_rank}/{world_size} using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device("cpu")
            logger.info(f"Rank {world_rank}/{world_size} using CPU")
        
        # Create model
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Wrap model with DDP if distributed
        if world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.config.use_gpu else None,
                find_unused_parameters=False,
                broadcast_buffers=True,
                gradient_as_bucket_view=True  # Performance optimization
            )
            logger.info(f"Model wrapped with DDP")
        
        # Create optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            nesterov=True
        )
        
        # Create learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < self.config.warmup_epochs:
                return (epoch + 1) / self.config.warmup_epochs
            return 0.5 * (1 + torch.cos(torch.tensor(
                (epoch - self.config.warmup_epochs) / 
                (self.config.epochs - self.config.warmup_epochs) * 3.14159265359
            )))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision scaler
        if self.config.mixed_precision in ["fp16", "bf16"]:
            self.scaler = GradScaler(enabled=True)
            logger.info(f"Mixed precision training enabled: {self.config.mixed_precision}")
        
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
        if self.config.use_gpu and torch.cuda.is_available():
            self.gpu_monitor = GPUMonitor(device_id=self.device.index)
        
        # Resume from checkpoint if specified
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)
        
        logger.info("Setup completed successfully")
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        if "resnet" in self.config.model.lower():
            model = create_resnet_model(
                model_name=self.config.model,
                num_classes=self.config.num_classes,
                pretrained=self.config.pretrained
            )
        elif "vit" in self.config.model.lower() or "transformer" in self.config.model.lower():
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
                logger.info("Gradient checkpointing enabled")
        
        return model
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_samples = 0
        
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
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.mixed_precision in ["fp16", "bf16"]:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_val is not None:
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
                
                self.optimizer.zero_grad()
            
            # Calculate metrics
            with torch.no_grad():
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum().item()
                accuracy = correct / targets.size(0)
                
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                epoch_accuracy += accuracy
                epoch_samples += targets.size(0)
            
            # Update global step
            self.global_step += 1
            
            # Logging
            if (batch_idx + 1) % self.config.log_freq == 0:
                batch_time = time.time() - batch_start_time
                throughput = targets.size(0) / batch_time
                
                metrics = {
                    "epoch": epoch,
                    "batch": batch_idx + 1,
                    "loss": loss.item() * self.config.gradient_accumulation_steps,
                    "accuracy": accuracy * 100,
                    "lr": self.scheduler.get_last_lr()[0],
                    "throughput": throughput,
                    "step": self.global_step
                }
                
                # Add GPU metrics if available
                if self.gpu_monitor:
                    gpu_metrics = self.gpu_monitor.get_metrics()
                    metrics.update(gpu_metrics)
                
                # Log to file
                logger.info(
                    f"Epoch [{epoch}/{self.config.epochs}] "
                    f"Batch [{batch_idx+1}/{num_batches}] "
                    f"Loss: {metrics['loss']:.4f} "
                    f"Acc: {metrics['accuracy']:.2f}% "
                    f"LR: {metrics['lr']:.6f} "
                    f"Throughput: {throughput:.1f} samples/sec"
                )
                
                # Track metrics
                self.metrics_tracker.log(metrics, self.global_step)
            
            # Checkpointing
            if self.global_step % self.config.checkpoint_freq == 0:
                self._save_checkpoint(epoch, batch_idx)
        
        # Epoch metrics
        epoch_time = time.time() - epoch_start_time
        epoch_metrics = {
            "loss": epoch_loss / num_batches,
            "accuracy": epoch_accuracy / num_batches,
            "throughput": epoch_samples / epoch_time,
            "epoch_time": epoch_time
        }
        
         # Log to file
        logger.info(
            f"Epoch [{epoch}/{self.config.epochs}] completed | "
            f"Avg Loss: {epoch_metrics['loss']:.4f} | "
            f"Avg Acc: {epoch_metrics['accuracy']*100:.2f}% | "
            f"Throughput: {epoch_metrics['throughput']:.1f} samples/sec | "
            f"Time: {epoch_time:.1f}s"
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
        
        logger.info(f"Validation | Loss: {val_loss:.4f} | Acc: {val_accuracy*100:.2f}%")

    
        # Update best accuracy
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self._save_checkpoint(epoch, is_best=True)
            logger.info(f"New best accuracy: {self.best_accuracy*100:.2f}%")
        
        return metrics
    
    def _save_checkpoint(
        self,
        epoch: int,
        batch_idx: int = 0,
        is_best: bool = False
    ):
        """Save training checkpoint"""
        
        # Get model state dict (handle DDP wrapper)
        if isinstance(self.model, DDP):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint_data = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "global_step": self.global_step,
            "model_state_dict": model_state,
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
        try:
            checkpoint = Checkpoint.from_directory(str(Path(checkpoint_path).parent))
            train.report(
                metrics={"epoch": epoch, "step": self.global_step},
                checkpoint=checkpoint
            )
        except:
            pass  # Not running under Ray Train
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint_data["epoch"]
        self.global_step = checkpoint_data["global_step"]
        self.best_accuracy = checkpoint_data["best_accuracy"]
        
        # Load model state (handle DDP wrapper)
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint_data["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint_data["model_state_dict"])
        
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
        
        if self.scaler and "scaler_state_dict" in checkpoint_data:
            self.scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
        
        logger.info(
            f"Checkpoint loaded | Epoch: {self.current_epoch} | "
            f"Step: {self.global_step} | Best Acc: {self.best_accuracy*100:.2f}%"
        )


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
    train_loader, _, _ = create_distributed_dataloader(
        dataset_name=training_config.dataset,
        data_path=training_config.data_path,
        batch_size=training_config.batch_size,
        is_train=True,
        num_workers=training_config.num_workers,
        prefetch_factor=training_config.prefetch_factor
    )
    
    val_loader, _, _ = create_distributed_dataloader(
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
        try:
            train.report(metrics)
        except:
            pass  # Not running under Ray Train
    
    return {"final_accuracy": trainer.best_accuracy}


def main():
    """Main entry point for standalone execution"""
    parser = argparse.ArgumentParser(description="Distributed Training with Ray")
    
    # Model config
    parser.add_argument("--model", type=str, default="resnet18",
                       choices=["resnet18", "resnet50", "resnet101", "vit_b_16"])
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--pretrained", action="store_true")
    
    # Dataset config
    parser.add_argument("--dataset", type=str, default="cifar10",
                       choices=["cifar10", "cifar100", "imagenet"])
    parser.add_argument("--data-path", type=str, default="/mnt/data")
    
    # Training config
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    
    # Distributed config
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpus-per-worker", type=int, default=1)
    parser.add_argument("--use-gpu", type=bool, default=True)
    
    # Optimization
    parser.add_argument("--mixed-precision", type=str, default="fp16",
                       choices=["none", "fp16", "bf16"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-clip-val", type=float, default=None)
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="/mnt/checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=1000)
    parser.add_argument("--resume-from", type=str, default=None)
    
    # Add these missing arguments
    parser.add_argument("--log-freq", type=int, default=10, help="Logging frequency in steps")
    parser.add_argument("--eval-freq", type=int, default=500, help="Evaluation frequency in steps")

    # Ray config
    parser.add_argument("--ray-address", type=str, default=None,
                       help="Ray cluster address (None for local)")
    
    # Backend config
    parser.add_argument("--backend", type=str, default="nccl",
                       choices=["nccl", "gloo"],
                       help="Distributed backend (nccl for GPU, gloo for CPU/Windows)")

    args = parser.parse_args()
    
    # Initialize Ray
    # When running in Docker, connect to the Ray head started by docker-compose
    # The Ray head is already running with dashboard enabled
    if args.ray_address:
        address = args.ray_address
            
        logger.info(f"Connecting to Ray cluster via Client at: {address}")
        ray.init(
            address=address,
            log_to_driver=True,  # Show worker logs in terminal
            logging_level=logging.INFO,
            runtime_env={
                    "env_vars": {
                        "PYTHONPATH": "/app/src:/app:$PYTHONPATH",
                        "PYTHONUNBUFFERED": "1",
                        "RAY_LOG_TO_DRIVER": "1"
                    },
                    "working_dir": "/app"
                }
        )
    else:
        # Check if Ray is already running (started by Docker)
        if ray.is_initialized():
            logger.info("Ray is already initialized")
        else:
            # Connect to local Ray head (started by docker-compose on port 6379)
            ray.init(
                address="auto",
                log_to_driver=True,  # Show worker logs in terminal
                logging_level=logging.INFO,
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": "/app/src:/app:$PYTHONPATH",
                        "PYTHONUNBUFFERED": "1",
                        "RAY_LOG_TO_DRIVER": "1"
                    },
                    "working_dir": "/app"
                }
            )
    
    # Log dashboard URL
    try:
        dashboard_url = ray.get_dashboard_url()
        if dashboard_url:
            logger.info("=" * 80)
            logger.info(f"🎯 Ray Dashboard: {dashboard_url}")
            logger.info("=" * 80)
    except Exception as e:
        logger.warning(f"Could not get dashboard URL: {e}")
    
    # Create config
    config = TrainingConfig(
        model=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        dataset=args.dataset,
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        num_gpu_workers=args.num_workers,
        gpus_per_worker=args.gpus_per_worker,
        use_gpu=args.use_gpu,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip_val=args.gradient_clip_val,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        resume_from=args.resume_from,
        ray_address=args.ray_address,
        log_freq=args.log_freq,    # Map the new log-freq
        eval_freq=args.eval_freq  # Map the new eval-freq
    )
    
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    for key, value in config.to_dict().items():
        logger.info(f"{key:30s}: {value}")
    logger.info("=" * 80)
    
    # Create trainer
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        resources_per_worker={"GPU": args.gpus_per_worker, "CPU": 4}
    )
    
    run_config = RunConfig(
        name=f"{args.model}_{args.dataset}",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3
        ),
        # Suppress verbose cleanup errors
        failure_config=ray.train.FailureConfig(
            max_failures=0  # Don't retry on failure, just report it cleanly
        )
    )
    
    trainer = TorchTrainer(
        train_func,
        train_loop_config=config.to_dict(),
        scaling_config=scaling_config,
        run_config=run_config,
        torch_config=TorchConfig(backend=args.backend)
    )

    import os
    os.environ['RAY_AIR_NEW_OUTPUT'] = '1'  # Enable new verbose output
    os.environ['RAY_DEDUP_LOGS'] = '0'      # ADD THIS - show all logs
    os.environ['RAY_COLOR_PREFIX'] = '1'    # ADD THIS - blue colors!
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '0'  # Keep loggers
    
    # Train
    logger.info("Starting training...")
    result = trainer.fit()
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Final metrics: {result.metrics}")
    logger.info(f"Checkpoint: {result.checkpoint}")
    logger.info("=" * 80)
    
    ray.shutdown()


if __name__ == "__main__":
    main()