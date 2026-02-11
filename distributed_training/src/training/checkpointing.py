"""
Checkpoint Management for Distributed Training

Provides robust checkpoint saving, loading, and management:
- Automatic checkpoint saving with step/epoch tracking
- Best model tracking by metric
- Old checkpoint cleanup
- Checkpoint metadata and versioning
"""

import torch
import shutil
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages training checkpoints with automatic cleanup and best model tracking.
    
    Usage:
        manager = CheckpointManager("./checkpoints", keep_last_n=3)
        
        # Save with the simple API (used by DistributedTrainer)
        path = manager.save(state_dict, epoch=5, step=1000, is_best=True)
        
        # Save with the extended API (used by tests)
        path = manager.save_checkpoint(state_dict, step=1000, metric=0.95)
        
        # Load
        data = manager.load_checkpoint(path)
    """
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._best_metric: Optional[float] = None
        self._best_path: Optional[Path] = None
    
    def save(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        step: int,
        is_best: bool = False
    ) -> Path:
        """
        Save a training checkpoint (backward-compatible API).
        
        Args:
            state_dict: Dict containing model, optimizer, etc. state
            epoch: Current epoch number
            step: Current global step
            is_best: Whether this is the best model so far
        
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Add metadata
        state_dict["_checkpoint_meta"] = {
            "epoch": epoch,
            "step": step,
            "timestamp": time.time(),
            "is_best": is_best,
        }
        
        torch.save(state_dict, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Copy to latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        shutil.copy(checkpoint_path, latest_path)
        
        # Copy to best
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            shutil.copy(checkpoint_path, best_path)
            self._best_path = best_path
            logger.info("New best checkpoint saved")
        
        self._cleanup_old_checkpoints()
        return checkpoint_path
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        step: int,
        metric: Optional[float] = None,
        epoch: Optional[int] = None
    ) -> Path:
        """
        Save a checkpoint with metric tracking (extended API).
        
        Args:
            state: Dict containing training state
            step: Current global step
            metric: Optional metric value (e.g., accuracy)
            epoch: Optional epoch number
        
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_name = f"checkpoint_step{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Embed metadata into the state
        save_data = {
            **state,
            "step": step,
            "metric": metric,
            "epoch": epoch,
            "timestamp": time.time(),
        }
        
        torch.save(save_data, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path} (metric={metric})")
        
        # Track best
        is_best = False
        if metric is not None:
            if self._best_metric is None or metric > self._best_metric:
                self._best_metric = metric
                is_best = True
                best_path = self.checkpoint_dir / "checkpoint_best.pt"
                shutil.copy(checkpoint_path, best_path)
                self._best_path = best_path
                logger.info(f"New best checkpoint: metric={metric}")
        
        # Copy to latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        shutil.copy(checkpoint_path, latest_path)
        
        self._cleanup_old_checkpoints()
        return checkpoint_path
    
    def load(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a checkpoint (backward-compatible API).
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, loads latest.
        
        Returns:
            Loaded state dict
        """
        return self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint with metadata.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, loads latest.
            map_location: Device to map tensors to (e.g., 'cpu')
        
        Returns:
            Loaded checkpoint dict including metadata
        """
        if checkpoint_path is None:
            path = self.checkpoint_dir / "checkpoint_latest.pt"
        else:
            path = Path(checkpoint_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        data = torch.load(path, map_location=map_location, weights_only=False)
        logger.info(f"Checkpoint loaded from: {path}")
        
        return data
    
    def load_best(self, map_location: Optional[str] = None) -> Dict[str, Any]:
        """Load the best checkpoint"""
        best_path = self.checkpoint_dir / "checkpoint_best.pt"
        if not best_path.exists():
            raise FileNotFoundError("No best checkpoint found")
        return torch.load(best_path, map_location=map_location, weights_only=False)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones"""
        # Find all step/epoch checkpoints (not best/latest)
        checkpoints = sorted(
            [p for p in self.checkpoint_dir.glob("checkpoint_*.pt")
             if "best" not in p.name and "latest" not in p.name],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for checkpoint in checkpoints[self.keep_last_n:]:
            checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {checkpoint}")
    
    def get_all_checkpoints(self) -> list:
        """List all available checkpoints"""
        return sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
    
    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists"""
        return (self.checkpoint_dir / "checkpoint_latest.pt").exists()
