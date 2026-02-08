"""Checkpoint management for training"""
import torch
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
    
    def save(self, state_dict, epoch, step, is_best=False):
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(state_dict, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        shutil.copy(checkpoint_path, latest_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            shutil.copy(checkpoint_path, best_path)
            logger.info("New best checkpoint saved")
        
        self._cleanup_old_checkpoints()
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        for checkpoint in checkpoints[self.keep_last_n:]:
            checkpoint.unlink()
    
    def load(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return torch.load(checkpoint_path)
