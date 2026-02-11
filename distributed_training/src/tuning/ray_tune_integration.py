"""
Ray Tune Hyperparameter Optimization

Integrates with the distributed trainer for automated hyperparameter search.
Uses ASHA scheduler for efficient early stopping of poor trials.
"""

import logging
from typing import Dict, Any, Optional

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune import TuneConfig

logger = logging.getLogger(__name__)


def create_tune_config(
    base_config: Dict[str, Any],
    search_space: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create Ray Tune search space merged with base config.
    
    Args:
        base_config: Base training config dict
        search_space: Custom search space overrides
    
    Returns:
        Config dict with tune search distributions
    """
    # Default search space
    default_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "momentum": tune.uniform(0.85, 0.99),
        "warmup_epochs": tune.choice([3, 5, 10]),
    }
    
    # Merge base config with search space
    config = {**base_config}
    space = search_space or default_space
    config.update(space)
    
    return config


def run_tuning(
    train_func,
    base_config: Optional[Dict[str, Any]] = None,
    search_space: Optional[Dict[str, Any]] = None,
    num_samples: int = 10,
    max_epochs: int = 100,
    grace_period: int = 10,
    num_gpu_workers: int = 1,
    gpus_per_worker: int = 1,
    metric: str = "val_accuracy",
    mode: str = "max",
):
    """
    Run hyperparameter tuning with ASHA early stopping.
    
    Args:
        train_func: Training function that accepts a config dict
        base_config: Base config to merge with search space
        search_space: Custom search space (uses default if None)
        num_samples: Number of hyperparameter configurations to try
        max_epochs: Maximum epochs per trial
        grace_period: Minimum epochs before trial can be stopped
        num_gpu_workers: Number of GPU workers per trial
        gpus_per_worker: GPUs per worker
        metric: Metric to optimize
        mode: "max" or "min"
    
    Returns:
        Ray Tune ResultGrid
    """
    base = base_config or {}
    tune_config = create_tune_config(base, search_space)
    
    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=max_epochs,
        grace_period=grace_period,
        reduction_factor=2,
    )
    
    scaling_config = ScalingConfig(
        num_workers=num_gpu_workers,
        use_gpu=True,
        resources_per_worker={"GPU": gpus_per_worker, "CPU": 4},
    )
    
    trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
    )
    
    tuner = tune.Tuner(
        trainer,
        param_space={"train_loop_config": tune_config},
        tune_config=TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            metric=metric,
            mode=mode,
        ),
        run_config=RunConfig(
            name="hyperparameter_tuning",
            checkpoint_config=CheckpointConfig(num_to_keep=2),
        ),
    )
    
    results = tuner.fit()
    
    # Log best result
    best = results.get_best_result(metric=metric, mode=mode)
    logger.info(f"Best trial config: {best.config}")
    logger.info(f"Best trial {metric}: {best.metrics.get(metric)}")
    
    return results
