"""Ray Tune hyperparameter optimization"""
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def create_tune_config(base_config):
    """Create Ray Tune search space"""
    return {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
    }

def run_tuning(train_func, num_samples=10):
    """Run hyperparameter tuning"""
    scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )
    
    result = tune.run(
        train_func,
        config=create_tune_config({}),
        num_samples=num_samples,
        scheduler=scheduler
    )
    
    return result
