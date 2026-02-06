"""Hyperparameter search spaces"""
from ray import tune

RESNET_SEARCH_SPACE = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([128, 256, 512]),
    "weight_decay": tune.loguniform(1e-5, 1e-3),
    "momentum": tune.uniform(0.8, 0.99),
}

TRANSFORMER_SEARCH_SPACE = {
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([64, 128, 256]),
    "weight_decay": tune.loguniform(1e-6, 1e-4),
}
