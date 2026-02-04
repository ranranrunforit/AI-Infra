"""
Routing Module - Intelligent Request Routing for Model Serving

This module provides sophisticated routing strategies for distributing inference
requests across multiple model endpoints. Includes support for A/B testing,
canary deployments, and various load balancing algorithms.

Components:
    - IntelligentRouter: Multi-strategy request router
    - ABTestRouter: Statistical A/B testing for model variants
    - CanaryDeployment: Progressive rollout with automatic rollback
    - ModelEndpoint: Backend endpoint representation

Example:
    ```python
    from routing import IntelligentRouter, ModelEndpoint

    # Define endpoints
    endpoints = [
        ModelEndpoint(url="http://gpu1:8000", weight=2),
        ModelEndpoint(url="http://gpu2:8000", weight=1),
    ]

    # Create router
    router = IntelligentRouter(endpoints, strategy="weighted")

    # Route requests
    endpoint = await router.route(request)
    response = await endpoint.send_request(request)
    ```
"""

from .router import IntelligentRouter, ModelEndpoint, RoutingStrategy
from .ab_testing import ABTestRouter, Experiment, Variant
from .canary import CanaryDeployment, DeploymentState

__all__ = [
    "IntelligentRouter",
    "ModelEndpoint",
    "RoutingStrategy",
    "ABTestRouter",
    "Experiment",
    "Variant",
    "CanaryDeployment",
    "DeploymentState",
]

__version__ = "1.0.0"
