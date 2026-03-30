"""
Model registry for comparison models.

This module provides:
- ModelRegistry: Central registry for model lookup
- @register_model: Decorator for automatic model registration
- Factory function: create_model()

Usage:
    from experiments.models import create_model, list_models

    # Create model from config
    model = create_model(config, encoder)

    # List all registered models
    models = list_models()
"""

from typing import Dict, Any, Type, Callable
import torch.nn as nn

from .base import BaseComparisonModel


class ModelRegistry:
    """
    Central registry for comparison models.

    Provides dynamic model lookup and instantiation based on model name.
    """

    _registry: Dict[str, Type[BaseComparisonModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseComparisonModel]) -> None:
        """
        Register a model class.

        Args:
            name: Model name (e.g., "MAML", "ProtoNet")
            model_class: Model class inheriting from BaseComparisonModel
        """
        if name in cls._registry:
            raise ValueError(f"Model '{name}' is already registered")

        if not issubclass(model_class, BaseComparisonModel):
            raise TypeError(f"Model class must inherit from BaseComparisonModel")

        cls._registry[name] = model_class
        print(f" Registered model: {name}")

    @classmethod
    def get(cls, name: str) -> Type[BaseComparisonModel]:
        """
        Get model class by name.

        Args:
            name: Model name

        Returns:
            Model class

        Raises:
            ValueError: If model not found
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )

        return cls._registry[name]

    @classmethod
    def list_models(cls) -> Dict[str, Type[BaseComparisonModel]]:
        """List all registered models."""
        return cls._registry.copy()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered."""
        return name in cls._registry


def register_model(name: str) -> Callable:
    """
    Decorator for automatic model registration.

    Usage:
        @register_model("MAML")
        class MAMLModel(BaseComparisonModel):
            ...

    Args:
        name: Model name for registry

    Returns:
        Decorator function
    """
    def decorator(model_class: Type[BaseComparisonModel]) -> Type[BaseComparisonModel]:
        ModelRegistry.register(name, model_class)
        return model_class

    return decorator


def create_model(config: Dict[str, Any], encoder: nn.Module) -> BaseComparisonModel:
    """
    Factory function to create model from config.

    Args:
        config: Model configuration from YAML
        encoder: Shared encoder (transformer_b_small)

    Returns:
        Instantiated model

    Raises:
        ValueError: If model name not found in registry
    """
    model_name = config.get('model', {}).get('name', None)

    if model_name is None:
        raise ValueError("Model name not specified in config")

    model_class = ModelRegistry.get(model_name)
    return model_class(encoder, config)


def list_models() -> list:
    """
    List all registered model names.

    Returns:
        List of model names
    """
    return list(ModelRegistry.list_models().keys())


# Import all model implementations to trigger registration
# Models will be imported in the order: base models → complex models
try:
    from .maml import MAMLModel
except ImportError:
    pass  # Model not yet implemented

try:
    from .protonet import ProtoNetModel
except ImportError:
    pass  # Model not yet implemented

try:
    from .tsa_maml import TSAMAMLModel
except ImportError:
    pass  # Model not yet implemented

try:
    from .joint_training import JointTrainingModel
except ImportError:
    pass  # Model not yet implemented

try:
    from .logreg import LogRegModel
except ImportError:
    pass  # Model not yet implemented


# Public API
__all__ = [
    'BaseComparisonModel',
    'ModelRegistry',
    'register_model',
    'create_model',
    'list_models',
]
