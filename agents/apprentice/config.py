"""
Configuration Management for Apprentice Agent

This module provides configuration management, hyperparameter tuning,
and model configuration for the CycleGAN-based artistic style transfer system.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CycleGANConfig:
    """Configuration for CycleGAN model architecture and training."""

    # Model architecture
    input_channels: int = 3
    output_channels: int = 3
    n_residual_blocks: int = 9
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0

    # Training parameters
    num_epochs: int = 200
    batch_size: int = 1
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999

    # Image processing
    image_size: int = 256

    # Training schedule
    lr_decay_start: int = 100
    lr_decay_factor: float = 0.5

    # Device configuration
    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleGANConfig":
        """Create config from dictionary."""
        return cls(**data)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Data directories
    data_root: str = "./data/apprentice"
    domain_a_dir: str = "domain_a"
    domain_b_dir: str = "domain_b"

    # Dataset parameters
    image_size: int = 256
    max_samples: Optional[int] = None

    # Data augmentation
    use_augmentation: bool = True
    augmentation_params: Dict[str, Any] = None

    # Satellite processing
    satellite_preprocessing: bool = True
    normalize_range: tuple = (0.0, 1.0)

    def __post_init__(self):
        if self.augmentation_params is None:
            self.augmentation_params = {
                "brightness": 0.1,
                "contrast": 0.1,
                "saturation": 0.1,
                "hue": 0.1,
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        """Create config from dictionary."""
        return cls(**data)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    # Checkpointing
    checkpoint_interval: int = 10
    save_best_model: bool = True
    keep_checkpoints: int = 5

    # Monitoring
    log_interval: int = 10
    sample_interval: int = 100

    # Early stopping
    use_early_stopping: bool = False
    patience: int = 20
    min_delta: float = 0.001

    # Validation
    validation_interval: int = 50
    validation_split: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**data)


@dataclass
class InferenceConfig:
    """Configuration for inference and style transfer."""

    # Model loading
    model_name: str = "best_model"
    checkpoint_path: Optional[str] = None

    # Style transfer
    direction: str = "A_to_B"  # "A_to_B" or "B_to_A"
    batch_size: int = 1

    # Output settings
    output_dir: str = "./outputs"
    save_intermediates: bool = False

    # Post-processing
    denormalize_output: bool = True
    apply_postprocessing: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceConfig":
        """Create config from dictionary."""
        return cls(**data)


@dataclass
class A2AConfig:
    """Configuration for A2A protocol integration."""

    # Server connection
    server_url: str = "http://localhost:8080"
    agent_name: str = "Apprentice_Agent"

    # Communication settings
    request_timeout: int = 30
    max_retries: int = 3

    # Collaboration settings
    enable_collaboration: bool = True
    collaboration_timeout: int = 300

    # Message handling
    message_queue_size: int = 100
    async_processing: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AConfig":
        """Create config from dictionary."""
        return cls(**data)


@dataclass
class ApprenticeConfig:
    """Main configuration class for Apprentice Agent."""

    # Component configs
    cyclegan: CycleGANConfig
    data: DataConfig
    training: TrainingConfig
    inference: InferenceConfig
    a2a: A2AConfig

    # General settings
    model_dir: str = "./models/apprentice"
    log_level: str = "INFO"
    random_seed: Optional[int] = 42

    # Performance settings
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2

    def __init__(self, **kwargs):
        # Initialize with defaults
        self.cyclegan = CycleGANConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.a2a = A2AConfig()

        # Override with provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary."""
        return {
            "cyclegan": self.cyclegan.to_dict(),
            "data": self.data.to_dict(),
            "training": self.training.to_dict(),
            "inference": self.inference.to_dict(),
            "a2a": self.a2a.to_dict(),
            "model_dir": self.model_dir,
            "log_level": self.log_level,
            "random_seed": self.random_seed,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprenticeConfig":
        """Create config from dictionary."""
        config = cls()

        # Load component configs
        if "cyclegan" in data:
            config.cyclegan = CycleGANConfig.from_dict(data["cyclegan"])
        if "data" in data:
            config.data = DataConfig.from_dict(data["data"])
        if "training" in data:
            config.training = TrainingConfig.from_dict(data["training"])
        if "inference" in data:
            config.inference = InferenceConfig.from_dict(data["inference"])
        if "a2a" in data:
            config.a2a = A2AConfig.from_dict(data["a2a"])

        # Load general settings
        for key in [
            "model_dir",
            "log_level",
            "random_seed",
            "num_workers",
            "pin_memory",
            "prefetch_factor",
        ]:
            if key in data:
                setattr(config, key, data[key])

        return config


class ConfigManager:
    """
    Manager class for handling configuration files and hyperparameter tuning.

    Provides utilities for:
    - Loading and saving configurations
    - Hyperparameter optimization
    - Configuration validation
    - Experiment tracking
    """

    def __init__(self, config_dir: str = "./config/apprentice"):
        """
        Initialize config manager.

        Args:
            config_dir: Directory for configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Default configuration
        self.default_config = ApprenticeConfig()

        logger.info(f"Config manager initialized with directory: {config_dir}")

    def load_config(self, config_name: str = "default") -> ApprenticeConfig:
        """
        Load configuration from file.

        Args:
            config_name: Name of configuration file (without extension)

        Returns:
            Loaded configuration
        """
        config_path = self.config_dir / f"{config_name}.json"

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                config = ApprenticeConfig.from_dict(data)
                logger.info(f"Loaded config: {config_name}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config {config_name}: {e}")
                return self.default_config
        else:
            logger.info(f"Config {config_name} not found, using defaults")
            return self.default_config

    def save_config(self, config: ApprenticeConfig, config_name: str = "default"):
        """
        Save configuration to file.

        Args:
            config: Configuration to save
            config_name: Name for configuration file
        """
        config_path = self.config_dir / f"{config_name}.json"

        try:
            with open(config_path, "w") as f:
                json.dump(config.to_dict(), f, indent=2)
            logger.info(f"Saved config: {config_name}")
        except Exception as e:
            logger.error(f"Failed to save config {config_name}: {e}")

    def create_experiment_config(
        self,
        base_config: ApprenticeConfig,
        experiment_name: str,
        hyperparams: Dict[str, Any],
    ) -> ApprenticeConfig:
        """
        Create a new configuration for hyperparameter experimentation.

        Args:
            base_config: Base configuration to modify
            experiment_name: Name for the experiment
            hyperparams: Hyperparameters to modify

        Returns:
            Modified configuration
        """
        # Deep copy the base config
        experiment_config = ApprenticeConfig.from_dict(base_config.to_dict())

        # Apply hyperparameter modifications
        self._apply_hyperparams(experiment_config, hyperparams)

        # Set experiment-specific settings
        experiment_config.model_dir = f"{base_config.model_dir}/{experiment_name}"

        return experiment_config

    def _apply_hyperparams(self, config: ApprenticeConfig, hyperparams: Dict[str, Any]):
        """Apply hyperparameter modifications to config."""
        for key, value in hyperparams.items():
            if "." in key:
                # Nested parameter (e.g., 'cyclegan.learning_rate')
                parts = key.split(".")
                obj = config
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        logger.warning(f"Invalid hyperparameter path: {key}")
                        return
                if hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)
                else:
                    logger.warning(f"Invalid hyperparameter: {key}")
            else:
                # Top-level parameter
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Invalid hyperparameter: {key}")

    def generate_hyperparameter_grid(
        self, base_config: ApprenticeConfig, param_ranges: Dict[str, List[Any]]
    ) -> List[ApprenticeConfig]:
        """
        Generate a grid of configurations for hyperparameter search.

        Args:
            base_config: Base configuration
            param_ranges: Dictionary of parameter names to lists of values

        Returns:
            List of configurations for grid search
        """
        from itertools import product

        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(product(*param_values))

        configs = []
        for i, combo in enumerate(combinations):
            hyperparams = dict(zip(param_names, combo))
            experiment_name = f"grid_search_{i}"
            config = self.create_experiment_config(
                base_config, experiment_name, hyperparams
            )
            configs.append(config)

        logger.info(f"Generated {len(configs)} configurations for grid search")
        return configs

    def validate_config(self, config: ApprenticeConfig) -> List[str]:
        """
        Validate configuration for consistency and correctness.

        Args:
            config: Configuration to validate

        Returns:
            List of validation error messages
        """
        errors = []

        # Validate CycleGAN config
        if config.cyclegan.image_size <= 0:
            errors.append("Image size must be positive")
        if config.cyclegan.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        if not (0 < config.cyclegan.lambda_cycle <= 100):
            errors.append("Lambda cycle should be between 0 and 100")

        # Validate data config
        if config.data.image_size != config.cyclegan.image_size:
            errors.append("Data and CycleGAN image sizes must match")

        # Validate training config
        if config.training.checkpoint_interval <= 0:
            errors.append("Checkpoint interval must be positive")

        # Validate paths
        if not Path(config.model_dir).parent.exists():
            errors.append(f"Model directory parent does not exist: {config.model_dir}")

        return errors

    def get_config_summary(self, config: ApprenticeConfig) -> Dict[str, Any]:
        """
        Get a summary of configuration settings.

        Args:
            config: Configuration to summarize

        Returns:
            Summary dictionary
        """
        return {
            "model_architecture": {
                "input_channels": config.cyclegan.input_channels,
                "residual_blocks": config.cyclegan.n_residual_blocks,
                "lambda_cycle": config.cyclegan.lambda_cycle,
                "lambda_identity": config.cyclegan.lambda_identity,
            },
            "training": {
                "epochs": config.cyclegan.num_epochs,
                "batch_size": config.cyclegan.batch_size,
                "learning_rate": config.cyclegan.learning_rate,
                "image_size": config.cyclegan.image_size,
            },
            "data": {
                "data_root": config.data.data_root,
                "image_size": config.data.image_size,
                "use_augmentation": config.data.use_augmentation,
            },
            "inference": {
                "model_name": config.inference.model_name,
                "direction": config.inference.direction,
                "batch_size": config.inference.batch_size,
            },
            "a2a": {
                "server_url": config.a2a.server_url,
                "agent_name": config.a2a.agent_name,
                "enable_collaboration": config.a2a.enable_collaboration,
            },
        }

    def export_config_for_experiment(
        self,
        config: ApprenticeConfig,
        experiment_name: str,
        output_dir: Optional[str] = None,
    ):
        """
        Export configuration for experiment tracking.

        Args:
            config: Configuration to export
            experiment_name: Name of the experiment
            output_dir: Output directory (optional)
        """
        if output_dir is None:
            output_dir = self.config_dir / "experiments"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full config
        config_path = output_dir / f"{experiment_name}_config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # Save summary
        summary_path = output_dir / f"{experiment_name}_summary.json"
        summary = self.get_config_summary(config)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save timestamp
        timestamp_path = output_dir / f"{experiment_name}_timestamp.txt"
        with open(timestamp_path, "w") as f:
            f.write(datetime.now().isoformat())

        logger.info(f"Exported config for experiment: {experiment_name}")


# Global config manager instance
config_manager = ConfigManager()


def get_default_config() -> ApprenticeConfig:
    """Get default configuration."""
    return config_manager.default_config


def load_config(config_name: str = "default") -> ApprenticeConfig:
    """Load configuration by name."""
    return config_manager.load_config(config_name)


def save_config(config: ApprenticeConfig, config_name: str = "default"):
    """Save configuration."""
    config_manager.save_config(config, config_name)
