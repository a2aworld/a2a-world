#!/usr/bin/env python3
"""
Training Script for Apprentice Agent CycleGAN

This script provides command-line interface for training CycleGAN models
for artistic style transfer in the Apprentice Agent.
"""

import argparse
import logging
import sys
import torch
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .cyclegan_model import CycleGAN
from .data_loader import create_unpaired_data_loaders, create_satellite_art_data_loaders
from .training_pipeline import TrainingPipeline, TrainingConfig
from .config import load_config, ApprenticeConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CycleGAN for artistic style transfer"
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing training data",
    )
    parser.add_argument(
        "--domain-a",
        type=str,
        default="domain_a",
        help="Subdirectory name for domain A images",
    )
    parser.add_argument(
        "--domain-b",
        type=str,
        default="domain_b",
        help="Subdirectory name for domain B images",
    )

    # Model arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/apprentice",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--config", type=str, default="default", help="Configuration name to use"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="Image size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0002,
        help="Learning rate for optimizers",
    )

    # Data processing arguments
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Maximum number of samples to use"
    )
    parser.add_argument(
        "--satellite-preprocessing",
        action="store_true",
        help="Use satellite-specific preprocessing",
    )

    # Training control arguments
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Log progress every N batches"
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for training (auto, cpu, cuda)",
    )

    # Experiment arguments
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this training experiment",
    )

    return parser.parse_args()


def setup_training(args):
    """
    Set up training components based on arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (model, train_loader, config)
    """
    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    config.cyclegan.num_epochs = args.epochs
    config.cyclegan.batch_size = args.batch_size
    config.cyclegan.image_size = args.image_size
    config.cyclegan.learning_rate = args.learning_rate
    config.data.image_size = args.image_size
    config.data.max_samples = args.max_samples
    config.training.checkpoint_interval = args.checkpoint_interval
    config.training.log_interval = args.log_interval

    # Set device
    if args.device != "auto":
        config.cyclegan.device = args.device

    # Update model directory for experiment
    if args.experiment_name:
        config.model_dir = str(Path(args.model_dir) / args.experiment_name)

    # Create model directory
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)

    # Initialize CycleGAN model
    device = config.cyclegan.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CycleGAN(
        device=torch.device(device),
        lambda_cycle=config.cyclegan.lambda_cycle,
        lambda_identity=config.cyclegan.lambda_identity,
    )

    # Create data loaders
    if args.satellite_preprocessing:
        train_loader, _ = create_satellite_art_data_loaders(
            satellite_dir=str(Path(args.data_dir) / args.domain_a),
            artwork_dir=str(Path(args.data_dir) / args.domain_b),
            image_size=config.data.image_size,
            batch_size=config.cyclegan.batch_size,
            satellite_preprocessing=True,
        )
    else:
        train_loader, _ = create_unpaired_data_loaders(
            data_dir=args.data_dir,
            domain_a_dir=args.domain_a,
            domain_b_dir=args.domain_b,
            image_size=config.data.image_size,
            batch_size=config.cyclegan.batch_size,
            max_samples=config.data.max_samples,
        )

    return model, train_loader, config


def main():
    """Main training function."""
    args = parse_args()

    logger.info("Starting CycleGAN training for Apprentice Agent")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Training for {args.epochs} epochs with batch size {args.batch_size}")

    try:
        # Setup training components
        model, train_loader, config = setup_training(args)

        # Create training configuration
        training_config = TrainingConfig(
            num_epochs=config.cyclegan.num_epochs,
            batch_size=config.cyclegan.batch_size,
            learning_rate=config.cyclegan.learning_rate,
            sample_interval=100,
            checkpoint_interval=config.training.checkpoint_interval,
            log_interval=config.training.log_interval,
        )

        # Initialize training pipeline
        pipeline = TrainingPipeline(model, Path(config.model_dir))

        # Start training
        logger.info("Training pipeline initialized, starting training...")

        results = pipeline.train(
            train_loader=train_loader, config=training_config, resume_from=args.resume
        )

        # Log final results
        logger.info("Training completed successfully!")
        logger.info(f"Final loss: {results.get('best_loss', 'N/A')}")
        logger.info(
            f"Total training time: {results.get('total_training_time', 0):.2f} seconds"
        )
        logger.info(f"Model saved to: {results.get('model_dir', 'N/A')}")

        # Save final configuration
        from .config import save_config

        save_config(
            config, f"{args.experiment_name}_final" if args.experiment_name else "final"
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
