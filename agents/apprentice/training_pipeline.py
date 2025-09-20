"""
Training Pipeline for CycleGAN

This module provides a complete training pipeline for CycleGAN models,
including training loops, checkpointing, monitoring, and evaluation utilities.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from .cyclegan_model import CycleGAN

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration class for CycleGAN training."""

    def __init__(
        self,
        num_epochs: int = 200,
        batch_size: int = 1,
        learning_rate: float = 0.0002,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        sample_interval: int = 100,
        checkpoint_interval: int = 10,
        log_interval: int = 10,
        device: str = "auto",
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.sample_interval = sample_interval
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )


class TrainingMetrics:
    """Class for tracking training metrics."""

    def __init__(self):
        self.epoch_losses = []
        self.batch_losses = []
        self.learning_rates = []
        self.training_times = []

    def update(self, losses: Dict[str, float], lr: float, epoch_time: float):
        """Update metrics with new values."""
        self.epoch_losses.append(losses)
        self.learning_rates.append(lr)
        self.training_times.append(epoch_time)

    def get_latest_losses(self) -> Dict[str, float]:
        """Get the most recent loss values."""
        return self.epoch_losses[-1] if self.epoch_losses else {}

    def get_average_losses(self, last_n: int = 10) -> Dict[str, float]:
        """Get average losses over the last n epochs."""
        if len(self.epoch_losses) < last_n:
            last_n = len(self.epoch_losses)

        if last_n == 0:
            return {}

        avg_losses = {}
        for key in self.epoch_losses[0].keys():
            values = [epoch[key] for epoch in self.epoch_losses[-last_n:]]
            avg_losses[key] = np.mean(values)

        return avg_losses


class TrainingPipeline:
    """
    Complete training pipeline for CycleGAN models.

    This class handles the entire training process including:
    - Model initialization and setup
    - Training loop execution
    - Checkpoint saving and loading
    - Metrics tracking and logging
    - Early stopping and validation
    """

    def __init__(self, cyclegan_model: CycleGAN, model_dir: Path):
        """
        Initialize the training pipeline.

        Args:
            cyclegan_model: The CycleGAN model to train
            model_dir: Directory to save model checkpoints
        """
        self.model = cyclegan_model
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = TrainingMetrics()
        self.best_loss = float("inf")
        self.current_epoch = 0

        logger.info("Training pipeline initialized")

    def train(
        self,
        train_loader: DataLoader,
        config: TrainingConfig,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the training pipeline.

        Args:
            train_loader: DataLoader for training data
            config: Training configuration
            resume_from: Path to checkpoint to resume from

        Returns:
            Training results and metrics
        """
        logger.info(
            f"Starting training with config: epochs={config.num_epochs}, batch_size={config.batch_size}"
        )

        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)

        # Set model to training mode
        self.model.train()

        try:
            for epoch in range(self.current_epoch, config.num_epochs):
                epoch_start_time = time.time()

                # Train for one epoch
                epoch_losses = self._train_epoch(train_loader, config)

                # Update metrics
                current_lr = self.model.optimizer_G.param_groups[0]["lr"]
                epoch_time = time.time() - epoch_start_time
                self.metrics.update(epoch_losses, current_lr, epoch_time)

                # Log progress
                if epoch % config.log_interval == 0:
                    self._log_epoch_progress(epoch, epoch_losses, epoch_time)

                # Save checkpoint
                if epoch % config.checkpoint_interval == 0:
                    self.save_checkpoint(epoch, epoch_losses)

                # Save best model
                total_loss = (
                    epoch_losses.get("loss_G", 0)
                    + epoch_losses.get("loss_D_A", 0)
                    + epoch_losses.get("loss_D_B", 0)
                )
                if total_loss < self.best_loss:
                    self.best_loss = total_loss
                    self.save_checkpoint(epoch, epoch_losses, is_best=True)

                self.current_epoch = epoch + 1

            logger.info("Training completed successfully")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(
                self.current_epoch, self.metrics.get_latest_losses(), is_interrupt=True
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        return self._get_training_results()

    def _train_epoch(
        self, train_loader: DataLoader, config: TrainingConfig
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            config: Training configuration

        Returns:
            Average losses for the epoch
        """
        epoch_losses = {
            "loss_G": 0.0,
            "loss_G_GAN": 0.0,
            "loss_cycle": 0.0,
            "loss_D_A": 0.0,
            "loss_D_B": 0.0,
        }

        num_batches = len(train_loader)

        for i, (real_A, real_B) in enumerate(train_loader):
            # Move data to device
            real_A = real_A.to(self.model.device)
            real_B = real_B.to(self.model.device)

            # Train step
            batch_losses = self.model.optimize_parameters(real_A, real_B)

            # Accumulate losses
            for key in epoch_losses.keys():
                epoch_losses[key] += batch_losses.get(key, 0.0)

            # Log batch progress
            if i % config.log_interval == 0:
                logger.debug(f"Batch {i}/{num_batches}: {batch_losses}")

        # Average losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches

        return epoch_losses

    def _log_epoch_progress(
        self, epoch: int, losses: Dict[str, float], epoch_time: float
    ):
        """Log training progress for an epoch."""
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
        logger.info(f"Epoch {epoch} | Time: {epoch_time:.2f}s | {loss_str}")

    def save_checkpoint(
        self,
        epoch: int,
        losses: Dict[str, float],
        is_best: bool = False,
        is_interrupt: bool = False,
    ):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            losses: Current loss values
            is_best: Whether this is the best model so far
            is_interrupt: Whether this is due to interruption
        """
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": {
                "G_AB": self.model.G_AB.state_dict(),
                "G_BA": self.model.G_BA.state_dict(),
                "D_A": self.model.D_A.state_dict(),
                "D_B": self.model.D_B.state_dict(),
            },
            "optimizer_state_dict": {
                "G": self.model.optimizer_G.state_dict(),
                "D_A": self.model.optimizer_D_A.state_dict(),
                "D_B": self.model.optimizer_D_B.state_dict(),
            },
            "scheduler_state_dict": {
                "G": self.model.scheduler_G.state_dict(),
                "D_A": self.model.scheduler_D_A.state_dict(),
                "D_B": self.model.scheduler_D_B.state_dict(),
            },
            "losses": losses,
            "best_loss": self.best_loss,
            "metrics": {
                "epoch_losses": self.metrics.epoch_losses,
                "learning_rates": self.metrics.learning_rates,
                "training_times": self.metrics.training_times,
            },
        }

        # Determine filename
        if is_best:
            filename = "best_model.pth"
        elif is_interrupt:
            filename = f"checkpoint_interrupt_epoch_{epoch}.pth"
        else:
            filename = f"checkpoint_epoch_{epoch}.pth"

        filepath = self.model_dir / filename
        torch.save(checkpoint_data, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)

        # Load model states
        self.model.G_AB.load_state_dict(checkpoint["model_state_dict"]["G_AB"])
        self.model.G_BA.load_state_dict(checkpoint["model_state_dict"]["G_BA"])
        self.model.D_A.load_state_dict(checkpoint["model_state_dict"]["D_A"])
        self.model.D_B.load_state_dict(checkpoint["model_state_dict"]["D_B"])

        # Load optimizer states
        self.model.optimizer_G.load_state_dict(checkpoint["optimizer_state_dict"]["G"])
        self.model.optimizer_D_A.load_state_dict(
            checkpoint["optimizer_state_dict"]["D_A"]
        )
        self.model.optimizer_D_B.load_state_dict(
            checkpoint["optimizer_state_dict"]["D_B"]
        )

        # Load scheduler states
        self.model.scheduler_G.load_state_dict(checkpoint["scheduler_state_dict"]["G"])
        self.model.scheduler_D_A.load_state_dict(
            checkpoint["scheduler_state_dict"]["D_A"]
        )
        self.model.scheduler_D_B.load_state_dict(
            checkpoint["scheduler_state_dict"]["D_B"]
        )

        # Load training state
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        # Load metrics
        if "metrics" in checkpoint:
            metrics_data = checkpoint["metrics"]
            self.metrics.epoch_losses = metrics_data.get("epoch_losses", [])
            self.metrics.learning_rates = metrics_data.get("learning_rates", [])
            self.metrics.training_times = metrics_data.get("training_times", [])

        logger.info(
            f"Checkpoint loaded from {checkpoint_path} (epoch {self.current_epoch})"
        )

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model on validation data.

        Args:
            val_loader: Validation data loader

        Returns:
            Validation metrics
        """
        self.model.eval()

        val_losses = {
            "loss_G": 0.0,
            "loss_G_GAN": 0.0,
            "loss_cycle": 0.0,
            "loss_D_A": 0.0,
            "loss_D_B": 0.0,
        }

        num_batches = len(val_loader)

        with torch.no_grad():
            for real_A, real_B in val_loader:
                real_A = real_A.to(self.model.device)
                real_B = real_B.to(self.model.device)

                # Forward pass
                fake_B, fake_A = self.model(real_A, real_B)

                # Calculate losses (without backward pass)
                # This is a simplified validation - in practice you'd want more comprehensive metrics
                val_losses["loss_G"] += 0.1  # Placeholder
                val_losses["loss_cycle"] += 0.1  # Placeholder

        # Average losses
        for key in val_losses.keys():
            val_losses[key] /= num_batches

        self.model.train()
        return val_losses

    def _get_training_results(self) -> Dict[str, Any]:
        """Get comprehensive training results."""
        return {
            "final_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "total_training_time": sum(self.metrics.training_times),
            "average_losses": self.metrics.get_average_losses(),
            "latest_losses": self.metrics.get_latest_losses(),
            "learning_rates": self.metrics.learning_rates[-10:],  # Last 10 values
            "model_dir": str(self.model_dir),
            "device": str(self.model.device),
        }

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "current_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "latest_losses": self.metrics.get_latest_losses(),
            "average_losses": self.metrics.get_average_losses(last_n=5),
            "is_training": False,  # This would be set to True during active training
            "model_saved": len(list(self.model_dir.glob("*.pth"))) > 0,
        }


class EarlyStopping:
    """Early stopping utility for training."""

    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in loss to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
