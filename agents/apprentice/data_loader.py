"""
Data Loading Utilities for CycleGAN

This module provides data loading utilities for unpaired image-to-image translation,
including image preprocessing, dataset classes, and data loaders optimized for
artistic style transfer tasks.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ImageTransform:
    """Image transformation utilities for CycleGAN training."""

    def __init__(self, image_size: int = 256):
        self.image_size = image_size

        # Training transforms with augmentation
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.12), transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Test transforms without augmentation
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(image_size, transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __call__(self, image: Image.Image, is_train: bool = True) -> torch.Tensor:
        """Apply transformations to an image."""
        if is_train:
            return self.train_transform(image)
        else:
            return self.test_transform(image)


class UnpairedImageDataset(Dataset):
    """
    Dataset for unpaired image-to-image translation.

    This dataset loads images from two different domains (A and B) without pairing them,
    which is essential for CycleGAN training.
    """

    def __init__(
        self,
        root_dir: str,
        domain_a_dir: str = "domain_a",
        domain_b_dir: str = "domain_b",
        image_size: int = 256,
        is_train: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the unpaired dataset.

        Args:
            root_dir: Root directory containing domain subdirectories
            domain_a_dir: Subdirectory name for domain A images
            domain_b_dir: Subdirectory name for domain B images
            image_size: Size to resize images to
            is_train: Whether this is for training (affects transforms)
            max_samples: Maximum number of samples to use (None for all)
        """
        self.root_dir = Path(root_dir)
        self.domain_a_dir = self.root_dir / domain_a_dir
        self.domain_b_dir = self.root_dir / domain_b_dir
        self.image_size = image_size
        self.is_train = is_train
        self.max_samples = max_samples

        # Initialize transforms
        self.transform = ImageTransform(image_size)

        # Load image paths
        self.domain_a_paths = self._get_image_paths(self.domain_a_dir)
        self.domain_b_paths = self._get_image_paths(self.domain_b_dir)

        # Limit samples if specified
        if max_samples is not None:
            self.domain_a_paths = self.domain_a_paths[:max_samples]
            self.domain_b_paths = self.domain_b_paths[:max_samples]

        # Use the smaller dataset size to ensure balanced sampling
        self.dataset_size = min(len(self.domain_a_paths), len(self.domain_b_paths))

        logger.info(f"Loaded {len(self.domain_a_paths)} images from domain A")
        logger.info(f"Loaded {len(self.domain_b_paths)} images from domain B")
        logger.info(f"Dataset size: {self.dataset_size}")

    def _get_image_paths(self, directory: Path) -> List[Path]:
        """Get all image file paths from a directory."""
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return []

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = []

        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_paths.append(file_path)

        return sorted(image_paths)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a pair of images from different domains.

        Args:
            index: Index of the sample

        Returns:
            Tuple of (domain_a_image, domain_b_image)
        """
        # Get images from both domains
        domain_a_path = self.domain_a_paths[index % len(self.domain_a_paths)]
        domain_b_path = self.domain_b_paths[index % len(self.domain_b_paths)]

        # Load and transform images
        try:
            domain_a_image = self._load_image(domain_a_path)
            domain_b_image = self._load_image(domain_b_path)

            return domain_a_image, domain_b_image

        except Exception as e:
            logger.error(f"Error loading images {domain_a_path}, {domain_b_path}: {e}")
            # Return dummy images on error
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            return dummy_image, dummy_image

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and transform a single image."""
        try:
            image = Image.open(path).convert("RGB")
            return self.transform(image, self.is_train)
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            # Return a blank image on error
            return torch.zeros(3, self.image_size, self.image_size)


class SatelliteArtDataset(Dataset):
    """
    Specialized dataset for satellite imagery to artwork translation.

    This dataset handles satellite images as domain A and artistic images as domain B,
    with preprocessing optimized for geospatial data.
    """

    def __init__(
        self,
        satellite_dir: str,
        artwork_dir: str,
        image_size: int = 256,
        is_train: bool = True,
        satellite_preprocessing: bool = True,
    ):
        """
        Initialize the satellite-art dataset.

        Args:
            satellite_dir: Directory containing satellite images
            artwork_dir: Directory containing artwork images
            image_size: Size to resize images to
            is_train: Whether this is for training
            satellite_preprocessing: Whether to apply satellite-specific preprocessing
        """
        self.satellite_dir = Path(satellite_dir)
        self.artwork_dir = Path(artwork_dir)
        self.image_size = image_size
        self.is_train = is_train
        self.satellite_preprocessing = satellite_preprocessing

        # Initialize transforms
        self.satellite_transform = self._get_satellite_transform()
        self.artwork_transform = ImageTransform(image_size)

        # Load image paths
        self.satellite_paths = self._get_image_paths(self.satellite_dir)
        self.artwork_paths = self._get_image_paths(self.artwork_dir)

        # Balance datasets
        min_size = min(len(self.satellite_paths), len(self.artwork_paths))
        self.satellite_paths = self.satellite_paths[:min_size]
        self.artwork_paths = self.artwork_paths[:min_size]
        self.dataset_size = min_size

        logger.info(f"Loaded satellite-art dataset with {self.dataset_size} samples")

    def _get_satellite_transform(self) -> transforms.Compose:
        """Get transforms optimized for satellite imagery."""
        if self.satellite_preprocessing:
            return transforms.Compose(
                [
                    transforms.Resize(self.image_size, transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    # Normalize satellite imagery (typically 0-255 range)
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            # Use test transform (no augmentation) when preprocessing is disabled
            temp_transform = ImageTransform(self.image_size)
            return temp_transform.test_transform

    def _get_image_paths(self, directory: Path) -> List[Path]:
        """Get image paths from directory."""
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return []

        image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        image_paths = []

        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_paths.append(file_path)

        return sorted(image_paths)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a satellite-artwork pair."""
        satellite_path = self.satellite_paths[index % len(self.satellite_paths)]
        artwork_path = self.artwork_paths[index % len(self.artwork_paths)]

        try:
            # Load satellite image
            satellite_image = Image.open(satellite_path).convert("RGB")
            satellite_tensor = self.satellite_transform(satellite_image)

            # Load artwork image
            artwork_image = Image.open(artwork_path).convert("RGB")
            artwork_tensor = self.artwork_transform(artwork_image, self.is_train)

            return satellite_tensor, artwork_tensor

        except Exception as e:
            logger.error(f"Error loading satellite-artwork pair: {e}")
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            return dummy_image, dummy_image


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create a DataLoader with optimized settings.

    Args:
        dataset: The dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def create_unpaired_data_loaders(
    data_dir: str,
    domain_a_dir: str = "domain_a",
    domain_b_dir: str = "domain_b",
    image_size: int = 256,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders for unpaired datasets.

    Args:
        data_dir: Root data directory
        domain_a_dir: Domain A subdirectory
        domain_b_dir: Domain B subdirectory
        image_size: Image size for transforms
        batch_size: Batch size
        max_samples: Maximum samples per dataset

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create training dataset
    train_dataset = UnpairedImageDataset(
        root_dir=data_dir,
        domain_a_dir=domain_a_dir,
        domain_b_dir=domain_b_dir,
        image_size=image_size,
        is_train=True,
        max_samples=max_samples,
    )

    # Create test dataset (smaller subset)
    test_dataset = UnpairedImageDataset(
        root_dir=data_dir,
        domain_a_dir=domain_a_dir,
        domain_b_dir=domain_b_dir,
        image_size=image_size,
        is_train=False,
        max_samples=min(max_samples // 10 if max_samples else 100, 100),
    )

    # Create data loaders
    train_loader = create_data_loader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = create_data_loader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def create_satellite_art_data_loaders(
    satellite_dir: str,
    artwork_dir: str,
    image_size: int = 256,
    batch_size: int = 1,
    satellite_preprocessing: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for satellite to artwork translation.

    Args:
        satellite_dir: Directory with satellite images
        artwork_dir: Directory with artwork images
        image_size: Image size
        batch_size: Batch size
        satellite_preprocessing: Whether to use satellite preprocessing

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create training dataset
    train_dataset = SatelliteArtDataset(
        satellite_dir=satellite_dir,
        artwork_dir=artwork_dir,
        image_size=image_size,
        is_train=True,
        satellite_preprocessing=satellite_preprocessing,
    )

    # Create test dataset
    test_dataset = SatelliteArtDataset(
        satellite_dir=satellite_dir,
        artwork_dir=artwork_dir,
        image_size=image_size,
        is_train=False,
        satellite_preprocessing=satellite_preprocessing,
    )

    # Create data loaders
    train_loader = create_data_loader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = create_data_loader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class DataLoaderManager:
    """
    Manager class for handling multiple data loaders and datasets.

    This class provides utilities for managing different types of datasets
    and switching between them during training.
    """

    def __init__(self):
        self.loaders = {}
        self.current_loader = None

    def add_loader(self, name: str, loader: DataLoader):
        """Add a data loader."""
        self.loaders[name] = loader
        logger.info(f"Added data loader: {name}")

    def get_loader(self, name: str) -> Optional[DataLoader]:
        """Get a data loader by name."""
        return self.loaders.get(name)

    def set_current_loader(self, name: str):
        """Set the current active loader."""
        if name in self.loaders:
            self.current_loader = self.loaders[name]
            logger.info(f"Set current loader to: {name}")
        else:
            logger.error(f"Loader {name} not found")

    def get_current_loader(self) -> Optional[DataLoader]:
        """Get the current active loader."""
        return self.current_loader

    def list_loaders(self) -> List[str]:
        """List all available loaders."""
        return list(self.loaders.keys())
