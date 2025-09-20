"""
Style Transfer Module for CycleGAN

This module provides inference capabilities for applying trained CycleGAN models
to perform artistic style transfer on new images, including satellite imagery.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

from .cyclegan_model import CycleGAN
from .data_loader import ImageTransform

logger = logging.getLogger(__name__)


class StyleTransfer:
    """
    Style transfer inference engine using trained CycleGAN models.

    This class handles:
    - Loading trained models
    - Preprocessing input images
    - Applying style transfer
    - Postprocessing and saving results
    - Batch processing capabilities
    """

    def __init__(self, cyclegan_model: CycleGAN, model_dir: Path):
        """
        Initialize the style transfer engine.

        Args:
            cyclegan_model: The CycleGAN model instance
            model_dir: Directory containing trained models
        """
        self.model = cyclegan_model
        self.model_dir = Path(model_dir)
        self.loaded_models = {}  # Cache for loaded model checkpoints

        # Image preprocessing
        self.transform = ImageTransform(image_size=256)

        # Set model to evaluation mode
        self.model.eval()

        logger.info("Style transfer engine initialized")

    def load_model(self, model_name: str) -> bool:
        """
        Load a trained model checkpoint.

        Args:
            model_name: Name of the model checkpoint file

        Returns:
            True if model loaded successfully
        """
        model_path = self.model_dir / model_name

        if not model_path.exists():
            logger.error(f"Model checkpoint not found: {model_path}")
            return False

        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.model.device)

            # Load model states
            self.model.G_AB.load_state_dict(checkpoint["model_state_dict"]["G_AB"])
            self.model.G_BA.load_state_dict(checkpoint["model_state_dict"]["G_BA"])

            # Cache the loaded model
            self.loaded_models[model_name] = checkpoint

            logger.info(f"Model loaded successfully: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def apply_style(
        self,
        input_path: str,
        style_name: str = "default",
        output_dir: Optional[str] = None,
        direction: str = "A_to_B",
    ) -> str:
        """
        Apply style transfer to an input image.

        Args:
            input_path: Path to input image
            style_name: Name of the style/model to use
            output_dir: Directory to save output (optional)
            direction: Translation direction ("A_to_B" or "B_to_A")

        Returns:
            Path to the generated image
        """
        # Load the model if not already loaded
        if style_name not in self.loaded_models:
            model_filename = f"{style_name}.pth"
            if not self.load_model(model_filename):
                # Try alternative naming
                model_filename = f"best_model.pth"
                if not self.load_model(model_filename):
                    raise ValueError(f"Could not load model for style: {style_name}")

        # Load and preprocess input image
        input_tensor = self._load_input_image(input_path)

        # Apply style transfer
        with torch.no_grad():
            if direction == "A_to_B":
                output_tensor = self.model.G_AB(input_tensor)
            elif direction == "B_to_A":
                output_tensor = self.model.G_BA(input_tensor)
            else:
                raise ValueError(f"Invalid direction: {direction}")

        # Postprocess and save
        output_path = self._save_output_image(
            output_tensor, input_path, style_name, output_dir, direction
        )

        logger.info(f"Style transfer completed: {input_path} -> {output_path}")
        return output_path

    def batch_style_transfer(
        self,
        input_paths: List[str],
        style_name: str = "default",
        output_dir: Optional[str] = None,
        direction: str = "A_to_B",
        batch_size: int = 4,
    ) -> List[str]:
        """
        Apply style transfer to multiple images in batches.

        Args:
            input_paths: List of input image paths
            style_name: Name of the style/model to use
            output_dir: Directory to save outputs
            direction: Translation direction
            batch_size: Number of images to process simultaneously

        Returns:
            List of output image paths
        """
        # Load the model
        if style_name not in self.loaded_models:
            model_filename = f"{style_name}.pth"
            if not self.load_model(model_filename):
                model_filename = f"best_model.pth"
                if not self.load_model(model_filename):
                    raise ValueError(f"Could not load model for style: {style_name}")

        output_paths = []

        # Process in batches
        for i in range(0, len(input_paths), batch_size):
            batch_paths = input_paths[i : i + batch_size]

            try:
                # Load batch of images
                batch_tensors = []
                for path in batch_paths:
                    tensor = self._load_input_image(path)
                    batch_tensors.append(tensor)

                batch_input = torch.stack(batch_tensors)

                # Apply style transfer
                with torch.no_grad():
                    if direction == "A_to_B":
                        batch_output = self.model.G_AB(batch_input)
                    else:
                        batch_output = self.model.G_BA(batch_input)

                # Save each output
                for j, (input_path, output_tensor) in enumerate(
                    zip(batch_paths, batch_output)
                ):
                    output_path = self._save_output_image(
                        output_tensor.unsqueeze(0),
                        input_path,
                        style_name,
                        output_dir,
                        direction,
                        batch_index=i + j,
                    )
                    output_paths.append(output_path)

                logger.info(
                    f"Processed batch {i//batch_size + 1}/{(len(input_paths) + batch_size - 1)//batch_size}"
                )

            except Exception as e:
                logger.error(f"Failed to process batch starting at index {i}: {e}")
                # Continue with next batch

        return output_paths

    def _load_input_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an input image.

        Args:
            image_path: Path to the input image

        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Apply preprocessing
            tensor = self.transform(image, is_train=False)

            # Add batch dimension
            tensor = tensor.unsqueeze(0).to(self.model.device)

            return tensor

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

    def _save_output_image(
        self,
        output_tensor: torch.Tensor,
        input_path: str,
        style_name: str,
        output_dir: Optional[str] = None,
        direction: str = "A_to_B",
        batch_index: Optional[int] = None,
    ) -> str:
        """
        Postprocess and save the output image.

        Args:
            output_tensor: Generated image tensor
            input_path: Original input path
            style_name: Style name used
            output_dir: Output directory
            direction: Translation direction
            batch_index: Index for batch processing

        Returns:
            Path to saved image
        """
        try:
            # Remove batch dimension if present
            if output_tensor.dim() == 4:
                output_tensor = output_tensor.squeeze(0)

            # Denormalize from [-1, 1] to [0, 1]
            output_tensor = (output_tensor + 1) / 2
            output_tensor = torch.clamp(output_tensor, 0, 1)

            # Convert to PIL Image
            output_image = TF.to_pil_image(output_tensor.cpu())

            # Determine output path
            if output_dir is None:
                output_dir = self.model_dir / "outputs"
            else:
                output_dir = Path(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            # Create output filename
            input_filename = Path(input_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if batch_index is not None:
                output_filename = f"{input_filename}_{style_name}_{direction}_{timestamp}_{batch_index}.png"
            else:
                output_filename = (
                    f"{input_filename}_{style_name}_{direction}_{timestamp}.png"
                )

            output_path = output_dir / output_filename

            # Save image
            output_image.save(output_path)

            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to save output image: {e}")
            raise

    def get_available_styles(self) -> List[str]:
        """
        Get list of available trained model styles.

        Returns:
            List of available style names
        """
        model_files = list(self.model_dir.glob("*.pth"))
        style_names = []

        for model_file in model_files:
            # Extract style name from filename
            name = model_file.stem
            if name.startswith("checkpoint_"):
                # Skip checkpoint files
                continue
            elif name == "best_model":
                style_names.append("default")
            else:
                style_names.append(name)

        return sorted(list(set(style_names)))

    def satellite_to_artwork(
        self,
        satellite_path: str,
        style_name: str = "default",
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Specialized method for converting satellite imagery to artwork.

        Args:
            satellite_path: Path to satellite image
            style_name: Artistic style to apply
            output_dir: Output directory

        Returns:
            Path to generated artwork
        """
        logger.info(f"Converting satellite image to artwork: {satellite_path}")

        # Apply style transfer (assuming satellite is domain A, artwork is domain B)
        return self.apply_style(
            input_path=satellite_path,
            style_name=style_name,
            output_dir=output_dir,
            direction="A_to_B",
        )

    def artwork_to_satellite(
        self,
        artwork_path: str,
        style_name: str = "default",
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Specialized method for converting artwork to satellite-like imagery.

        Args:
            artwork_path: Path to artwork image
            style_name: Style model to use
            output_dir: Output directory

        Returns:
            Path to generated satellite-like image
        """
        logger.info(f"Converting artwork to satellite imagery: {artwork_path}")

        # Apply reverse style transfer
        return self.apply_style(
            input_path=artwork_path,
            style_name=style_name,
            output_dir=output_dir,
            direction="B_to_A",
        )

    def blend_styles(
        self,
        input_path: str,
        style_names: List[str],
        weights: Optional[List[float]] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Blend multiple artistic styles on a single input image.

        Args:
            input_path: Path to input image
            style_names: List of style names to blend
            weights: Weights for each style (normalized automatically)
            output_dir: Output directory

        Returns:
            Path to blended result
        """
        if weights is None:
            weights = [1.0 / len(style_names)] * len(style_names)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]

        logger.info(f"Blending styles {style_names} with weights {weights}")

        # Load input image
        input_tensor = self._load_input_image(input_path)

        blended_output = torch.zeros_like(input_tensor)

        # Apply each style and accumulate
        for style_name, weight in zip(style_names, weights):
            # Load style model
            if style_name not in self.loaded_models:
                model_filename = f"{style_name}.pth"
                if not self.load_model(model_filename):
                    logger.warning(f"Could not load style {style_name}, skipping")
                    continue

            # Apply style transfer
            with torch.no_grad():
                styled_tensor = self.model.G_AB(input_tensor)
                blended_output += weight * styled_tensor

        # Save blended result
        output_path = self._save_output_image(
            blended_output,
            input_path,
            f"blend_{'_'.join(style_names)}",
            output_dir,
            "blended",
        )

        return output_path

    def get_model_info(self, style_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a loaded model.

        Args:
            style_name: Name of the style/model

        Returns:
            Model information dictionary or None if not loaded
        """
        if style_name in self.loaded_models:
            checkpoint = self.loaded_models[style_name]
            return {
                "style_name": style_name,
                "epoch": checkpoint.get("epoch", "unknown"),
                "losses": checkpoint.get("losses", {}),
                "best_loss": checkpoint.get("best_loss", "unknown"),
                "loaded_at": datetime.now().isoformat(),
            }
        return None
