#!/usr/bin/env python3
"""
Inference Script for Apprentice Agent Style Transfer

This script provides command-line interface for applying trained CycleGAN models
to perform artistic style transfer on new images.
"""

import argparse
import logging
import sys
import torch
from pathlib import Path
from typing import List

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .style_transfer import StyleTransfer
from .cyclegan_model import CycleGAN
from .config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply artistic style transfer using trained CycleGAN models"
    )

    # Input arguments
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input image or directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save output images",
    )

    # Model arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/apprentice",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="best_model",
        help="Name of the model checkpoint to use",
    )
    parser.add_argument(
        "--config", type=str, default="default", help="Configuration name to use"
    )

    # Style transfer arguments
    parser.add_argument(
        "--direction",
        type=str,
        default="A_to_B",
        choices=["A_to_B", "B_to_A"],
        help="Translation direction",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing multiple images",
    )

    # Processing arguments
    parser.add_argument(
        "--recursive", action="store_true", help="Process directories recursively"
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.jpg,*.jpeg,*.png",
        help="File patterns to match (comma-separated)",
    )

    # Device arguments
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )

    return parser.parse_args()


def collect_input_files(
    input_path: str, recursive: bool = False, patterns: List[str] = None
) -> List[str]:
    """
    Collect input files from path.

    Args:
        input_path: Path to input file or directory
        recursive: Whether to search recursively
        patterns: File patterns to match

    Returns:
        List of input file paths
    """
    if patterns is None:
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]

    input_path = Path(input_path)

    if input_path.is_file():
        return [str(input_path)]

    if input_path.is_dir():
        files = []
        glob_pattern = "**/*" if recursive else "*"

        for pattern in patterns:
            if recursive:
                files.extend(input_path.glob(f"**/{pattern}"))
            else:
                files.extend(input_path.glob(pattern))

        return sorted(list(set(str(f) for f in files)))

    return []


def setup_inference(args):
    """
    Set up inference components.

    Args:
        args: Parsed command line arguments

    Returns:
        StyleTransfer instance
    """
    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    config.inference.model_name = args.model_name
    config.inference.direction = args.direction
    config.inference.batch_size = args.batch_size
    config.inference.output_dir = args.output_dir

    # Set device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize CycleGAN model
    model = CycleGAN(device=torch.device(device))

    # Initialize style transfer engine
    style_transfer = StyleTransfer(model, Path(args.model_dir))

    # Load the specified model
    if not style_transfer.load_model(args.model_name):
        logger.error(f"Failed to load model: {args.model_name}")
        sys.exit(1)

    return style_transfer, config


def main():
    """Main inference function."""
    args = parse_args()

    logger.info("Starting style transfer inference")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model: {args.model_name}")

    try:
        # Setup inference
        style_transfer, config = setup_inference(args)

        # Collect input files
        patterns = [p.strip() for p in args.file_pattern.split(",")]
        input_files = collect_input_files(args.input, args.recursive, patterns)

        if not input_files:
            logger.error(f"No input files found at: {args.input}")
            sys.exit(1)

        logger.info(f"Found {len(input_files)} input files")

        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Process files
        if len(input_files) == 1:
            # Single file processing
            output_path = style_transfer.apply_style(
                input_path=input_files[0],
                style_name=args.model_name,
                output_dir=args.output_dir,
                direction=args.direction,
            )
            logger.info(f"Style transfer completed: {output_path}")

        else:
            # Batch processing
            output_paths = style_transfer.batch_style_transfer(
                input_paths=input_files,
                style_name=args.model_name,
                output_dir=args.output_dir,
                direction=args.direction,
                batch_size=args.batch_size,
            )
            logger.info(
                f"Batch processing completed: {len(output_paths)} files processed"
            )

        logger.info("Inference completed successfully!")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
