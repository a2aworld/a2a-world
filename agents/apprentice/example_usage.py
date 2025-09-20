#!/usr/bin/env python3
"""
Example Usage of Apprentice Agent

This script demonstrates how to use the Apprentice Agent for artistic style transfer,
including training, inference, and A2A protocol integration.
"""

import asyncio
import logging
import sys
import torch
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .apprentice_agent import ApprenticeAgent
from .cyclegan_model import CycleGAN
from .data_loader import create_unpaired_data_loaders
from .training_pipeline import TrainingPipeline, TrainingConfig
from .style_transfer import StyleTransfer
from .config import ApprenticeConfig, ConfigManager
from .input_handlers import InputPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Example of basic Apprentice Agent usage."""
    logger.info("=== Basic Apprentice Agent Usage Example ===")

    # Initialize the agent
    agent = ApprenticeAgent(
        llm=None,  # No LLM needed for this example
        model_dir="./models/apprentice_example",
        data_dir="./data/apprentice_example",
    )

    logger.info("Apprentice Agent initialized")
    logger.info(f"Model directory: {agent.model_dir}")
    logger.info(f"Data directory: {agent.data_dir}")

    # Example style transfer (would work with trained model)
    logger.info("Example style transfer call (requires trained model):")
    logger.info("result = agent.perform_style_transfer('path/to/input/image.jpg')")

    return agent


async def example_training_pipeline():
    """Example of training a CycleGAN model."""
    logger.info("=== Training Pipeline Example ===")

    # Create sample data directories (in real usage, these would contain actual images)
    data_dir = Path("./data/apprentice_example")
    domain_a_dir = data_dir / "satellite_images"
    domain_b_dir = data_dir / "artwork"

    domain_a_dir.mkdir(parents=True, exist_ok=True)
    domain_b_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Created sample data directories:")
    logger.info(f"Domain A (satellite): {domain_a_dir}")
    logger.info(f"Domain B (artwork): {domain_b_dir}")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CycleGAN(device=device)

    # Create data loaders (would load actual data in real usage)
    try:
        train_loader, test_loader = create_unpaired_data_loaders(
            data_dir=str(data_dir),
            domain_a_dir="satellite_images",
            domain_b_dir="artwork",
            image_size=256,
            batch_size=1,
            max_samples=10,  # Small sample for demo
        )
        logger.info("Data loaders created successfully")
    except Exception as e:
        logger.warning(f"Data loader creation failed (expected without real data): {e}")
        return

    # Training configuration
    config = TrainingConfig(
        num_epochs=5,  # Very short for demo
        batch_size=1,
        learning_rate=0.0002,
        sample_interval=10,
        checkpoint_interval=2,
        log_interval=1,
    )

    # Initialize training pipeline
    model_dir = Path("./models/apprentice_training_example")
    pipeline = TrainingPipeline(model, model_dir)

    logger.info("Training pipeline initialized")
    logger.info("In real usage, you would call:")
    logger.info("results = pipeline.train(train_loader, config)")

    return pipeline


async def example_style_transfer():
    """Example of style transfer inference."""
    logger.info("=== Style Transfer Example ===")

    # Initialize model and style transfer engine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CycleGAN(device=device)
    model_dir = Path("./models/apprentice_example")

    style_transfer = StyleTransfer(model, model_dir)

    logger.info("Style transfer engine initialized")
    logger.info("Available methods:")
    logger.info("- apply_style(input_path, style_name)")
    logger.info("- batch_style_transfer(input_paths, style_name)")
    logger.info("- satellite_to_artwork(satellite_path, style_name)")
    logger.info("- blend_styles(input_path, style_names, weights)")

    # Example calls (would work with real trained models and data)
    logger.info("\nExample calls:")
    logger.info("output_path = style_transfer.apply_style('input.jpg', 'my_style')")
    logger.info(
        "artwork_path = style_transfer.satellite_to_artwork('satellite.jpg', 'impressionist')"
    )

    return style_transfer


async def example_configuration():
    """Example of configuration management."""
    logger.info("=== Configuration Management Example ===")

    # Initialize config manager
    config_manager = ConfigManager("./config/apprentice_examples")

    # Get default configuration
    default_config = config_manager.default_config
    logger.info("Default configuration loaded")
    logger.info(f"Image size: {default_config.cyclegan.image_size}")
    logger.info(f"Learning rate: {default_config.cyclegan.learning_rate}")
    logger.info(f"Batch size: {default_config.cyclegan.batch_size}")

    # Create custom configuration
    custom_config = ApprenticeConfig()
    custom_config.cyclegan.num_epochs = 100
    custom_config.cyclegan.learning_rate = 0.0001
    custom_config.data.image_size = 512

    # Save custom configuration
    config_manager.save_config(custom_config, "custom_example")
    logger.info("Custom configuration saved as 'custom_example'")

    # Load configuration
    loaded_config = config_manager.load_config("custom_example")
    logger.info("Configuration loaded successfully")
    logger.info(f"Loaded epochs: {loaded_config.cyclegan.num_epochs}")

    return config_manager


async def example_input_processing():
    """Example of input processing pipeline."""
    logger.info("=== Input Processing Example ===")

    # Initialize input pipeline
    pipeline = InputPipeline(target_size=256)

    logger.info("Input pipeline initialized")
    logger.info("Supported input types:")
    logger.info("- Standard images (jpg, png, etc.)")
    logger.info("- Satellite imagery")
    logger.info("- Geospatial data")
    logger.info("- Multi-modal composites")

    # Example processing (would work with real data)
    logger.info("\nExample processing calls:")
    logger.info("results = pipeline.process(['image1.jpg', 'satellite.tif'])")
    logger.info("composite = pipeline.process({'type': 'satellite', 'path': 'data/'})")

    return pipeline


async def example_a2a_integration():
    """Example of A2A protocol integration."""
    logger.info("=== A2A Protocol Integration Example ===")

    # Initialize agent with A2A support
    agent = ApprenticeAgent(
        llm=None, a2a_server_url="http://localhost:8080"  # No LLM for this example
    )

    logger.info("Agent initialized with A2A protocol support")
    logger.info("Available A2A methods:")
    logger.info("- request_artistic_inspiration(theme, medium)")
    logger.info("- share_artwork(artwork_path, description, style)")
    logger.info("- request_style_collaboration(description, target_agent)")
    logger.info("- provide_creative_feedback(artwork_id, feedback, rating)")
    logger.info("- request_narrative_for_artwork(description, target_agent)")

    # Example A2A calls (would work with running A2A server)
    logger.info("\nExample A2A calls:")
    logger.info(
        "response = await agent.request_artistic_inspiration('abstract', 'visual')"
    )
    logger.info(
        "await agent.share_artwork('artwork.jpg', 'My creation', 'surrealist', 'agent2')"
    )

    return agent


async def example_full_workflow():
    """Example of a complete workflow."""
    logger.info("=== Complete Workflow Example ===")

    # This would demonstrate a full workflow from data preparation
    # to training to inference to A2A collaboration

    logger.info("Complete workflow steps:")
    logger.info("1. Prepare training data (satellite images + artwork)")
    logger.info("2. Configure model and training parameters")
    logger.info("3. Train CycleGAN model")
    logger.info("4. Evaluate model performance")
    logger.info("5. Apply style transfer to new images")
    logger.info("6. Share results via A2A protocol")
    logger.info("7. Collaborate with other agents")

    logger.info("\nExample code structure:")
    logger.info(
        """
# Setup
config = ApprenticeConfig()
config.cyclegan.num_epochs = 200
config.data.data_root = "./my_data"

# Training
model = CycleGAN(device=torch.device("cuda"))
pipeline = TrainingPipeline(model, Path("./models"))
results = pipeline.train(train_loader, config)

# Inference
style_transfer = StyleTransfer(model, Path("./models"))
output_path = style_transfer.apply_style("input.jpg", "trained_style")

# A2A Collaboration
agent = ApprenticeAgent(llm=my_llm)
await agent.share_artwork(output_path, "Generated artwork", "my_style", "colleague_agent")
"""
    )


async def main():
    """Run all examples."""
    logger.info("Apprentice Agent Examples")
    logger.info("=" * 50)

    try:
        # Run examples
        await example_basic_usage()
        print()

        await example_training_pipeline()
        print()

        await example_style_transfer()
        print()

        await example_configuration()
        print()

        await example_input_processing()
        print()

        await example_a2a_integration()
        print()

        await example_full_workflow()
        print()

        logger.info("All examples completed successfully!")
        logger.info("\nTo run actual training/inference, you would need:")
        logger.info("1. Real training data (satellite images + artwork)")
        logger.info("2. Trained model checkpoints")
        logger.info("3. Running A2A protocol server (for collaboration)")

    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
