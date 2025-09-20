# Apprentice Agent

The Apprentice Agent is a generative AI agent specialized in artistic style transfer using CycleGAN architecture. It learns and replicates user artistic styles to generate novel artwork from various inputs including satellite imagery.

## Overview

The Apprentice Agent implements a complete CycleGAN-based style transfer system with the following key features:

- **CycleGAN Architecture**: Unpaired image-to-image translation for artistic style transfer
- **Multi-Modal Input Processing**: Support for satellite imagery, standard images, and geospatial data
- **A2A Protocol Integration**: Collaborative artistic creation with other agents
- **Training Pipeline**: Complete training infrastructure with monitoring and checkpointing
- **Style Library**: Management of multiple learned artistic styles
- **Inference Engine**: High-performance style transfer application

## Architecture

### Core Components

1. **CycleGAN Model** (`cyclegan_model.py`)
   - Generator networks (G_AB, G_BA) for style transfer
   - Discriminator networks (D_A, D_B) for adversarial training
   - Cycle consistency and identity loss functions
   - Training loop with optimizers and learning rate scheduling

2. **Data Pipeline** (`data_loader.py`)
   - Unpaired image dataset handling
   - Satellite image preprocessing
   - Data augmentation and transformation
   - Batch processing utilities

3. **Training Pipeline** (`training_pipeline.py`)
   - Complete training orchestration
   - Checkpoint management
   - Metrics tracking and logging
   - Early stopping and validation

4. **Style Transfer Engine** (`style_transfer.py`)
   - Model loading and inference
   - Batch processing capabilities
   - Style blending and composition
   - Output postprocessing

5. **Input Handlers** (`input_handlers.py`)
   - Satellite image processing
   - Geospatial data integration
   - Multi-modal input support
   - Format conversion utilities

6. **Configuration Management** (`config.py`)
   - Hierarchical configuration system
   - Hyperparameter optimization
   - Experiment tracking
   - Configuration validation

7. **Apprentice Agent** (`apprentice_agent.py`)
   - Main agent class inheriting from BaseSpecialistAgent
   - LangChain integration for tool usage
   - A2A protocol communication
   - Autonomous operation capabilities

## Installation

The Apprentice Agent is part of the Terra Constellata system. Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 2.6.0
- TorchVision
- PIL (Pillow)
- NumPy
- LangChain
- A2A Protocol client

## Quick Start

### 1. Basic Usage

```python
from terra_constellata.agents.apprentice import ApprenticeAgent

# Initialize the agent
agent = ApprenticeAgent(
    llm=your_llm_instance,  # LangChain LLM
    model_dir="./models/apprentice",
    data_dir="./data/apprentice"
)

# Perform style transfer
result = agent.perform_style_transfer("path/to/input/image.jpg")
print(f"Style transfer completed: {result}")
```

### 2. Training a Model

```python
from terra_constellata.agents.apprentice import train

# Train from command line
python -m terra_constellata.agents.apprentice.train \
    --data-dir ./data/training_data \
    --domain-a satellite_images \
    --domain-b artwork \
    --epochs 200 \
    --batch-size 1 \
    --model-dir ./models/my_style
```

### 3. Style Transfer Inference

```python
from terra_constellata.agents.apprentice import infer

# Apply trained style
python -m terra_constellata.agents.apprentice.infer \
    --input ./input_images/ \
    --output-dir ./outputs/ \
    --model-name my_trained_style \
    --direction A_to_B
```

## Configuration

The Apprentice Agent uses a hierarchical configuration system. Create a configuration file:

```python
from terra_constellata.agents.apprentice.config import ApprenticeConfig

# Create custom configuration
config = ApprenticeConfig()
config.cyclegan.num_epochs = 150
config.cyclegan.learning_rate = 0.0001
config.data.image_size = 512

# Save configuration
from terra_constellata.agents.apprentice.config import save_config
save_config(config, "my_config")
```

## Data Preparation

### Training Data Structure

```
data/
├── domain_a/          # Source domain (e.g., satellite images)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── domain_b/          # Target domain (e.g., artwork)
    ├── artwork1.jpg
    ├── artwork2.jpg
    └── ...
```

### Supported Input Formats

- **Images**: JPEG, PNG, BMP, TIFF
- **Satellite Imagery**: GeoTIFF, standard formats
- **Geospatial Data**: Integration with PostGIS and CKG

## Training

### Basic Training

```python
from terra_constellata.agents.apprentice import CycleGAN, TrainingPipeline, TrainingConfig
from terra_constellata.agents.apprentice.data_loader import create_unpaired_data_loaders

# Create data loaders
train_loader, val_loader = create_unpaired_data_loaders(
    data_dir="./data",
    domain_a_dir="satellite_images",
    domain_b_dir="artwork",
    image_size=256,
    batch_size=1
)

# Initialize model and training
model = CycleGAN(device=torch.device("cuda"))
pipeline = TrainingPipeline(model, Path("./models"))

config = TrainingConfig(
    num_epochs=200,
    batch_size=1,
    learning_rate=0.0002
)

# Train the model
results = pipeline.train(train_loader, config)
```

### Training with Satellite Imagery

```python
from terra_constellata.agents.apprentice.data_loader import create_satellite_art_data_loaders

# Create satellite-specific data loaders
train_loader, val_loader = create_satellite_art_data_loaders(
    satellite_dir="./data/satellite",
    artwork_dir="./data/artwork",
    image_size=256,
    batch_size=1,
    satellite_preprocessing=True
)
```

## Inference

### Single Image Style Transfer

```python
from terra_constellata.agents.apprentice import StyleTransfer, CycleGAN

# Initialize style transfer engine
model = CycleGAN(device=torch.device("cuda"))
style_transfer = StyleTransfer(model, Path("./models"))

# Load trained model
style_transfer.load_model("my_trained_style")

# Apply style transfer
output_path = style_transfer.apply_style(
    input_path="input.jpg",
    style_name="my_trained_style",
    output_dir="./outputs"
)
```

### Batch Processing

```python
# Process multiple images
input_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
output_paths = style_transfer.batch_style_transfer(
    input_paths=input_paths,
    style_name="my_style",
    output_dir="./batch_outputs",
    batch_size=4
)
```

### Satellite to Artwork

```python
# Specialized satellite processing
artwork_path = style_transfer.satellite_to_artwork(
    satellite_path="satellite_image.tif",
    style_name="impressionist",
    output_dir="./satellite_art"
)
```

## A2A Protocol Integration

### Collaborative Style Transfer

```python
# Request inspiration from other agents
inspiration = await agent.request_artistic_inspiration(
    theme="abstract_expressionism",
    medium="visual"
)

# Share generated artwork
await agent.share_artwork(
    artwork_path="generated_artwork.jpg",
    description="AI-generated abstract piece",
    style_used="expressionist",
    target_agent="art_critic_agent"
)

# Request collaboration
collaboration = await agent.request_style_collaboration(
    input_description="urban satellite imagery",
    collaboration_type="style_fusion",
    target_agent="creative_agent"
)
```

### Creative Feedback

```python
# Provide feedback on artwork
await agent.provide_creative_feedback(
    original_artwork_id="artwork_123",
    feedback_content="Excellent use of color and composition",
    rating=5,
    target_agent="artist_agent",
    suggestions=["Try adding more contrast", "Consider warmer tones"]
)
```

## Advanced Features

### Style Blending

```python
# Blend multiple artistic styles
blended_path = style_transfer.blend_styles(
    input_path="input.jpg",
    style_names=["impressionist", "cubist", "abstract"],
    weights=[0.5, 0.3, 0.2],
    output_dir="./blended"
)
```

### Multi-Modal Input Processing

```python
from terra_constellata.agents.apprentice.input_handlers import InputPipeline

# Initialize input pipeline
pipeline = InputPipeline(target_size=256)

# Process various input types
results = pipeline.process([
    "satellite_image.tif",      # Satellite imagery
    "photo.jpg",               # Standard photo
    {"type": "elevation", "path": "dem.tif"}  # Elevation data
])
```

### Configuration Optimization

```python
from terra_constellata.agents.apprentice.config import ConfigManager

# Set up hyperparameter search
config_manager = ConfigManager()
base_config = config_manager.load_config("base")

# Define parameter ranges
param_ranges = {
    "cyclegan.learning_rate": [0.0001, 0.0002, 0.0005],
    "cyclegan.lambda_cycle": [5.0, 10.0, 15.0],
    "cyclegan.num_epochs": [100, 150, 200]
}

# Generate experiment configurations
experiment_configs = config_manager.generate_hyperparameter_grid(
    base_config, param_ranges
)
```

## API Reference

### ApprenticeAgent

Main agent class for artistic style transfer.

**Methods:**
- `perform_style_transfer(input_description)`: Apply style transfer
- `train_model(training_config)`: Train new models
- `request_artistic_inspiration(theme, medium)`: Request inspiration
- `share_artwork(path, description, style, target)`: Share artwork
- `provide_creative_feedback(id, content, rating, target)`: Give feedback

### CycleGAN

Core CycleGAN model implementation.

**Methods:**
- `forward(real_A, real_B)`: Forward pass
- `optimize_parameters(real_A, real_B)`: Training step
- `save_model(path)`: Save model checkpoint
- `load_model(path)`: Load model checkpoint

### StyleTransfer

Inference engine for style transfer.

**Methods:**
- `apply_style(input_path, style_name)`: Single image transfer
- `batch_style_transfer(paths, style_name)`: Batch processing
- `satellite_to_artwork(satellite_path, style_name)`: Satellite processing
- `blend_styles(input_path, style_names, weights)`: Style blending

## Examples

See `example_usage.py` for comprehensive usage examples including:

- Basic agent initialization and usage
- Training pipeline setup
- Style transfer inference
- Configuration management
- A2A protocol integration
- Complete workflow demonstrations

## Testing

Run the test suite:

```bash
python -m pytest terra_constellata/agents/apprentice/test_apprentice.py -v
```

Or run specific test categories:

```bash
# Test CycleGAN components
python -m unittest terra_constellata.agents.apprentice.test_apprentice.TestCycleGAN

# Test data loading
python -m unittest terra_constellata.agents.apprentice.test_apprentice.TestDataLoader

# Test style transfer
python -m unittest terra_constellata.agents.apprentice.test_apprentice.TestStyleTransfer
```

## Performance Optimization

### GPU Usage

The Apprentice Agent automatically detects and uses GPU when available:

```python
# Force CPU usage
config.cyclegan.device = "cpu"

# Force GPU usage
config.cyclegan.device = "cuda"
```

### Memory Optimization

For large images or batch processing:

```python
# Reduce batch size for memory constraints
config.cyclegan.batch_size = 1

# Use gradient checkpointing (if implemented)
config.training.gradient_checkpointing = True
```

### Multi-GPU Training

```python
# DataParallel for multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller image sizes
   - Enable gradient checkpointing

2. **Training Not Converging**
   - Adjust learning rates
   - Check data quality and pairing
   - Increase training epochs
   - Verify loss function implementation

3. **Poor Style Transfer Quality**
   - Ensure sufficient training data
   - Check domain alignment
   - Adjust lambda_cycle and lambda_identity weights
   - Use better data augmentation

4. **A2A Connection Issues**
   - Verify A2A server is running
   - Check network connectivity
   - Validate agent registration

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When contributing to the Apprentice Agent:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure compatibility with existing A2A protocol
5. Test with various input formats and edge cases

## License

This component is part of the Terra Constellata system. See the main project license for details.

## References

- [CycleGAN Paper](https://arxiv.org/abs/1703.10593)
- [Unpaired Image-to-Image Translation](https://junyanz.github.io/CycleGAN/)
- [LangChain Documentation](https://python.langchain.com/)
- [Terra Constellata A2A Protocol](https://github.com/terra-constellata/a2a-protocol)