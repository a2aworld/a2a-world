"""
Tests for Apprentice Agent

This module contains comprehensive tests for the Apprentice Agent components,
including unit tests, integration tests, and validation utilities.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

# Add the project root to Python path for testing
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .apprentice_agent import ApprenticeAgent
from .cyclegan_model import CycleGAN, Generator, Discriminator
from .data_loader import UnpairedImageDataset, ImageTransform
from .training_pipeline import TrainingPipeline, TrainingConfig
from .style_transfer import StyleTransfer
from .config import ApprenticeConfig, ConfigManager
from .input_handlers import SatelliteImageProcessor, MultiModalInputHandler


class TestCycleGAN(unittest.TestCase):
    """Test CycleGAN model components."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")  # Use CPU for testing
        self.batch_size = 2
        self.image_size = 64  # Small size for testing

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")  # Use CPU for testing
        self.batch_size = 2
        self.image_size = 64  # Small size for testing

    def test_generator_creation(self):
        """Test generator network creation."""
        generator = Generator(image_size=self.image_size)

        # Test forward pass
        input_tensor = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        output_tensor = generator(input_tensor)

        self.assertEqual(output_tensor.shape, input_tensor.shape)
        self.assertTrue(torch.is_tensor(output_tensor))

    def test_discriminator_creation(self):
        """Test discriminator network creation."""
        discriminator = Discriminator()

        # Test forward pass
        input_tensor = torch.randn(
            self.batch_size, 3, 256, 256
        )  # Standard discriminator input
        output_tensor = discriminator(input_tensor)

        self.assertEqual(output_tensor.shape, (self.batch_size, 1, 30, 30))
        self.assertTrue(torch.is_tensor(output_tensor))

    def test_cyclegan_initialization(self):
        """Test CycleGAN model initialization."""
        model = CycleGAN(device=self.device)

        # Check that networks are created
        self.assertIsNotNone(model.G_AB)
        self.assertIsNotNone(model.G_BA)
        self.assertIsNotNone(model.D_A)
        self.assertIsNotNone(model.D_B)

        # Check that optimizers are created
        self.assertIsNotNone(model.optimizer_G)
        self.assertIsNotNone(model.optimizer_D_A)
        self.assertIsNotNone(model.optimizer_D_B)

    @patch("torch.save")
    def test_model_save(self, mock_save):
        """Test model saving."""
        model = CycleGAN(device=self.device)

        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_model(f"{temp_dir}/test_model.pth")

        mock_save.assert_called_once()

    @patch("torch.load")
    def test_model_load(self, mock_load):
        """Test model loading."""
        mock_load.return_value = {
            "G_AB_state_dict": {},
            "G_BA_state_dict": {},
            "D_A_state_dict": {},
            "D_B_state_dict": {},
            "optimizer_G_state_dict": {},
            "optimizer_D_A_state_dict": {},
            "optimizer_D_B_state_dict": {},
            "scheduler_G_state_dict": {},
            "scheduler_D_A_state_dict": {},
            "scheduler_D_B_state_dict": {},
        }

        model = CycleGAN(device=self.device)

        with tempfile.TemporaryDirectory() as temp_dir:
            model.load_model(f"{temp_dir}/test_model.pth")

        mock_load.assert_called_once()


class TestDataLoader(unittest.TestCase):
    """Test data loading components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.domain_a_dir = Path(self.temp_dir) / "domain_a"
        self.domain_b_dir = Path(self.temp_dir) / "domain_b"

        # Create directories
        self.domain_a_dir.mkdir()
        self.domain_b_dir.mkdir()

        # Create dummy images
        self._create_dummy_image(self.domain_a_dir / "img1.jpg")
        self._create_dummy_image(self.domain_a_dir / "img2.jpg")
        self._create_dummy_image(self.domain_b_dir / "img1.jpg")
        self._create_dummy_image(self.domain_b_dir / "img2.jpg")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_dummy_image(self, path: Path):
        """Create a dummy image file."""
        # Create a small dummy image
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        # For simplicity, just create an empty file
        path.touch()

    def test_image_transform(self):
        """Test image transformation."""
        transform = ImageTransform(image_size=64)

        # Create dummy PIL image
        dummy_image = Mock()
        dummy_image.convert.return_value = dummy_image

        # Mock the transform to return a tensor
        with patch("torchvision.transforms.functional.to_tensor") as mock_to_tensor:
            mock_to_tensor.return_value = torch.randn(3, 64, 64)

            result = transform(dummy_image, is_train=True)
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.shape, (3, 64, 64))

    def test_unpaired_dataset_creation(self):
        """Test unpaired dataset creation."""
        dataset = UnpairedImageDataset(
            root_dir=self.temp_dir,
            domain_a_dir="domain_a",
            domain_b_dir="domain_b",
            image_size=64,
            is_train=True,
        )

        self.assertEqual(len(dataset), 2)  # Should have 2 samples (min of both domains)

    def test_satellite_processor(self):
        """Test satellite image processor."""
        processor = SatelliteImageProcessor(target_size=64)

        # Test with dummy image path
        dummy_path = str(self.domain_a_dir / "img1.jpg")

        # Mock the image loading and processing
        with patch("PIL.Image.open") as mock_open:
            mock_image = Mock()
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image

            with patch("torchvision.transforms.functional.to_tensor") as mock_to_tensor:
                mock_to_tensor.return_value = torch.randn(3, 64, 64)

                tensor, metadata = processor.process_satellite_image(dummy_path)

                self.assertIsInstance(tensor, torch.Tensor)
                self.assertIsInstance(metadata, dict)


class TestStyleTransfer(unittest.TestCase):
    """Test style transfer components."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.temp_dir) / "models"
        self.model_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_style_transfer_initialization(self):
        """Test style transfer engine initialization."""
        model = CycleGAN(device=self.device)
        style_transfer = StyleTransfer(model, self.model_dir)

        self.assertIsNotNone(style_transfer.model)
        self.assertEqual(style_transfer.model_dir, self.model_dir)

    @patch("PIL.Image.open")
    @patch("torchvision.transforms.functional.to_tensor")
    def test_apply_style(self, mock_to_tensor, mock_open):
        """Test style application."""
        # Setup mocks
        mock_image = Mock()
        mock_image.convert.return_value = mock_image
        mock_open.return_value = mock_image
        mock_to_tensor.return_value = torch.randn(3, 256, 256)

        model = CycleGAN(device=self.device)
        style_transfer = StyleTransfer(model, self.model_dir)

        # Mock model loading
        style_transfer.load_model = Mock(return_value=True)

        # Test style application
        with patch("torchvision.transforms.functional.to_pil_image") as mock_to_pil:
            mock_to_pil.return_value = mock_image

            with patch.object(mock_image, "save") as mock_save:
                result_path = style_transfer.apply_style(
                    input_path="/fake/path/image.jpg", style_name="test_style"
                )

                self.assertIsInstance(result_path, str)
                self.assertIn("test_style", result_path)


class TestConfiguration(unittest.TestCase):
    """Test configuration management."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_config_creation(self):
        """Test configuration creation."""
        config = ApprenticeConfig()

        self.assertIsNotNone(config.cyclegan)
        self.assertIsNotNone(config.data)
        self.assertIsNotNone(config.training)
        self.assertIsNotNone(config.inference)
        self.assertIsNotNone(config.a2a)

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = ApprenticeConfig()
        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertIn("cyclegan", config_dict)
        self.assertIn("data", config_dict)

        # Test deserialization
        new_config = ApprenticeConfig.from_dict(config_dict)
        self.assertEqual(new_config.cyclegan.image_size, config.cyclegan.image_size)

    def test_config_manager(self):
        """Test configuration manager."""
        manager = ConfigManager(str(self.config_dir))

        # Test saving and loading
        config = ApprenticeConfig()
        config.cyclegan.num_epochs = 50

        manager.save_config(config, "test_config")
        loaded_config = manager.load_config("test_config")

        self.assertEqual(loaded_config.cyclegan.num_epochs, 50)


class TestApprenticeAgent(unittest.TestCase):
    """Test Apprentice Agent integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.temp_dir) / "models"
        self.data_dir = Path(self.temp_dir) / "data"
        self.model_dir.mkdir()
        self.data_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("terra_constellata.agents.apprentice.apprentice_agent.A2AClient")
    def test_agent_initialization(self, mock_a2a_client):
        """Test agent initialization."""
        mock_a2a_client.return_value = Mock()

        agent = ApprenticeAgent(
            llm=None,  # No LLM for testing
            model_dir=str(self.model_dir),
            data_dir=str(self.data_dir),
        )

        self.assertEqual(agent.name, "Apprentice_Agent")
        self.assertIsNotNone(agent.cyclegan)
        self.assertIsNotNone(agent.training_pipeline)
        self.assertIsNotNone(agent.style_transfer)

    def test_agent_tools(self):
        """Test agent tools."""
        agent = ApprenticeAgent(
            llm=None, model_dir=str(self.model_dir), data_dir=str(self.data_dir)
        )

        # Check that tools are created
        self.assertEqual(len(agent.tools), 2)
        tool_names = [tool.name for tool in agent.tools]
        self.assertIn("style_transfer", tool_names)
        self.assertIn("train_model", tool_names)


class TestInputHandlers(unittest.TestCase):
    """Test input handling components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_multi_modal_handler(self):
        """Test multi-modal input handler."""
        handler = MultiModalInputHandler(target_size=64)

        self.assertIsNotNone(handler.satellite_processor)
        self.assertIsNotNone(handler.geospatial_handler)

    def test_satellite_processor_creation(self):
        """Test satellite processor creation."""
        processor = SatelliteImageProcessor(target_size=64)

        self.assertEqual(processor.target_size, 64)
        self.assertIsNotNone(processor.preprocess_transforms)


class IntegrationTest(unittest.TestCase):
    """Integration tests for Apprentice Agent components."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.temp_dir) / "models"
        self.data_dir = Path(self.temp_dir) / "data"
        self.model_dir.mkdir()
        self.data_dir.mkdir()

    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_full_pipeline_integration(self):
        """Test integration of full pipeline components."""
        # This would test the integration of all components
        # For now, just test that components can be instantiated together

        device = torch.device("cpu")

        # Initialize components
        model = CycleGAN(device=device)
        training_pipeline = TrainingPipeline(model, self.model_dir)
        style_transfer = StyleTransfer(model, self.model_dir)

        # Verify components are properly connected
        self.assertIsNotNone(training_pipeline.model)
        self.assertIsNotNone(style_transfer.model)
        self.assertEqual(training_pipeline.model_dir, self.model_dir)
        self.assertEqual(style_transfer.model_dir, self.model_dir)


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
