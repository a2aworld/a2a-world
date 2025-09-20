"""
Apprentice Agent Package

This package contains the Apprentice Agent, a generative AI agent specialized in
artistic style transfer using CycleGAN architecture. The agent learns and replicates
user artistic styles to generate novel artwork from various inputs including
satellite imagery.
"""

from .apprentice_agent import ApprenticeAgent
from .cyclegan_model import CycleGAN
from .data_loader import ImageDataset, DataLoader
from .training_pipeline import TrainingPipeline
from .style_transfer import StyleTransfer

__all__ = [
    "ApprenticeAgent",
    "CycleGAN",
    "ImageDataset",
    "DataLoader",
    "TrainingPipeline",
    "StyleTransfer",
]
