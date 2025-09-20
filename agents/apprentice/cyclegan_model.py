"""
CycleGAN Model Implementation

This module implements the CycleGAN architecture for unpaired image-to-image translation,
used for artistic style transfer in the Apprentice Agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block for the generator network."""

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """Generator network for CycleGAN."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        n_residual_blocks: int = 9,
    ):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """Discriminator network for CycleGAN."""

    def __init__(self, input_channels: int = 3):
        super(Discriminator, self).__init__()

        def discriminator_block(
            in_filters: int, out_filters: int, normalize: bool = True
        ):
            """Returns downsampling layers of each discriminator block."""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CycleGAN(nn.Module):
    """
    Complete CycleGAN model for unpaired image-to-image translation.

    This implementation includes:
    - Two generators (G_AB, G_BA)
    - Two discriminators (D_A, D_B)
    - Cycle consistency loss
    - Identity loss
    - Adversarial loss
    """

    def __init__(
        self,
        device: torch.device,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
    ):
        super(CycleGAN, self).__init__()

        self.device = device
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        # Initialize networks
        self.G_AB = Generator().to(device)  # Generator A -> B
        self.G_BA = Generator().to(device)  # Generator B -> A
        self.D_A = Discriminator().to(device)  # Discriminator A
        self.D_B = Discriminator().to(device)  # Discriminator B

        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # Initialize optimizers
        self.optimizer_G = Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=0.0002,
            betas=(0.5, 0.999),
        )
        self.optimizer_D_A = Adam(self.D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D_B = Adam(self.D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Learning rate schedulers
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=lambda epoch: 1 - max(0, epoch - 100) / 100
        )
        self.scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=lambda epoch: 1 - max(0, epoch - 100) / 100
        )
        self.scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=lambda epoch: 1 - max(0, epoch - 100) / 100
        )

        logger.info("CycleGAN model initialized")

    def forward(
        self, real_A: torch.Tensor, real_B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CycleGAN.

        Args:
            real_A: Images from domain A
            real_B: Images from domain B

        Returns:
            Tuple of (fake_B, fake_A) generated images
        """
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        return fake_B, fake_A

    def backward_D_basic(
        self, netD: nn.Module, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Calculate GAN loss for the discriminator."""
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))

        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        # Total loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self, real_A: torch.Tensor, fake_A: torch.Tensor) -> torch.Tensor:
        """Calculate discriminator A loss."""
        return self.backward_D_basic(self.D_A, real_A, fake_A)

    def backward_D_B(self, real_B: torch.Tensor, fake_B: torch.Tensor) -> torch.Tensor:
        """Calculate discriminator B loss."""
        return self.backward_D_basic(self.D_B, real_B, fake_B)

    def backward_G(
        self, real_A: torch.Tensor, real_B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate generator losses including GAN, cycle, and identity losses.

        Returns:
            Tuple of (loss_G, loss_G_GAN, loss_cycle) for monitoring
        """
        # GAN loss
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)

        pred_fake_B = self.D_B(fake_B)
        pred_fake_A = self.D_A(fake_A)

        loss_G_AB = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))
        loss_G_BA = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))
        loss_G_GAN = loss_G_AB + loss_G_BA

        # Cycle loss
        recovered_A = self.G_BA(fake_B)
        recovered_B = self.G_AB(fake_A)

        loss_cycle_A = self.criterion_cycle(recovered_A, real_A)
        loss_cycle_B = self.criterion_cycle(recovered_B, real_B)
        loss_cycle = loss_cycle_A + loss_cycle_B

        # Identity loss
        same_A = self.G_BA(real_A)
        same_B = self.G_AB(real_B)

        loss_identity_A = self.criterion_identity(same_A, real_A)
        loss_identity_B = self.criterion_identity(same_B, real_B)
        loss_identity = loss_identity_A + loss_identity_B

        # Total loss
        loss_G = (
            loss_G_GAN
            + self.lambda_cycle * loss_cycle
            + self.lambda_identity * loss_identity
        )
        loss_G.backward()

        return loss_G, loss_G_GAN, loss_cycle

    def optimize_parameters(
        self, real_A: torch.Tensor, real_B: torch.Tensor
    ) -> Dict[str, float]:
        """
        Optimize model parameters for one training step.

        Args:
            real_A: Images from domain A
            real_B: Images from domain B

        Returns:
            Dictionary with loss values for monitoring
        """
        # Generate fake images
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)

        # Optimize discriminators
        self.set_requires_grad([self.D_A, self.D_B], True)
        self.optimizer_D_A.zero_grad()
        self.optimizer_D_B.zero_grad()

        loss_D_A = self.backward_D_A(real_A, fake_A)
        loss_D_B = self.backward_D_B(real_B, fake_B)

        self.optimizer_D_A.step()
        self.optimizer_D_B.step()

        # Optimize generators
        self.set_requires_grad([self.D_A, self.D_B], False)
        self.optimizer_G.zero_grad()

        loss_G, loss_G_GAN, loss_cycle = self.backward_G(real_A, real_B)
        self.optimizer_G.step()

        # Update learning rates
        self.scheduler_G.step()
        self.scheduler_D_A.step()
        self.scheduler_D_B.step()

        return {
            "loss_G": loss_G.item(),
            "loss_G_GAN": loss_G_GAN.item(),
            "loss_cycle": loss_cycle.item(),
            "loss_D_A": loss_D_A.item(),
            "loss_D_B": loss_D_B.item(),
        }

    def set_requires_grad(self, nets: list, requires_grad: bool = False):
        """Set requires_grad for networks."""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_model(self, path: str):
        """Save model checkpoints."""
        torch.save(
            {
                "G_AB_state_dict": self.G_AB.state_dict(),
                "G_BA_state_dict": self.G_BA.state_dict(),
                "D_A_state_dict": self.D_A.state_dict(),
                "D_B_state_dict": self.D_B.state_dict(),
                "optimizer_G_state_dict": self.optimizer_G.state_dict(),
                "optimizer_D_A_state_dict": self.optimizer_D_A.state_dict(),
                "optimizer_D_B_state_dict": self.optimizer_D_B.state_dict(),
                "scheduler_G_state_dict": self.scheduler_G.state_dict(),
                "scheduler_D_A_state_dict": self.scheduler_D_A.state_dict(),
                "scheduler_D_B_state_dict": self.scheduler_D_B.state_dict(),
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.G_AB.load_state_dict(checkpoint["G_AB_state_dict"])
        self.G_BA.load_state_dict(checkpoint["G_BA_state_dict"])
        self.D_A.load_state_dict(checkpoint["D_A_state_dict"])
        self.D_B.load_state_dict(checkpoint["D_B_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        self.optimizer_D_A.load_state_dict(checkpoint["optimizer_D_A_state_dict"])
        self.optimizer_D_B.load_state_dict(checkpoint["optimizer_D_B_state_dict"])
        self.scheduler_G.load_state_dict(checkpoint["scheduler_G_state_dict"])
        self.scheduler_D_A.load_state_dict(checkpoint["scheduler_D_A_state_dict"])
        self.scheduler_D_B.load_state_dict(checkpoint["scheduler_D_B_state_dict"])
        logger.info(f"Model loaded from {path}")

    def eval(self):
        """Set model to evaluation mode."""
        self.G_AB.eval()
        self.G_BA.eval()
        self.D_A.eval()
        self.D_B.eval()

    def train(self):
        """Set model to training mode."""
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()
