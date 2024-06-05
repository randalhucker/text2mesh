from datetime import datetime
import os
from typing import Tuple, Optional, Dict, Any

import torch
from torch import nn, optim
import torch.nn.functional as F

from utils import FourierFeatureTransform, device


class ProgressiveEncoding(nn.Module):
    def __init__(self, mapping_size: int, T: int, d=3, apply=True):
        """Initialize the ProgressiveEncoding module.

        Args:
            mapping_size (int): Size of the Fourier feature mapping.
            T (int): Total number of iterations for progressive encoding.
            d (int, optional): Dimensionality of the input. Defaults to 3.
            apply (bool, optional): Whether to apply progressive encoding. Defaults to True.
        """
        super(ProgressiveEncoding, self).__init__()
        self._t = nn.Parameter(
            torch.tensor(0, dtype=torch.float32, device=device), requires_grad=False
        )
        self.n = mapping_size
        self.T = T
        self.d = d
        self._tau = 2 * self.n / self.T
        self.indices = torch.tensor([i for i in range(self.n)], device=device)
        self.apply = apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the ProgressiveEncoding module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with applied progressive encoding.
        """
        alpha = (
            ((self._t - self._tau * self.indices) / self._tau).clamp(0, 1).repeat(2)
        )  # No need to reduce d or to check cases

        if not self.apply:
            alpha = torch.ones_like(
                alpha, device=device
            )  # This layer means pure ffn without progress.

        alpha = torch.cat([torch.ones(self.d, device=device), alpha], dim=0)
        self._t += 1
        return x * alpha


class NeuralStyleField(nn.Module):
    # Same base then split into two separate modules
    def __init__(
        self,
        sigma: float,
        depth: int,
        width: int,
        encoding: str,
        colordepth=2,
        normdepth=2,
        normratio=0.1,
        clamp: str | None = None,
        normclamp: str | None = None,
        niter=6000,
        input_dim=3,
        progressive_encoding=True,
        exclude=0,
        text_encoding_dim=512
    ):
        """Initialize the NeuralStyleField module.

        Args:
            sigma (float): Standard deviation for Gaussian Fourier features.
            depth (int): Depth of the base MLP.
            width (int): Width of the MLP layers.
            encoding (str): Encoding type ('gaussian' or other).
            colordepth (int, optional): Depth of the color branch. Defaults to 2.
            normdepth (int, optional): Depth of the normal branch. Defaults to 2.
            normratio (float, optional): Scaling factor for normals. Defaults to 0.1.
            clamp (str, optional): Clamping method for colors ('tanh' or 'clamp'). Defaults to None.
            normclamp (str, optional): Clamping method for normals ('tanh' or 'clamp'). Defaults to None.
            niter (int, optional): Number of iterations for progressive encoding. Defaults to 6000.
            input_dim (int, optional): Dimensionality of the input. Defaults to 3.
            progressive_encoding (bool, optional): Whether to apply progressive encoding. Defaults to True.
            exclude (int, optional): Exclusion parameter for Fourier features. Defaults to 0.
            text_encoding_dim (int, optional): Dimensionality of the text encoding. Defaults to 512.
        """
        super(NeuralStyleField, self).__init__()
        self.pe = ProgressiveEncoding(mapping_size=width, T=niter, d=input_dim)
        self.clamp = clamp
        self.normclamp = normclamp
        self.normratio = normratio
        layers = []
        
        combined_input_dim = input_dim + text_encoding_dim

        if encoding == "gaussian":
            layers.append(FourierFeatureTransform(combined_input_dim, width, sigma, exclude))
            if progressive_encoding:
                layers.append(self.pe)
            layers.append(nn.Linear(width * 2 + combined_input_dim, width))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(combined_input_dim, width))
            layers.append(nn.ReLU())

        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        self.base = nn.ModuleList(layers)

        # Color branch
        color_layers = []
        for _ in range(colordepth):
            color_layers.append(nn.Linear(width, width))
            color_layers.append(nn.ReLU())
        color_layers.append(nn.Linear(width, 3))
        self.mlp_rgb = nn.ModuleList(color_layers)

        # Normal branch
        normal_layers = []
        for _ in range(normdepth):
            normal_layers.append(nn.Linear(width, width))
            normal_layers.append(nn.ReLU())
        normal_layers.append(nn.Linear(width, 1))
        self.mlp_normal = nn.ModuleList(normal_layers)

        print(self.base)
        print(self.mlp_rgb)
        print(self.mlp_normal)

    def reset_weights(self):
        """Reset the weights of the output layers of the MLP branches."""
        self.mlp_rgb[-1].weight.data.zero_()
        self.mlp_rgb[-1].bias.data.zero_()
        self.mlp_normal[-1].weight.data.zero_()
        self.mlp_normal[-1].bias.data.zero_()

    def forward(self, vertices: torch.Tensor, text_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the NeuralStyleField module.

        Args:
            vertices (torch.Tensor): Input tensor of vertices.
            text_encoding (torch.Tensor): Encoded text tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors for colors and displacements.
        """
        # Concatenate vertices with the text encoding
        combined_input = torch.cat((vertices, text_encoding.expand(vertices.shape[0], -1)), dim=1)
        
        for layer in self.base:
            combined_input = layer(combined_input)
        colors = self.mlp_rgb[0](combined_input)
        for layer in self.mlp_rgb[1:]:
            colors = layer(colors)
        displ = self.mlp_normal[0](combined_input)
        for layer in self.mlp_normal[1:]:
            displ = layer(displ)

        if self.clamp == "tanh":
            colors = F.tanh(colors) / 2
        elif self.clamp == "clamp":
            colors = torch.clamp(colors, 0, 1)

        if self.normclamp == "tanh":
            displ = F.tanh(displ) * self.normratio
        elif self.normclamp == "clamp":
            displ = torch.clamp(displ, -self.normratio, self.normratio)

        return colors, displ


def save_model(
    model: nn.Module,
    optim: optim.Optimizer,
    lr_scheduler: Optional[optim.lr_scheduler._LRScheduler],
    loss: float,
    output_dir: str,
) -> None:
    """Save the model, optimizer, and learning rate scheduler to a checkpoint file.
    
    Args:
        model (nn.Module): The model to save. A PyTorch neural network module.
        optim (optim.Optimizer): The optimizer used to train the model.
        lr_scheduler (Optional[optim.lr_scheduler._LRScheduler]): The learning rate scheduler used to train the model.
        loss (float): The last computed loss value.
        output_dir (str): The directory to save the checkpoint file.
    """
    save_dict: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "scheduler_state_dict": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "loss": loss,
    }

    filename = f"checkpoint_{ datetime.now().strftime("%d%m%Y_%H%M%S") }.pth.tar"
    path = os.path.join(output_dir, filename)

    torch.save(save_dict, path)

def load_model(
    model: nn.Module,
    optim: optim.Optimizer,
    lr_scheduler: Optional[optim.lr_scheduler._LRScheduler],
    model_path: str
) -> Tuple[nn.Module, optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler], float]:
    """Load a model, optimizer, and learning rate scheduler from a checkpoint file.

    Args:
        model (nn.Module): The model to load. A PyTorch neural network module.
        optim (optim.Optimizer): The optimizer to load.
        lr_scheduler (Optional[optim.lr_scheduler._LRScheduler]): The learning rate scheduler to load.
        model_path (str): The path to the checkpoint file.

    Returns:
        Tuple[nn.Module, optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler], float]: The loaded model, optimizer, learning rate scheduler, and loss value.
    """
    
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    loss = checkpoint["loss"]
    
    return model, optim, lr_scheduler, loss