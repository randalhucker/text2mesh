import os
from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch.optim

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
        """
        super(NeuralStyleField, self).__init__()
        self.pe = ProgressiveEncoding(mapping_size=width, T=niter, d=input_dim)
        self.clamp = clamp
        self.normclamp = normclamp
        self.normratio = normratio
        layers = []

        if encoding == "gaussian":
            layers.append(FourierFeatureTransform(input_dim, width, sigma, exclude))
            if progressive_encoding:
                layers.append(self.pe)
            layers.append(nn.Linear(width * 2 + input_dim, width))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(input_dim, width))
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the NeuralStyleField module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors for colors and displacements.
        """
        for layer in self.base:
            x = layer(x)
        colors = self.mlp_rgb[0](x)
        for layer in self.mlp_rgb[1:]:
            colors = layer(colors)
        displ = self.mlp_normal[0](x)
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


def save_model(model, loss, iter, optim, output_dir):
    save_dict = {
        "iter": iter,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "loss": loss,
    }

    path = os.path.join(output_dir, "checkpoint.pth.tar")

    torch.save(save_dict, path)
