import torch
import torch.nn as nn
from typing import Optional
from big_brain.models.layers import ConvLayer, DepthConvLayer

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self, 
            x: torch.Tensor,
            *,
            residuals_from: list[int] | None = [1, 3, 5, 8]
            ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        # Init residuals storage if needed
        residuals: list[torch.Tensor] = [] if residuals_from else None

        # Forward pass through the blocks
        for i, layer in enumerate(self._modules.values()): # _modules.values() returns the layers in the order they were added
            x = layer(x)
            if i in residuals_from:
                residuals.append(x)
        return x, residuals


class ConvPoolEncoder(Encoder):
    def __init__(self):
        super().__init__()
        # [1, 96, 112, 96]
        self.conv1a = ConvLayer(1, 32, pool=None)
        self.conv1b = ConvLayer(32, 32, pool="max") # 1
        # [32, 48, 56, 48]
        self.conv2a = ConvLayer(32, 64, pool=None)
        self.conv2b = ConvLayer(64, 64, pool="max") # 3
        # [64, 24, 28, 24]
        self.conv3a = ConvLayer(64, 128, pool=None)
        self.conv3b = ConvLayer(128, 128, pool="max") # 5
        # [128, 12, 14, 12]
        self.conv4a = ConvLayer(128, 64, kernel_size=1, padding=0, pool=None)
        self.conv4b = ConvLayer(64, 64, pool="max")
        self.conv4c = ConvLayer(64, 32, kernel_size=1, padding=0, pool=None)    # 8
        # [32, 6, 7, 6]


class DepthPoolEncoder(Encoder):
    def __init__(self):
        super().__init__()
        # [1, 96, 112, 96]
        self.conv1a = DepthConvLayer(1, 32, pool=None)
        self.conv1b = DepthConvLayer(32, 32, pool="max")    # 1
        # [32, 48, 56, 48]
        self.conv2a = DepthConvLayer(32, 64, pool=None)
        self.conv2b = DepthConvLayer(64, 64, pool="max")    # 3
        # [64, 24, 28, 24]
        self.conv3a = DepthConvLayer(64, 128, pool=None)
        self.conv3b = DepthConvLayer(128, 128, pool="max")  # 5
        # [128, 12, 14, 12]
        self.conv4a = DepthConvLayer(128, 64, kernel_size=1, padding=0, pool=None)
        self.conv4b = DepthConvLayer(64, 64, pool="max")
        self.conv4c = DepthConvLayer(64, 32, kernel_size=1, padding=0, pool=None)   # 8
        # [32, 6, 7, 6]


class ConvStrideEncoder(Encoder):
    def __init__(self):
        super().__init__()
        # [1, 96, 112, 96]
        self.conv1a = ConvLayer(1, 32, pool=None)
        self.conv1b = ConvLayer(32, 32, padding=0, stride=2, pool=None)
        # [32, 48, 56, 48]
        self.conv2a = ConvLayer(32, 64, pool=None)
        self.conv2b = ConvLayer(64, 64, padding=0, stride=2, pool=None)
        # [64, 24, 28, 24]
        self.conv3a = ConvLayer(64, 128, pool=None)
        self.conv3b = ConvLayer(128, 128, padding=0, stride=2, pool=None)
        # [128, 12, 14, 12]
        self.conv4a = ConvLayer(128, 64, kernel_size=1, padding=0, pool=None)
        self.conv4b = ConvLayer(64, 64, padding=0, stride=2, pool=None)
        self.conv4c = ConvLayer(64, 32, kernel_size=1, padding=0, pool=None)
        # [32, 6, 7, 6]


class DepthStrideEncoder(Encoder):
    def __init__(self):
        super().__init__()
        # [1, 96, 112, 96]
        self.conv1a = DepthConvLayer(1, 32, pool=None)
        self.conv1b = DepthConvLayer(32, 32, padding=0, stride=2, pool=None)
        # [32, 48, 56, 48]
        self.conv2a = DepthConvLayer(32, 64, pool=None)
        self.conv2b = DepthConvLayer(64, 64, padding=0, stride=2, pool=None)
        # [64, 24, 28, 24]
        self.conv3a = DepthConvLayer(64, 128, pool=None)
        self.conv3b = DepthConvLayer(128, 128, padding=0, stride=2, pool=None)
        # [128, 12, 14, 12]
        self.conv4a = DepthConvLayer(128, 64, kernel_size=1, padding=0, pool=None)
        self.conv4b = DepthConvLayer(64, 64, padding=0, stride=2, pool=None)
        self.conv4c = DepthConvLayer(64, 32, kernel_size=1, padding=0, pool=None)
        # [32, 6, 7, 6]