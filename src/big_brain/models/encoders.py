import torch
import torch.nn as nn
from typing import Optional
from big_brain.models.blocks import ConvLayer, DepthConvLayer

class Encoder(nn.Module):
    def __init__(self, blocks: nn.Sequential):
        super().__init__()
        self.blocks = blocks

    def forward(
            self, 
            x: torch.Tensor,
            collect_residuals: bool = False
            ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        # Forward pass through the encoder.
        if not collect_residuals:       # if residuals are not needed
            for layer in self.blocks:   # just pass through the layers
                x = layer(x)
            return x, None              # return only the output tensor
        
        # If residuals are needed, collect them during forward pass.
        residuals = []
        for layer in self.blocks:
            x = layer(x)
            residuals.append(x)
        return x, residuals        # return output tensor and list of residuals

      