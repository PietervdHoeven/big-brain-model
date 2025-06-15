import torch
import torch.nn as nn

from typing import Optional

from big_brain.models.layers import UpconvLayer, ConvLayer

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            x: torch.Tensor,
            residuals: Optional[list[torch.Tensor]] = None
            ) -> torch.Tensor:
        # Forward pass through the decoder.
        for i, layer in enumerate(self._modules.values()):
            if residuals is not None:
                r = residuals[-(i + 1)]  # Get the corresponding residual (reverse order, last encoder r becomes first decoder)
                x = torch.cat((x, r), dim=1)  # Concatenate along the channel dimension
            x = layer(x)
        return x
    

class ConvDecoder(Decoder):
    def __init__(self, skip_connections: bool = False):
        super().__init__()
        self.multiplier = 2 if skip_connections else 1
        # [32, 6, 7, 6]
        self.upconv1a = UpconvLayer(32 * self.multiplier, 128, kernel_size=2, stride=2)
        self.conv1b = ConvLayer()

