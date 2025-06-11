import torch
import torch.nn as nn

from typing import Optional

class Decoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(
            self,
            x: torch.Tensor,
            residuals: Optional[list[torch.Tensor]] = None
            ) -> torch.Tensor:
        # Forward pass through the decoder.
        if residuals is None:               # if no residuals are provided
            for layer in self.blocks:       # just pass through the layers
                x = layer(x)
            return x
        for idx, layer in enumerate(self.blocks):   # iterate through the layers
            skip = residuals[-(idx +1)]             # get residuals, starting from the last layer in the encoder
            x = torch.cat([x, skip], dim=1)         # concatenate the skip connection
            x = layer(x)                            # Compute the layer output
        return x