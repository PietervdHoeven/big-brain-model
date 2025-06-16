import torch
import torch.nn as nn

from big_brain.models.blocks import ConvLayer, DeconvLayer

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # [B, 512, 1, 1, 1]
        self.unbottle1 = DeconvLayer(512, 512, kernel_size=(2,3,2), stride=1, padding=0)
        self.unbottle2 = DeconvLayer(512, 256, kernel_size=3, stride=1, padding=0)
        self.unbottle3 = DeconvLayer(256, 128, kernel_size=3, stride=1, padding=0)
        # [B, 256, 6, 7, 6]
        self.up1 = DeconvLayer(128, 128, kernel_size=2, stride=2, padding=0)
        self.conv1 = ConvLayer(128, 64, pool=None)
        # [B, 128, 12, 14, 12]
        self.up2 = DeconvLayer(64, 64, kernel_size=2, stride=2, padding=0)
        self.conv2 = ConvLayer(64, 32, pool=None)
        # [B, 64, 24, 28, 24]
        self.up3 = DeconvLayer(32, 32, kernel_size=2, stride=2, padding=0)
        self.conv3 = ConvLayer(32, 16, pool=None)
        # [B, 32, 48, 56, 48]
        self.up4 = DeconvLayer(16, 16, kernel_size=2, stride=2, padding=0)
        self.conv4 = ConvLayer(16, 1, pool=None)
        # [B, 1, 96, 112, 96]

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        for layer in self._modules.values():
            x = layer(x)
            print(f"After {layer.__class__.__name__}: {x.shape}")
        return x
    
# Example usage
x = torch.randn(16, 512, 1, 1, 1)  # Example input tensor
decoder = Decoder()
output = decoder(x)
print(f"Output shape: {output.shape}")  # Should be [B, 1, 96, 112, 96]