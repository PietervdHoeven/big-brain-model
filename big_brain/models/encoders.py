import torch
import torch.nn as nn

from big_brain.models.blocks import ConvLayer

class Encoder(nn.Module):
    def __init__(self, norm: str = "group", activation: str = "gelu"):
        super().__init__()
        # [B, 1, 96, 112, 96]
        self.conv1 = ConvLayer(1,  16, pool=None, activation=activation, norm=norm)
        self.down1 = ConvLayer(
            16, 16, kernel_size=2, stride=2, padding=0, pool=None, activation=activation, norm=norm)
        
        # [B, 16, 48, 56, 48]
        self.conv2 = ConvLayer(16, 32, pool=None)
        self.down2 = ConvLayer(
            32, 32, kernel_size=2, stride=2, padding=0, pool=None, activation=activation, norm=norm)
        
        # [B, 32, 24, 28, 24]
        self.conv3 = ConvLayer(32, 64, pool=None)
        self.down3 = ConvLayer(
            64, 64, kernel_size=2, stride=2, padding=0, pool=None, activation=activation, norm=norm)
        
        # [B, 64, 12, 14, 12]
        self.conv4 = ConvLayer(64, 128, pool=None)
        self.down4 = ConvLayer(
            128, 128, kernel_size=2, stride=2, padding=0, pool=None, activation=activation, norm=norm)
        
        # [B, 128, 6, 7, 6]
        self.bottleneck1 = ConvLayer(
            128, 256, kernel_size=3, stride=1, padding=0, pool=None, activation=activation, norm=norm)
        self.bottleneck2 = ConvLayer(
            256, 512, kernel_size=3, stride=1, padding=0, pool=None, activation=activation, norm=norm)
        self.bottleneck3 = ConvLayer(
            512, 512, kernel_size=(2,3,2), stride=1, padding=0, pool=None, activation=activation, norm=norm,
            dropout=0.1)
        
        # [B, 512, 1, 1, 1]

    def forward(
            self, 
            x: torch.Tensor,
            ) -> torch.Tensor:
        # Forward pass through the blocks
        for layer in self._modules.values(): # _modules.values() returns the layers in the order they were added
            x = layer(x)
            #print(f"After {layer.__class__.__name__}: {x.shape}")
        return x
    
# x = torch.randn(16, 1, 96, 112, 96)  # Example input tensor
# encoder = Encoder()
# output = encoder(x)
# print(f"Output shape: {output.shape}")  # Should be [B, 512, 1, 1, 1]