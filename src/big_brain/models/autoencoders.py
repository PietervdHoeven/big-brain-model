import torch
import torch.nn as nn
from big_brain.models.encoders import Encoder
from big_brain.models.decoders import Decoder

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y
    
ae = AutoEncoder(Encoder(), Decoder())

# x = torch.randn(16, 1, 96, 112, 96)  # Example input tensor
# output = ae(x)
# print(f"Output shape: {output.shape}")  # Should be [B, 1, 96, 112, 96]