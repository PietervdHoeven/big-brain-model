import torch.nn as nn
from big_brain.models.encoders import SlimEncoder01
from big_brain.models.decoders import SlimDecoder01

class SlimAutoencoder(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.encoder = SlimEncoder01(z_dim)
        self.decoder = SlimDecoder01(z_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
