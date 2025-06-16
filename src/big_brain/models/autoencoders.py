import torch
import torch.nn as nn

class autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y
    
