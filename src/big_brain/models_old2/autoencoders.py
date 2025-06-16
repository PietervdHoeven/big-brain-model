import torch
import torch.nn as nn

from big_brain.models.encoders import Encoder
from big_brain.models.decoders import Decoder

class Autoencoder(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        latent_dim: int,
        skips: bool = False,
        input_size: tuple[int,int,int] = (96,112,96)
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.skips = skips
        self.latent_dim = latent_dim
        # infer bottleneck dims with dummy
        dummy = torch.zeros(1, *input_size)
        with torch.no_grad():
            z, _ = self.encoder(dummy, collect_residuals=False)
        flat_dim = z.numel()
        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, residuals = self.encoder(x, collect_residuals=self.skips)
        latent = self.fc_enc(z.view(z.size(0), -1))
        z_hat = self.fc_dec(latent).view_as(z)
        if self.skips:
            x_hat = self.decoder(z_hat, residuals)
        else:
            x_hat = self.decoder(z_hat)
        return x_hat
    
