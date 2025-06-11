import torch
import torch.nn as nn

class ConvAutoEncoder(nn.Module):
    """
    Modular 3D convolutional autoencoder.

    Args:
        encoder_cfg (list[nn.Module]): list of instantiated encoder blocks
        decoder_cfg (list[nn.Module]): list of instantiated decoder blocks
        input_shape (tuple): shape of one input volume (C, D, H, W)
        latent_dim (int): size of the bottleneck vector
    """
    def __init__(
        self,
        encoder_cfg: list,
        decoder_cfg: list,
        input_shape: tuple = (1, 96, 112, 96),
        latent_dim: int = 512,
    ):
        super().__init__()

        # Build encoder and decoder 
        self.encoder = nn.Sequential(*encoder_cfg)
        self.decoder = nn.Sequential(*decoder_cfg)

        # # Print for sanity
        # print(f"Encoder blocks:\n{self.encoder}\n")
        # print(f"Decoder blocks:\n{self.decoder}\n")

        # Probe encoder to find flattened feature size 
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)    # (B=1, C, D, H, W)
            feat  = self.encoder(dummy)             # -> (1, C_e, D_e, H_e, W_e)
            self._enc_feat_shape = feat.shape       # save for reshape
            flat_features = feat.numel()            # total # elements

        # Bottleneck MLP 
        self.flatten      = nn.Flatten()
        self.bottleneck   = nn.Linear(flat_features, latent_dim)
        self.unbottleneck = nn.Linear(latent_dim, flat_features)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, D, H, W)
        returns:
            x_hat: reconstructed volume, same shape as x
            z:      latent code, shape (B, latent_dim)
        """
        # Encode
        f = self.encoder(x)                       # (B, C_e, D_e, H_e, W_e)
        g = self.flatten(f)                       # (B, C_e*D_e*H_e*W_e)
        z = self.bottleneck(g)                    # (B, latent_dim)

        # Decode
        u = self.unbottleneck(z)                            # (B, C_e*D_e*H_e*W_e)
        u = u.view(x.size(0), *self._enc_feat_shape[1:])    # (B, C_e, D_e, H_e, W_e)
        x_hat = self.decoder(u)                             # (B, C, D, H, W)



        # print(
        #     f"Input shape: {x.shape}",
        #     f"\nEncoded shape: {f.shape}",
        #     f"\nLatent shape: {z.shape}",
        #     f"\nDecoded shape: {u.shape},"
        #     f"\nReconstructed shape: {x_hat.shape}")

        return x_hat