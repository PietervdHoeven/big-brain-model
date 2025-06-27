import torch
import torch.nn as nn
import torch.nn.functional as F

from big_brain.models.blocks import TokenEmbedder, TiedLinear



# embedder = TokenEmbedder(d_model=384)

# x = embedder(z, g)
# print("input z shape:", z.shape)  # Should be [B, L, D]
# print("input g shape:", g.shape)  # Should be [B, L, 4]
# print(f"Output shape: {x.shape}")  # Should be [B, L, D]

# h = encoder(x, src_key_padding_mask=attn)

# print(f"Encoder output shape: {h.shape}")  # Should be [B, L, D]

class DWIBert(nn.Module):
    def __init__(self,
                 d_model: int = 384,
                 n_head: int = 6,
                 d_ff: int = 1536,
                 n_layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()

        self.embedder = TokenEmbedder(d_model=d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='relu',
                batch_first=True,  # Ensure input is [B, L, D]
                norm_first=True  # Use pre-norm following J&M
            ),
            num_layers=n_layers
        )

        self.reconstructor = TiedLinear(self.embedder.proj_z.weight)

    def forward(self, 
                z: torch.Tensor, 
                g: torch.Tensor, 
                attn_mask: torch.Tensor = None
                ) -> torch.Tensor:
        """
        Forward pass through the DWI-BERT model.
        Args:
            z (torch.Tensor): Latent vector of shape [B, L, 512].
            g (torch.Tensor): Gradient vector of shape [B, L, 4].
            attn_mask (torch.Tensor, optional): Attention mask of shape [B, L]. Defaults to None.
        Returns:
            torch.Tensor: Reconstructed latent vector of shape [B, L, 512].
        """
        # project z and g to the model dimension
        x = self.embedder(z, g)                 # Shape [B, L, D]
        x[:, 0] = x[:, 0] + self.cls_token      # Add [CLS] token at the start

        # Pass through the encoder
        h = self.encoder(x, src_key_padding_mask=~attn_mask)  # Encoder output [B, L, D]

        # Reconstruct the latent vector
        z_pred = self.reconstructor(h)

        return z_pred  # Shape [B, L, 512], reconstructed latent vector

# B, L = 4, 100
# z = torch.randn(B, L, 512)  # Example input tensor
# g = torch.randn(B, L, 4)  # Example gradient tensor
# attn = torch.ones(B, L, dtype=torch.bool)  # Example attention mask
# mdm_mask = torch.rand(B, L) < 0.15  # Example mask for MDM loss
# labels = z.clone()  # Example labels for MDM loss

# # dwi_bert = DWIBert(d_model=384, n_head=6, d_ff=1536, n_layers=6, dropout=0.1)
# # z_pred = dwi_bert(z, g, attn_mask=attn)
# # print(f"Output shape: {z_pred.shape}")  # Should be [B, L, 512]

# model = DWIBert()
# opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Z_hat = model(z, g, attn)               # forward
# loss = F.smooth_l1_loss(Z_hat[mdm_mask], labels[mdm_mask])
# loss.backward()

# # Verify weight tying
# W = model.embedder.proj_z.weight
# grad_embed = W.grad.clone()
# grad_recon = model.reconstructor.weight.grad.clone()  # same object!

# assert torch.allclose(grad_embed, grad_recon)   