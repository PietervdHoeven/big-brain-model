import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsoluteError

from big_brain.models.blocks import TokenEmbedder, TiedLinear

class DWIBert(pl.LightningModule):
    def __init__(
            self,
            # Model parameters
            d_model: int = 384,
            n_head: int = 6,
            d_ff: int = 1536,
            n_layers: int = 6,
            dropout: float = 0.1,
            # Optimizer parameters
            lr: float = 1e-4,
            weight_decay: float = 1e-2
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        # Model architecture
        self.embedder = TokenEmbedder(d_model=d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # Ensure input is [B, L, D]
            norm_first=True  # Use pre-norm following J&M
        )

        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
        self.reconstructor = TiedLinear(self.embedder.proj_z.weight)

        # metrics
        self.mae = MeanAbsoluteError()

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
    
    def _step(self, batch, stage: str):
        """ Perform a single training/validation step.
        Args:
            batch (tuple): A tuple containing (z, g, attn_mask, labels, mdm_mask).
            stage (str): The stage of the training process ('train', 'val', 'test').
        Returns:
            torch.Tensor: The loss value for the step.
        """
        # Unpack the batch
        z, g, attn_mask, labels, mdm_mask = batch

        # Forward pass
        z_pred = self.forward(z, g, attn_mask)

        # Compute the loss and other metrics
        loss = F.smooth_l1_loss(z_pred[mdm_mask], labels[mdm_mask])
        mae = self.mae(z_pred[mdm_mask], labels[mdm_mask])

        # Log the loss and metrics
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log(f"{stage}_mae", mae, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')
    
    def configure_optimizers(self):
        """ Configure the optimizer and the learning rate scheduler for the model.
        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        # Use AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Calculate the total number of training steps and take a warmup proportion
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.05)
        cosine_steps = total_steps - warmup_steps

        # Define the schedulers for cosine annealing and warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=self.hparams.lr * 0.01
        )

        # Combine the schedulers into a SequentialLR
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update the scheduler every step
                'frequency': 1,      # Frequency of updates
            }
        }
    
# Example usage:
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