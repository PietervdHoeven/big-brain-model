import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import StructuralSimilarityIndexMeasure
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from big_brain.models.encoders import Encoder
from big_brain.models.decoders import Decoder

class AutoEncoder(pl.LightningModule):
    def __init__(
            self,
            # Model parameters
            norm: str = "group",
            activation: str = "gelu",
            # Optimizer parameters
            lr: float = 0.001,
            weight_decay: float = 1e-05
            ):
        super().__init__()
        self.save_hyperparameters()

        # Optimizer parameters
        self.lr = lr
        self.weight_decay = weight_decay

        # Model architecture
        self.encoder = Encoder(norm=norm, activation=activation)
        self.decoder = Decoder(norm=norm, activation=activation)

        # Metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x):
        """
        Forward pass through the AutoEncoder.
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W].
        Returns:
            torch.Tensor: Reconstructed tensor of shape [B, C, D, H, W].
        """
        # Enode the input tensor
        z = self.encoder(x)
        # Decode the latent representation
        y = self.decoder(z)
        return y
    
    def _step(self, batch, stage: str):
        """
        Perform a single step of the training/validation loop.
        Args:
            batch (torch.Tensor): Input batch of shape [B, C, D, H, W].
            batch_idx (int): Index of the batch.
        Returns:
            torch.Tensor: Loss value.
        """
        # Forward pass
        x = batch["vol"]
        y = self(x)

        # Calculate loss and metrics
        loss = nn.functional.mse_loss(y, x)

        return x, y, loss
    
    def training_step(self, batch, batch_idx):
        # Perform a single training step.
        _, _, loss = self._step(batch, 'train')
        # Log the training loss
        self.log('train/loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        # Return the loss for the optimizer
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Perform a single validation step.
        x, y, loss = self._step(batch, 'val')
        ssim = self.ssim(y, x)
        # Log the validation loss
        self.log('val/loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        # Log the SSIM metric
        self.log('val/ssim', ssim, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        # Return the loss for validation
        return loss
    
    def test_step(self, batch, batch_idx):
        # Perform a single test step.
        x, y, loss = self._step(batch, 'test')
        ssim = self.ssim(y, x)
        # Log the test loss
        self.log('test/loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        # Log the SSIM metric for testing
        self.log('test/ssim', ssim, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        # Return the loss for testing
        return loss
    
    def configure_optimizers(self):
        """
        Configure the optimizer for the AutoEncoder.
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

# x = torch.randn(16, 1, 96, 112, 96)  # Example input tensor
# output = ae(x)
# print(f"Output shape: {output.shape}")  # Should be [B, 1, 96, 112, 96]