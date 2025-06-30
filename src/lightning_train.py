from big_brain.data.datamodules import TFDataModule, AEDataModule
from big_brain.models.transformer import DWIBert
from big_brain.models.autoencoders import AutoEncoder
import pytorch_lightning as pl

SEED = 42  # Set a seed for reproducibility
pl.seed_everything(SEED)  # For reproducibility
# dm = TFDataModule(data_dir="data/transformer", batch_size=8, seed=SEED)
# model = DWIBert()

dm = AEDataModule(data_dir="data/autoencoder", seed=SEED, use_sampler=True, alpha=0.3, batch_size=6)
model = AutoEncoder()
trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=1, gradient_clip_val=1.0, precision="16-mixed")
trainer.fit(model, datamodule=dm)