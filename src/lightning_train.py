from big_brain.data.datamodules import TFDataModule
from big_brain.models.transformer import DWIBert
import pytorch_lightning as pl

dm = TFDataModule(data_dir="data/transformer", batch_size=8)
model = DWIBert()
trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=1)
trainer.fit(model, datamodule=dm)