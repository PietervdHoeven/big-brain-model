# big_brain/train.py
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    # Print the configuration
    print(OmegaConf.to_yaml(cfg))

    # set random seed for reproducibility
    seed_everything(cfg.seed, workers=True)

    # set logger
    logger = instantiate(cfg.logger)

    # set callbacks
    earlystopping = instantiate(cfg.early_stopping)
    checkpoint = instantiate(cfg.checkpoint)
    callbacks = [earlystopping, checkpoint]

    # Instantiate the Trainer
    trainer = Trainer()

    # instantiate the model
    model = instantiate(cfg.model)

    # instantiate the data module
    datamodule = instantiate(cfg.datamodule)

    # instantiate the trainer
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # Train
    trainer.fit(model=model, datamodule=datamodule)

    # Test (optional)
    if cfg.test:
        trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()
