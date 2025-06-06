# src/big_brain/train.py
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(42, workers=True)

    # Placeholder: comment these out until you add data.py and models/__init__.py
    # model = build_model(cfg.model)
    # model.criterion = hydra.utils.instantiate(cfg.loss)
    # datamodule = hydra.utils.instantiate(cfg.datamodule)
    # trainer = pl.Trainer(**cfg.trainer)
    # trainer.fit(model, datamodule)

    print("✅  Configuration loaded successfully:")
    print(cfg)

if __name__ == "__main__":
    main()
