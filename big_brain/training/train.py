# src/train.py
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from importlib import import_module
import logging
log = logging.getLogger(__name__)

def _build_callbacks(cfg) -> list[Callback]:
    cb_list = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            cb = hydra.utils.instantiate(cb_conf)
            cb_list.append(cb)
    return cb_list

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))

    # 1) reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # 2) objects
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model      = hydra.utils.instantiate(cfg.model)
    callbacks  = _build_callbacks(cfg)

    # 3) trainer (accepts arbitrary flags via cfg.trainer)
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks)

    # 4) run
    if cfg.train:
        trainer.fit(model, datamodule=datamodule)

    if cfg.test:
        trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
