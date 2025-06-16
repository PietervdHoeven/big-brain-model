# src/big_brain/train.py
from pathlib import Path
import gc

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything

from big_brain.training.utils import run_epoch, persist_history, generate_visualisations

import logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # 0. Clean up cached memory from GPU (if any) -----------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    log.info("Cleared cache.")

    # 1. log merged config and save to output directory -----------------------------
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info("Merged Hydra config:\n" + OmegaConf.to_yaml(cfg))
    config_save_path = out_dir / "config.yaml"
    OmegaConf.save(cfg, config_save_path)
    log.info(f"Hydra config saved to {config_save_path}")

    # 2.  Reproducibility & device ------------------------------------------------
    seed_everything(cfg.seed, workers=True, verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3.  Instantiate DataModule, Model, Loss ------------------------------------
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader   = datamodule.val_dataloader()

    model: nn.Module   = hydra.utils.instantiate(cfg.model).to(device)

    loss_fn: nn.Module = hydra.utils.instantiate(cfg.loss)

    # 4.  Build optimiser ---------------------------------------------------------
    optimiser = hydra.utils.instantiate(cfg.optimiser, model.parameters())

    # 4b.  learning‑rate scheduler -----------------------------------------------
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimiser)

    # 5.  Early‑stopping & checkpoint helpers ------------------------------------
    stopper = hydra.utils.instantiate(cfg.stopper)

    # 6.  Epoch loop --------------------------------------------------------------
    best_ckpt = out_dir / "checkpoint.pth"
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn,
                            optimiser=optimiser, device=device)

        torch.cuda.empty_cache()

        val_loss = run_epoch(model, val_loader, loss_fn,
                            optimiser=None, device=device)

        # logging
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        log.info(f"[{epoch:03d}/{cfg.max_epochs}] train {train_loss:.4f} | val {val_loss:.4f}")

        # scheduler step (if configured)
        scheduler.step()
        log.info(f"Current learning rate: {scheduler.get_last_lr()}")

        # early stop & checkpoint
        improved = stopper.step(val_loss)
        if improved:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "val_loss":  val_loss,
            }, best_ckpt)
            log.info(f"New best model saved -> {best_ckpt}  (val={val_loss:.4e})")

        if stopper.should_stop:
            log.info(f"Early stopping triggered (patience={stopper.patience}).")
            break

    # 7.  Persist history ---------------------------------------------------------
    persist_history(history, out_dir)

    # 8.  auto‑generate visualisations ----------------------------------
    generate_visualisations(out_dir)


if __name__ == "__main__":
    main()
