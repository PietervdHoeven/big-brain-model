import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_schedulers
from pytorch_lightning import seed_everything
import pandas as pd

from big_brain.visualise import plot_history
from big_brain.visualise import show_recon

from big_brain.training.utils import run_epoch
from big_brain.training.stopping import EarlyStopper

import logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # 0. log merged config and save to output directory -----------------------------
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info("Merged Hydra config:\n" + OmegaConf.to_yaml(cfg))
    config_save_path = out_dir / "config.yaml"
    OmegaConf.save(cfg, config_save_path)
    log.info(f"Hydra config saved to {config_save_path}")

    # 1.  Reproducibility & device ------------------------------------------------
    seed_everything(cfg.seed, workers=True, verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2.  Instantiate DataModule, Model, Loss ------------------------------------
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader   = datamodule.val_dataloader()

    model: nn.Module   = hydra.utils.instantiate(cfg.model).to(device)

    loss_fn: nn.Module = hydra.utils.instantiate(cfg.loss)

    # 3.  Build optimiser ---------------------------------------------------------
    optimiser = hydra.utils.instantiate(cfg.trainer.optimiser, model.parameters())

    # 3b.  learning‑rate scheduler -----------------------------------------------
    scheduler = hydra.utils.instantiate(cfg.trainer.lr_scheduler, optimiser)

    # 4.  Early‑stopping & checkpoint helpers ------------------------------------
    stopper = hydra.utils.instantiate(cfg.trainer.early_stopping)
    best_ckpt = out_dir / "checkpoint.pth"

    # 5.  Epoch loop --------------------------------------------------------------
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.trainer.max_epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn,
                               optimiser=optimiser, device=device)

        torch.cuda.empty_cache()

        val_loss = run_epoch(model, val_loader, loss_fn,
                             optimiser=None, device=device)

        # logging
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        log.info(f"[{epoch:03d}/{cfg.trainer.max_epochs}] train {train_loss:.4e} | val {val_loss:.4e}")

        # scheduler step (if configured)
        scheduler.step(val_loss)
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

    # 6.  Persist history ---------------------------------------------------------
    hist_path_json = out_dir / "history.json"
    hist_path_csv  = out_dir / "history.csv"

    with hist_path_json.open("w") as fp:
        json.dump(history, fp, indent=2)
    pandas_df = pd.DataFrame(history)
    pandas_df.to_csv(hist_path_csv, index=False)
    log.info(f"History saved to {hist_path_json} & {hist_path_csv}")

    # 7.  Optional: auto‑generate visualisations ----------------------------------
    if cfg.trainer.save_plots:
        try:
            log.info("Generating visualisations (loss curves & reconstructions)...")
            plot_history.main(out_dir)
            n_recon = getattr(cfg.trainer, "n_recon", 4)
            show_recon.main(out_dir, n_show=n_recon)
            log.info("All plots saved in run directory.")
        except Exception as e:
            log.info(f"[WARN] Plot generation failed: {e}")


if __name__ == "__main__":
    main()
