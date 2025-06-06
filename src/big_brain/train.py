# src/big_brain/train.py

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import seed_everything
import pandas as pd

from big_brain.training.utils import run_epoch
from big_brain.training.stopping import EarlyStopper


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # 0.  Print merged config
    print("Merged Hydra config: \n", OmegaConf.to_yaml(cfg))

    # 1.  Reproducibility & device
    seed_everything(cfg.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2.  Instantiate DataModule, Model, Loss
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader   = datamodule.val_dataloader()

    model: nn.Module    = hydra.utils.instantiate(cfg.model).to(device)
    criterion: nn.Module= hydra.utils.instantiate(cfg.loss)

    # 3.  Build optimiser from config
    lr       = cfg.trainer.learning_rate
    opt_name = cfg.trainer.optimizer.lower()
    if opt_name == "adam":
        optimiser = optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        optimiser = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.trainer.optimizer}")

    # 4.  Early-stopping & checkpointing helpers
    stopper = EarlyStopper(
        patience=cfg.trainer.early_stopping.patience,
        delta=cfg.trainer.early_stopping.delta,
        mode="min",
    )

    # make checkpoint folder under Hydra run dir
    best_ckpt = Path.cwd() / "ae_checkpoint.pth"

    # 5.  Epoch loop
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.trainer.max_epochs + 1):
        train_loss = run_epoch(
            model, train_loader, criterion,
            optimiser=optimiser, device=device
        )

        torch.cuda.empty_cache()  # small OOM guard

        val_loss = run_epoch(
            model, val_loader, criterion,
            optimiser=None, device=device
        )

        # logging & history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[{epoch:03d}/{cfg.trainer.max_epochs}] "
              f"train {train_loss:.4e}  |  val {val_loss:.4e}")

        # early-stop & checkpoint
        improved = stopper.step(val_loss)
        if improved:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "val_loss": val_loss,
                },
                best_ckpt,
            )
            print(f"New best model saved -> {best_ckpt}  (val={val_loss:.4e})")

        if stopper.should_stop:
            print(f"Early stopping triggered (patience={stopper.patience}).")
            break

    # 6.  Save history under Hydra run dir (JSON + CSV)
    hist_path_json = Path.cwd() / "history.json"
    hist_path_csv  = hist_path_json.with_suffix(".csv")

    with hist_path_json.open("w") as fp:
        json.dump(history, fp, indent=2)

    pd.DataFrame(history).to_csv(hist_path_csv, index=False)
    print(f"History saved to {hist_path_json} and {hist_path_csv}")


if __name__ == "__main__":
    main()
