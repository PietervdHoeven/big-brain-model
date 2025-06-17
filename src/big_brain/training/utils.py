# src/big_brain/training/utils.py

import logging
from tqdm import tqdm
import torch
from torch.amp import autocast, GradScaler

from pathlib import Path
import json
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)

# Instantiate a GradScaler for mixed precision
_SCALER = GradScaler()

def run_epoch(model, loader, loss_fn, optimiser=None, device='cuda'):
    """
    Train or validate for one epoch with optional AMP support.

    Args:
        model (nn.Module): the autoencoder model
        loader (DataLoader): data loader
        loss_fn (callable): loss function
        optimiser (Optimizer, optional): if provided, training mode is enabled
        device (str or torch.device): compute device

    Returns:
        float: average loss over the epoch
    """
    is_train = optimiser is not None
    model.train(mode=is_train)
    running_loss = 0.0
    n_batches = 0

    log.info("Mode: Train" if is_train else "Mode: Eval")
    # Optionally log dataset details if available
    try:
        patients = len(set(loader.dataset.patients))
        sessions = len(set(loader.dataset.sessions))
        log.debug(f"Patients: {patients}, Sessions: {sessions}")
    except Exception:
        pass

    # Choose appropriate context
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in tqdm(loader, desc=("Train" if is_train else "Val")):
            x = batch["vol"].to(device, dtype=torch.float32, non_blocking=True)

            # Forward + loss under autocast for mixed precision
            with autocast(enabled=is_train, device_type=str(device)):
                recon = model(x)
                loss = loss_fn(recon, x)

            if is_train:
                optimiser.zero_grad(set_to_none=True)
                # Scale loss, backward, unscale for grad clipping
                _SCALER.scale(loss).backward()
                _SCALER.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Optimizer step and update scaler
                _SCALER.step(optimiser)
                _SCALER.update()

            running_loss += loss.item()
            n_batches += 1

    avg_loss = running_loss / max(1, n_batches)
    log.info(f"Epoch completed. Avg loss: {avg_loss:.4e}")
    return avg_loss


def persist_history(history: dict, out_dir: Path) -> None:
    # Save training history to JSON and CSV files
    hist_path_json = out_dir / "history.json"
    hist_path_csv  = out_dir / "history.csv"
    # dump to JSON
    with hist_path_json.open("w") as fp:
        json.dump(history, fp, indent=2)
    pandas_df = pd.DataFrame(history)
    pandas_df.to_csv(hist_path_csv, index=False)
    log.info(f"History saved to {hist_path_json} & {hist_path_csv}")


def plot_history(out_dir: Path) -> None:
    """
    Plot training history from JSON or CSV file.
    """
    # 1) load history
    hist_json = out_dir / "history.json"
    if hist_json.exists():
        with open(hist_json) as f:
            history = json.load(f)
        df = pd.DataFrame(history)
    else:                                 # fallback to CSV
        df = pd.read_csv(out_dir / "history.csv")

    # 2) plot
    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(data=df, x="epoch", y="train_loss",
                      marker="o", label="train")
    sns.lineplot(data=df, x="epoch", y="val_loss",
                 marker="x", linestyle="--", label="val", ax=ax)
    ax.set(xlabel="Epoch", ylabel="MSE loss", title="AE reconstruction loss")
    fig = ax.get_figure()

    # 3) save under out_dir
    out_png = out_dir / "loss_curve.png"
    out_pdf = out_dir / "loss_curve.pdf"
    fig.savefig(out_png, dpi=300); fig.savefig(out_pdf)
    log.info(f"Saved curves -> {out_png}  &  {out_pdf}")