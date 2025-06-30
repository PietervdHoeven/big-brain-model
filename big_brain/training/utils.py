# src/big_brain/training/utils.py

import logging
import hydra
from pathlib import Path
import json
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)

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


def build_callbacks(cfg):
    """
    Build a list of callbacks from the configuration.
    """
    cb_list = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            cb = hydra.utils.instantiate(cb_conf)
            cb_list.append(cb)
    return cb_list