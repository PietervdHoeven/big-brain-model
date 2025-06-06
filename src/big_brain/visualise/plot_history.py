# src/big_brain/visualize/plot_history.py
import sys, json
from pathlib import Path
import pandas as pd
import seaborn as sns

def main(run_dir: Path):
    # 1) load history
    hist_json = run_dir / "history.json"
    if hist_json.exists():
        with open(hist_json) as f:
            history = json.load(f)
        df = pd.DataFrame(history)
    else:                                 # fallback to CSV
        df = pd.read_csv(run_dir / "history.csv")

    # 2) plot
    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(data=df, x="epoch", y="train_loss",
                      marker="o", label="train")
    sns.lineplot(data=df, x="epoch", y="val_loss",
                 marker="x", linestyle="--", label="val", ax=ax)
    ax.set(xlabel="Epoch", ylabel="MSE loss", title="AE reconstruction loss")
    fig = ax.get_figure()

    # 3) save under run dir
    out_png = run_dir / "loss_curve.png"
    out_pdf = run_dir / "loss_curve.pdf"
    fig.savefig(out_png, dpi=300); fig.savefig(out_pdf)
    print(f"Saved curves → {out_png}  &  {out_pdf}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m big_brain.visualize.plot_history <run_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]).expanduser().resolve())
