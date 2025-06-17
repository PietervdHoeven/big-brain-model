# src/big_brain/visualize/show_recon.py
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra

import logging
log = logging.getLogger(__name__)

def load_cfg(run_dir: Path):
    cfg_path = run_dir / "config.yaml"
    return OmegaConf.load(cfg_path)

def sample_slices(vol: torch.Tensor):
    """Return axial, coronal, sagittal mid-slices (HxW arrays)"""
    z, y, x = vol.shape[-3:]
    axial     = vol[..., z // 2, :, : ]
    coronal   = vol[..., :, y // 2, : ]
    sagittal  = vol[..., :, :, x // 2 ]
    return axial, coronal, sagittal

def main(run_dir: Path, n_show: int = 4):
    # 1) read cfg and re-instantiate datamodule + model
    cfg = load_cfg(run_dir)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()                     # use val set
    val_loader = datamodule.val_dataloader()

    model = hydra.utils.instantiate(cfg.model)
    ckpt  = torch.load(run_dir / "checkpoint.pth",
                       map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2) collect N random samples from validation data
    vols = []
    for batch in val_loader:
        vols.append(batch["vol"])
        if len(torch.cat(vols)) >= n_show:
            break
    samples = torch.cat(vols)[:n_show].to(device)

    # 3) run reconstructions
    with torch.no_grad():
        recons = model(samples)

    # 4) plot grid (N rows × 3 cols × 2 (input+recon) = 6 cols)
    rows, cols = n_show, 6
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    col_titles = ["Axial-IN", "Cor-IN", "Sag-IN",
                  "Axial-REC", "Cor-REC", "Sag-REC"]

    for r in range(rows):
        in_axial, in_cor, in_sag = sample_slices(samples[r, 0])
        rc_axial, rc_cor, rc_sag = sample_slices(recons[r, 0])

        slices = [in_axial, in_cor, in_sag, rc_axial, rc_cor, rc_sag]

        for c, sl in enumerate(slices):
            ax = axes[r, c] if rows > 1 else axes[c]
            ax.imshow(sl.cpu(), cmap="gray")
            ax.axis("off")
            if r == 0:  ax.set_title(col_titles[c])

    plt.tight_layout()
    out_png = run_dir / "recon_examples.png"
    fig.savefig(out_png, dpi=300)
    log.info(f"Saved recon grid -> {out_png}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m big_brain.visualize.show_recon <run_dir> [n=4]")
        sys.exit(1)
    run   = Path(sys.argv[1]).expanduser().resolve()
    n     = int(sys.argv[2].split("=")[1]) if len(sys.argv) > 2 else 4
    main(run, n_show=n)
