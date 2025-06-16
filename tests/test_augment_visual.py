import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from big_brain.data.datamodule import AEDataModule


# ──────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────
def mid_slices(vol: torch.Tensor):
    """Return axial, coronal, sagittal mid-slices (H×W arrays)."""
    v = vol.squeeze(0)          # (D,H,W)
    D, H, W = v.shape
    return (
        v[D // 2, :, :],        # axial
        v[:, H // 2, :],        # coronal
        v[:, :, W // 2],        # sagittal
    )


def make_loader(dataset, bs, nw):
    """Non-shuffling loader so sample i has both orig & aug for the same case."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )


# ──────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────
def main(args):
    # 1) build the DM with augmentation enabled (wrapper fills "orig")
    dm = AEDataModule(
        cache_root=Path(args.cache_root),
        enable_aug=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # …other init args…
    )
    dm.setup()

    loader = make_loader(dm.train_dataset, args.batch_size, args.num_workers)

    for item in loader:
        print(f"Batch size: {item['vol'].shape[0]}")
        print(item.items())
        print(item)
        break

    # 2) gather exactly n_show samples (could span multiple mini-batches)
    orig_list, aug_list, pat_list, ses_list = [], [], [], []
    for batch in loader:
        orig_list.append(batch["orig"])
        aug_list.append(batch["vol"])
        pat_list.extend(batch["patient"])
        ses_list.extend(batch["session"])
        if len(pat_list) >= args.n_show:
            break

    # cat & trim to n_show
    vols_orig = torch.cat(orig_list, dim=0)[: args.n_show]
    vols_aug  = torch.cat(aug_list,  dim=0)[: args.n_show]
    patients  = pat_list[: args.n_show]
    sessions  = ses_list[: args.n_show]

    # 3) plot
    n = vols_orig.shape[0]
    fig, axes = plt.subplots(n, 6, figsize=(12, 2 * n), squeeze=False)
    fig.suptitle("Original vs Augmented — Mid-Slices", fontsize=16)

    for i in range(n):
        # label each row
        label = f"{patients[i]}\n{sessions[i]}"
        axes[i, 0].set_ylabel(label, rotation=0, labelpad=45,
                              fontsize=10, va="center")

        # get slices
        slices_o = mid_slices(vols_orig[i])
        slices_a = mid_slices(vols_aug[i])

        # original → cols 0-2
        for j, sl in enumerate(slices_o):
            ax = axes[i, j]
            ax.imshow(sl.cpu(), cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_title(["Axial", "Coronal", "Sagittal"][j], fontsize=10)

        # augmented → cols 3-5
        for j, sl in enumerate(slices_a):
            ax = axes[i, j + 3]
            ax.imshow(sl.cpu(), cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_title(
                    [f"{t} (aug)" for t in ("Axial", "Coronal", "Sagittal")][j],
                    fontsize=10,
                )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "augment_viz_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison preview → {out_path}")


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root",  required=True, help="path to *.npz cache")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_show",      type=int, default=8,
                   help="rows to visualise")
    main(p.parse_args())
