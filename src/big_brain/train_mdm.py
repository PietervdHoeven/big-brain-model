# src/big_brain/training/train_mlm.py
"""
Masked-latent pre-training loop for DWIBert
==========================================

Uses:
  • big_brain.data.datamodules.TFDataModule
  • big_brain.models.transformer.DWIBert   (with true weight tying)
  • big_brain.schedulers.get_warmup_cosine
  • big_brain.training.stopping.EarlyStopper

"""
from pathlib import Path
import argparse, json, time, torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import logging

# ------------------------------------------------------------------ repo imports
from big_brain.data.datamodules import TFDataModule
from big_brain.models.transformer import DWIBert
from big_brain.training.schedulers import get_warmup_cosine
from big_brain.training.stopping import EarlyStopper

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

# ------------------------------------------------------------------ CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="root with *latent.npz")
    p.add_argument("--out_dir",   required=True, help="where checkpoints go")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--val_split",  type=float, default=0.05)
    p.add_argument("--patience",   type=int, default=5,
                   help="early-stop patience (epochs without dev improvement)")
    p.add_argument("--lr",         type=float, default=3e-4)
    return p.parse_args()

# ------------------------------------------------------------------ util
def masked_mae(pred, labels, mask):
    return torch.nn.functional.smooth_l1_loss(
        pred[mask], labels[mask], reduction="mean")

# ------------------------------------------------------------------ main
def main():
    cfg = parse_args()
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data
    dm = TFDataModule(cfg.data_root,
                      batch_size=cfg.batch_size,
                      val_split=cfg.val_split)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    # 2) Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = DWIBert().to(device)

    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=1e-2)
    scheduler = get_warmup_cosine(optimiser,
                                  warmup_epochs=max(1, cfg.epochs // 10),
                                  max_epochs=cfg.epochs)
    scaler     = GradScaler()
    stopper    = EarlyStopper(patience=cfg.patience, mode="min")

    history = {"epoch": [], "train_loss": [], "val_loss": []}

    # 3) Loop
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train(); running = 0.0; n_steps = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            Z, G, attn, labels, mask = [b.to(device) for b in batch]

            with autocast(device_type=device, enabled=True):
                pred  = model(Z, G, attn)
                loss  = masked_mae(pred, labels, mask)

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser); scaler.update()

            running += loss.item(); n_steps += 1
        scheduler.step()

        train_loss = running / n_steps
        # --------------- validation ------------------------------
        model.eval(); val_running = 0.0; n_val_steps = 0
        with torch.no_grad(), autocast(device_type=device, enabled=True):
            for Z,G,attn,labels,mask in tqdm(val_loader, desc="val"):
                Z,G,attn,labels,mask = [t.to(device) for t in
                                        (Z,G,attn,labels,mask)]
                val_running += masked_mae(model(Z,G,attn), labels, mask).item()
                n_val_steps += 1
        val_loss = val_running / n_val_steps
        dt = time.time() - t0

        log.info(f"Ep {epoch:02d} | train {train_loss:.4f} | "
                 f"val {val_loss:.4f} | {dt:.1f}s")

        # history & checkpoint
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimiser.state_dict()},
                   out_dir / "last.ckpt")

        if stopper.step(val_loss):
            torch.save(model.state_dict(), out_dir / "best_encoder.pt")
            log.info("  ✓ new best model saved.")

        if stopper.should_stop:
            log.info("Early stopping triggered.")
            break

    # persist history
    with (out_dir / "history.json").open("w") as fp:
        json.dump(history, fp, indent=2)
    log.info(f"Training complete. History saved to {out_dir / 'history.json'}")

if __name__ == "__main__":
    main()
