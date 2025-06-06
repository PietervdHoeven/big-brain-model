from tqdm import tqdm
import torch

import logging
log = logging.getLogger(__name__)

def run_epoch(model, loader, criterion, optimiser=None, device='cuda'):
    """
    Train if optimiser is given, otherwise just validate.
    Returns the average loss for the epoch.
    """
    is_train = optimiser is not None
    model.train(mode=is_train)
    running_loss, n_batches = 0.0, 0

    log.info("Mode:", "Train" if is_train else "Eval")
    log.debug(f"Patients: {len(set(loader.dataset.patients))}, Sessions: {len(set(loader.dataset.sessions))}")

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for entry in tqdm(loader):
            x = entry["vol"].to(device, dtype=torch.float32, non_blocking=True)

            # Forward
            recon = model(x)
            loss = criterion(recon, x)

            if is_train:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

            running_loss += loss.item()
            n_batches += 1

    return running_loss / n_batches