# src/big_brain/schedulers.py

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ReduceLROnPlateau

def get_warmup_cosine(optimiser, warmup_epochs = 5, max_epochs = 200, eta_min = 1e-6):
    linear = LinearLR(optimiser, 0.01, 1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimiser, T_max=max_epochs-warmup_epochs, eta_min=eta_min)
    return SequentialLR(optimiser, [linear, cosine], milestones=[warmup_epochs])

def get_plateau(optimiser, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=False):
    return ReduceLROnPlateau(optimiser, mode=mode, factor=factor, patience=patience, verbose=verbose, min_lr=min_lr)