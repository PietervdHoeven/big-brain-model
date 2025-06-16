import numpy as np
import torch
from torch.utils.data import Subset, WeightedRandomSampler
from collections import defaultdict, Counter
import torchio as tio

def create_val_test_split(
        dataset,
        val_fraction: float = 0.1
):
    """
    Randomly split `dataset` into train / val subsets and
    attach `.patients`  and `.sessions` lists that correspond
    only to the samples inside each subset.

    Returns
    -------
    train_subset : torch.utils.data.Subset
    val_subset   : torch.utils.data.Subset
    """
    n_total = len(dataset)
    n_val   = int(np.floor(val_fraction * n_total))

    # deterministic shuffle
    rng      = np.random.default_rng()
    all_idx  = np.arange(n_total)
    rng.shuffle(all_idx)

    val_idx   = all_idx[:n_val]
    train_idx = all_idx[n_val:]

    def make_subset(idxs):
        sub = Subset(dataset, idxs.tolist())
        # attach split-specific metadata for later weighting
        sub.patients  = [dataset.patients[i]  for i in idxs]
        sub.sessions  = [dataset.sessions[i]  for i in idxs]
        sub.bvals     = [dataset.bvals[i]     for i in idxs]

        return sub

    return make_subset(train_idx), make_subset(val_idx)


def make_balanced_sampler(
    dataset, alpha=0.3
):
    """
    Build a sampler that up-weights rare shells but still lets b=1000
    be seen fairly often. We use w_shell[b] = 1 / (k_b ** alpha).
    """

    # 1) Count how many sessions per patient
    patient2sessions = defaultdict(set)
    for p, s in zip(dataset.patients, dataset.sessions):
        patient2sessions[p].add(s)
    S_counts = {p: len(sset) for p, sset in patient2sessions.items()} # |S_p| for each patient p
    # e.g. {'sub-0001': 3, 'sub-0002': 2, ...} where 3 means 3 sessions for that patient

    # 2) Count how many volumes in each (patient,session)
    sess_keys   = list(zip(dataset.patients, dataset.sessions))
    V_cnt = Counter(sess_keys)    # |V_{p,s}| for each (patient, session) pair
    # e.g. {('sub-OAS30001', 'ses-d0757'): 26, ('sub-OAS30001', 'ses-d1234'): 120, ...}

    # 3) Count how many volumes in each shell
    shell_cnt = Counter(dataset.bvals)  # N_b for each b-value
    # e.g. {0: 300, 500: 320, 1000: 4500, ...}

    # 4) Build the final weights for each sample
    weights = []
    for (p, s, b) in zip(dataset.patients, dataset.sessions, dataset.bvals):
        w  = 1.0 / S_counts[p]                  # patient factor
        w *= 1.0 / V_cnt[(p, s)]                # session factor
        w *= 1.0 / (shell_cnt[b] ** alpha)      # tempered shell factor
        w *= 10                                 # boost the weights to make them more pronounced
        weights.append(w)

    # 5) Create the PyTorch sampler
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True
    )
    return sampler


def make_augmentation_transforms():
    """Return the standard 3-D augmentation Compose."""
    return tio.Compose([
        # geometric
        tio.RandomAffine(scales=0.5, degrees=50, translation=50),
        tio.RandomElasticDeformation(num_control_points=7,
                                     max_displacement=3,
                                     locked_borders=2),
        tio.RandomFlip(axes=('LR',), flip_probability=0.5),
        # intensity
        tio.RandomGamma(log_gamma=(-0.1, 0.1)),
        tio.RandomNoise(std=0.01),
    ])