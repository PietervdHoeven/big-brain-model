import numpy as np
import torch
from torch.utils.data import Subset, WeightedRandomSampler, Dataset
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
import torchio as tio
from typing import List, Tuple, Callable


def create_train_val_split(
        dataset,
        val_fraction: float = 0.1,
        seed: int = 42
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
    rng      = np.random.default_rng(seed)
    all_idx  = np.arange(n_total)
    rng.shuffle(all_idx)

    val_idx   = all_idx[:n_val] # first 10% is for validation
    train_idx = all_idx[n_val:] # Rest is for training

    return make_subset(dataset, train_idx), make_subset(dataset, val_idx)



def create_group_stratified_split(
        dataset,
        group_key: str = "patient",
        n_splits: int = 5, # yields 80% train / 10% val / 10% test
        seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create a group-stratified split of the dataset.
    The split is done based on the `group_key` which can be
    either "patient" or "session". The split is deterministic
    and uses the `seed` for reproducibility.

    Returns
    -------
    train_subset : torch.utils.data.Subset
    val_subset   : torch.utils.data.Subset
    test_subset  : torch.utils.data.Subset
    """

    # Get groups based on the group_key
    if group_key == "patient":
        groups = dataset.patients
    elif group_key == "session":
        groups = dataset.sessions
    else:
        raise ValueError("group_key must be either 'patient' or 'session'")
    
    # Ensure groups are numpy arrays
    labels = np.asarray(dataset.labels)
    groups = np.asarray(groups)

    # Check for ints or floats in labels (itns = classification, floats = regression)
    is_classification = np.issubdtype(labels.dtype, np.integer)

    if is_classification:
        # Create stratified group k-fold splitter
        skgf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # Split the dataset into train and rest sets
        train_idx, rest_idx = next(skgf.split(np.zeros(len(groups)), labels, groups))
        # Make a new StratifiedGroupKFold for the rest with 50 50 split
        skgf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
        # split the rest into val and test
        val_idx, test_idx = next(skgf.split(np.zeros(len(rest_idx)), labels[rest_idx], groups[rest_idx]))

    else:
        # create group shuffle split for regression
        gss = GroupShuffleSplit(n_splits=n_splits, random_state=seed)
        # Split the dataset into train rest sets
        train_idx, rest_idx = next(gss.split(np.zeros(len(groups)), groups=groups))
        # Make a new GroupShuffleSplit for the rest with 50 50 split
        gss = GroupShuffleSplit(n_splits=2, random_state=seed)
        # split the rest into val and test
        val_idx, test_idx = next(gss.split(np.zeros(len(rest_idx)), groups=groups[rest_idx]))


    return (
        make_subset(dataset, train_idx),
        make_subset(dataset, val_idx),
        make_subset(dataset, test_idx)
    )
    

def make_subset(dataset, idxs):
    sub = Subset(dataset, idxs.tolist())
    # attach split-specific metadata for later weighting
    sub.patients  = [dataset.patients[i]  for i in idxs]
    sub.sessions  = [dataset.sessions[i]  for i in idxs]
    if hasattr(dataset, 'bvals'):
        sub.bvals = [dataset.bvals[i] for i in idxs]
    if hasattr(dataset, 'lengths'):
        sub.lengths = [dataset.lengths[i] for i in idxs]
    if hasattr(dataset, 'labels'):
        sub.labels = [dataset.labels[i] for i in idxs]
    return sub


def make_ae_sampler(
    dataset, alpha=0.3
):
    """
    Build a sampler that up-weights rare shells but still lets b=1000
    be seen fairly often. We use w_shell[b] = 1 / (k_b ** alpha).
    """
    print(f"Building sampler for {len(dataset)} samples with alpha={alpha}...")

    # 1) Count how many sessions per patient
    patient2sessions = defaultdict(set)
    for p, s in zip(dataset.patients, dataset.sessions):
        patient2sessions[p].add(s)
    S_counts = {p: len(sset) for p, sset in patient2sessions.items()} # |S_p| for each patient p
    print(f"Number of patients: {len(S_counts)}")
    # e.g. {'sub-0001': 3, 'sub-0002': 2, ...} where 3 means 3 sessions for that patient

    # 2) Count how many volumes in each (patient,session)
    sess_keys   = list(zip(dataset.patients, dataset.sessions))
    V_cnt = Counter(sess_keys)    # |V_{p,s}| for each (patient, session) pair
    print(f"Number of sessions: {len(V_cnt)}")
    # e.g. {('sub-OAS30001', 'ses-d0757'): 26, ('sub-OAS30001', 'ses-d1234'): 120, ...}

    # 3) Count how many volumes in each shell
    shell_cnt = Counter(dataset.bvals)  # N_b for each b-value
    print(f"Number of shells: {len(shell_cnt)}")
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


def make_mdm_sampler(dataset):
    # 1) Count how many sessions per patient
    patient2sessions = defaultdict(set)
    for p, s in zip(dataset.patients, dataset.sessions):
        patient2sessions[p].add(s)
    S_counts = {p: len(sset) for p, sset in patient2sessions.items()} # |S_p| for each patient p

    # 2) Build weights
    weights = [1.0 / S_counts[p] for p in dataset.patients]

    # 3) Create Pytorch wighted sampler
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True
    )
    return sampler


def make_finetuner_sampler(dataset):
    # 1) Count the distribution of labels
    label_counts = Counter(dataset.labels)

    # 2) Build weights
    weights = [1.0 / label_counts[label] for label in dataset.labels]

    # 3) Create Pytorch weighted sampler
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


def collate_mdm(batch: List[Tuple[torch.Tensor, torch.Tensor, int]], p: float = 0.15):
    # Get metadata from batch
    B = len(batch)
    L_max = max([item[2] for item in batch])

    # allocate tensors
    Z_pad = torch.zeros((B, L_max+1, 512), dtype=torch.float32)         # L_max+1 because we need to make room for a [CLS] token at the first i
    G_pad = torch.zeros(B, L_max+1, 4, dtype=torch.float32)
    attn_mask = torch.zeros(B, L_max+1, dtype=torch.bool)
    mdm_labels = torch.zeros_like(Z_pad, dtype=torch.float32)
    mdm_mask = torch.zeros(B, L_max+1, dtype=torch.bool)

    # collate batch
    for b, (z, g, L) in enumerate(batch):
        # Prepend room for [cls] token (tensor is already 0)
        Z_pad[b,1:L+1] = z
        G_pad[b,1:L+1] = g
        attn_mask[b,0:L+1] = True

        # 80/10/10 mask (See BERT paper)
        n_mask = int(p * L)                         # Number of elements we need to mask according to p
        idx = torch.randperm(L)[:n_mask] + 1        # Take all the indices, shuffle them and take the first n_mask (+ 1 because we never want to pick the 0 index because it's for the [CLS] token)
        mdm_mask[b, idx] = True                     # Mask that indicates what indices are going to be masked in the Masked Diffusion Modelling (MDM)
        mdm_labels[b, idx] = Z_pad[b, idx]          # We have the true values in Z_pad. Store them into mdm_labels. These will be the target labels

        rand = torch.rand(n_mask)                       # For each index in sample a random value between [0,1)
        mask_idx = idx[rand < 0.80]                     # All indices with random sample below 0.8 get masked
        rand_idx = idx[(rand >= 0.80) & (rand < 0.90)]  # All indices between 0.8 and 0.9 get appointed a random z
                                                        # All indices above 0.9 are left as is
        if len(rand_idx):                               # [RANDOM]  # Take len(rand_idx) random zs from Z_pad from indices 1:L+1 and replace them with the orignal zs
            Z_pad[b, rand_idx] = Z_pad[b, torch.randint(1, L+1, (len(rand_idx),))]
        Z_pad[b, mask_idx] = 0                          # [MASK]    # Replace the masked tokens with empty z's [0, ..., 0].shape = 512

    return Z_pad, G_pad, attn_mask, mdm_labels, mdm_mask


def collate_finetuner(batch: List[Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]]):
    """
    Collate function for finetuning datasets.
    It pads the z and g tensors to the maximum length in the batch.
    """
    B = len(batch)
    L_max = max([item[2] for item in batch])

    # allocate tensors
    Z_pad = torch.zeros((B, L_max+1, 512), dtype=torch.float32)     # [B, L_max+1, 512]
    G_pad = torch.zeros((B, L_max+1, 4), dtype=torch.float32)       # [B, L_max+1, 4]
    attn_mask = torch.zeros(B, L_max+1, dtype=torch.bool)           # [B, L_max+1]
    ys = []

    # collate batch
    for b, (z, g, L, y) in enumerate(batch):
        # Prepend room for [cls] token (tensor is already 0)
        Z_pad[b,1:L+1] = z
        G_pad[b,1:L+1] = g
        attn_mask[b,0:L+1] = True
        ys.append(y)

    Y = torch.stack(ys)
    return Z_pad, G_pad, attn_mask, Y
    


MAPPERS: dict[str, dict | None] = {
    "gender":     {"male": 0, "female": 1},
    "handedness": {"right": 0, "left": 1, "both": 2},

    # CDR variants
    "bin_cdr": {0.0: 0, 0.5: 1, 1.0: 1, 2.0: 1},
    "tri_cdr": {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 2},
    "ord_cdr": {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3},

    # Identity / regression
    "age": None,
}
