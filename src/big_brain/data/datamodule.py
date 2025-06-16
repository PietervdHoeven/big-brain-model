# ae_data/datamodule.py
from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from big_brain.data.dataset import AEVolumes
from big_brain.data.utils   import create_val_test_split, make_balanced_sampler

class AEDataModule:
    """
    Hydrate-friendly wrapper around AEVolumes.

    - Splits into train / val / test using your helper.
    - Optionally builds WeightedRandomSamplers with your make_balanced_sampler.
    - Produces PyTorch DataLoaders that the training loop (or Lightning Trainer)
      can consume directly.
    """

    def __init__(
        self,
        cache_root: str,                        # path to the dir containing all cached and normalised volumes .npz files
        batch_size: int = 32,                   # batch size for training / validation / testing
        val_split: float = 0.10,                # fraction of the dataset to use for validation
        test_split: float = 0.10,               # fraction of the dataset to use for testing
        num_workers: int = 4,                   # number of workers for DataLoader (Determine for the slurm script)
        pin_memory: bool = True,                # pin_memory=True keep batch in (pinned) RAM so that when you do batch.to("cuda"), this usually speeds up transfers. If you’re only on CPU, it has no effect.
        use_weighted_sampler: bool = True,      # whether to use WeightedRandomSampler for training / validation / testing
        alpha: float = 0.3,                     # exponent in make_balanced_sampler, alpha=0 means no upweighting of rare shells
        sample_fraction: float = 0.0,           # 0.05 → keep 5 % of files | 0.0 → keep all files (default, no sampling)
    ):
        self.cache_root = cache_root
        self.batch_size = batch_size
        self.val_split  = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.pin_memory  = pin_memory
        self.use_weighted_sampler = use_weighted_sampler
        self.alpha = alpha
        self.sample_fraction = sample_fraction

        # will be filled by `setup()`
        self.train_dataset: Optional[Subset] = None
        self.val_dataset  : Optional[Subset] = None
        self.test_dataset : Optional[Subset] = None

        self.train_sampler: Optional[WeightedRandomSampler] = None
        self.val_sampler  : Optional[WeightedRandomSampler] = None
        self.test_sampler : Optional[WeightedRandomSampler] = None

    def _three_way_split(self, full_ds: AEVolumes) -> Tuple[Subset, Subset, Subset]:
        """
        Two successive calls to your create_val_test_split to obtain
        train / val / test subsets.
        """
        train_ds, valtest_ds = create_val_test_split(
            full_ds,
            val_fraction=self.val_split + self.test_split,
        )
        val_ratio_inside = self.val_split / (self.val_split + self.test_split)

        val_ds, test_ds = create_val_test_split(
            valtest_ds,
            val_fraction=val_ratio_inside,
        )
        return train_ds, val_ds, test_ds

    # public API expected by training loop (or Lightning)
    def setup(self):
        full_ds = AEVolumes(self.cache_root)

        if self.sample_fraction != 0.0:
            _, full_ds = create_val_test_split(
                full_ds,
                val_fraction=self.sample_fraction,
            )

        self.train_dataset, self.val_dataset, self.test_dataset = self._three_way_split(full_ds)

        if self.use_weighted_sampler:
            self.train_sampler = make_balanced_sampler(self.train_dataset, alpha=self.alpha)
            self.val_sampler   = make_balanced_sampler(self.val_dataset, alpha=self.alpha)
            self.test_sampler  = make_balanced_sampler(self.test_dataset, alpha=self.alpha)
        else:
            self.train_sampler = self.val_sampler = self.test_sampler = None

    def _dl(self, dataset, sampler, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(shuffle and sampler is None),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    # Lightning - style names, but work fine in plain PyTorch loops too
    def train_dataloader(self):
        return self._dl(self.train_dataset, self.train_sampler, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.val_dataset, self.val_sampler)

    def test_dataloader(self):
        return self._dl(self.test_dataset, self.test_sampler)
