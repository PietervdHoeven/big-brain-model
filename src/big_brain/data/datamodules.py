# ae_data/datamodule.py
from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from big_brain.data.datasets import AEVolumes, TFLatents
from big_brain.data.utils   import create_train_val_split, make_ae_sampler, make_tf_sampler, collate_batch

import logging
log = logging.getLogger(__name__)

class AEDataModule:
    """
    Hydra-friendly wrapper around AEVolumes.

    - Splits into train / val / test using your helper.
    - Optionally builds WeightedRandomSamplers with your make_ae_sampler.
    - Produces PyTorch DataLoaders that the training loop (or Lightning Trainer)
      can consume directly.
    """

    def __init__(
        self,
        cache_root: str,                        # path to the dir containing all cached and normalised volumes .npz files
        batch_size: int = 32,                   # batch size for training / validation / testing
        val_split: float = 0.10,                # fraction of the dataset to use for validation
        test_split: float = 0.10,               # fraction of the dataset to use for testing
        num_workers: int = 8,                   # number of workers for DataLoader (Determine for the slurm script)
        pin_memory: bool = True,                # pin_memory=True keep batch in (pinned) RAM so that when you do batch.to("cuda"), this usually speeds up transfers. If you’re only on CPU, it has no effect.
        use_sampler: bool = True,               # whether to use WeightedRandomSampler for training
        alpha: float = 0.3,                     # exponent in make_ae_sampler, alpha=0 means no upweighting of rare shells
        sample_fraction: float = 0.0,           # 0.05 → keep 5 % of files | 0.0 → keep all files (default, no sampling)
        enable_aug: bool = True,                # whether to apply augmentation transforms to the training set
    ):
        self.cache_root = cache_root
        self.batch_size = batch_size
        self.val_split  = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.pin_memory  = pin_memory
        self.use_sampler = use_sampler
        self.alpha = alpha
        self.sample_fraction = sample_fraction
        self.enable_aug = enable_aug

        # will be filled by `setup()`
        self.train_dataset: Optional[Subset] = None
        self.val_dataset  : Optional[Subset] = None
        self.test_dataset : Optional[Subset] = None

        self.train_sampler: Optional[WeightedRandomSampler] = None
        self.val_sampler  : Optional[WeightedRandomSampler] = None
        self.test_sampler : Optional[WeightedRandomSampler] = None

    def _three_way_split(self, full_ds: AEVolumes) -> Tuple[Subset, Subset, Subset]:
        """
        Two successive calls to your create_train_val_split to obtain
        train / val / test subsets.
        """
        train_ds, valtest_ds = create_train_val_split(
            full_ds,
            val_fraction=self.val_split + self.test_split,
        )
        val_ratio_inside = self.val_split / (self.val_split + self.test_split)

        val_ds, test_ds = create_train_val_split(
            valtest_ds,
            val_fraction=val_ratio_inside,
        )
        return train_ds, val_ds, test_ds

    # public API expected by training loop (or Lightning)
    def setup(self):
        log.info("Setting up AEDataModule...")
        full_ds = AEVolumes(self.cache_root)

        if self.sample_fraction != 0.0:
            _, full_ds = create_train_val_split(
                full_ds,
                val_fraction=self.sample_fraction,
            )

        self.train_dataset, self.val_dataset, self.test_dataset = self._three_way_split(full_ds)

        # Optionally apply transforms to the datasets
        # if self.enable_aug:
        #     transform = make_augmentation_transforms()
        #     self.train_dataset = WithTransforms(self.train_dataset, transform)

        if self.use_sampler:
            self.sampler = make_ae_sampler(self.train_dataset, alpha=self.alpha)
        else:
            self.sampler = None

    def _dl(self, dataset, sampler=None, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(shuffle and sampler is None),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self._dl(self.train_dataset, self.train_sampler, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.val_dataset)

    def test_dataloader(self):
        return self._dl(self.test_dataset)


class TFDataModule:
    def __init__(
            self,
            data_root,                              # directory where the data is saved
            batch_size: int = 8,                    # batch size for training / validation
            val_split: float = 0.05,                # fraction of the dataset to use for validation
            num_workers: int = 8,                   # number of workers for DataLoader (Determine for the slurm script)
            pin_memory: bool = True,                # pin_memory=True keep batch in (pinned) RAM so that when you do batch.to("cuda"), this usually speeds up transfers. If you’re only on CPU, it has no effect.
            use_sampler: bool = True,               # whether to use WeightedRandomSampler for training
            sample_fraction: float = 0.0,           # 0.05 → keep 5 % of files | 0.0 → keep all files (default, no sampling)
            ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_sampler = use_sampler
        self.sample_fraction = sample_fraction

        # will be filled by setup()
        self.train_dataset: Optional[Subset] = None
        self.val_dataset  : Optional[Subset] = None

        self.sampler: Optional[WeightedRandomSampler] = None

    def setup(self):
        log.info("Setting up TFDataModule...")

        dataset = TFLatents(data_root=self.data_root)

        if self.sample_fraction > 0:
            _, dataset = create_train_val_split(
                dataset,
                val_fraction=self.sample_fraction,
            )
        
        self.train_dataset, self.val_dataset = create_train_val_split(dataset=dataset, val_fraction=self.val_split)

        if self.use_sampler:
            self.sampler = make_tf_sampler(dataset=self.train_dataset)
        else:
            self.sampler = None

    def _dl(self, dataset, sampler=None, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(shuffle and sampler is None),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_batch
        )

    def train_dataloader(self):
        return self._dl(self.train_dataset, self.sampler, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.val_dataset)





        