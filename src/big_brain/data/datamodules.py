# ae_data/datamodule.py
from pathlib import Path
from typing import Optional, Tuple, Callable, Any

from torch.utils.data import DataLoader, Subset, Dataset, WeightedRandomSampler
import pytorch_lightning as pl

from big_brain.data.datasets import AEVolumes, TFLatents
from big_brain.data.utils   import create_train_val_split, make_ae_sampler, make_tf_sampler, make_probe_sampler, collate_mdm, collate_probe

import logging
log = logging.getLogger(__name__)

class AEDataModule(pl.LightningDataModule):
    """
    Hydra-friendly wrapper around AEVolumes.

    - Splits into train / val / test using your helper.
    - Optionally builds WeightedRandomSamplers with your make_ae_sampler.
    - Produces PyTorch DataLoaders that the training loop (or Lightning Trainer)
      can consume directly.
    """

    def __init__(
        self,
        data_dir: str,                          # path to the dir containing all cached and normalised volumes .npz files
        batch_size: int = 8,                    # batch size for training / validation / testing
        val_split: float = 0.10,                # fraction of the dataset to use for validation
        test_split: float = 0.10,               # fraction of the dataset to use for testing
        num_workers: int = 8,                   # number of workers for DataLoader (Determine for the slurm script)
        pin_memory: bool = True,                # pin_memory=True keep batch in (pinned) RAM so that when you do batch.to("cuda"), this usually speeds up transfers. If you’re only on CPU, it has no effect.
        use_sampler: bool = True,               # whether to use WeightedRandomSampler for training
        alpha: float = 0.3,                     # exponent in make_ae_sampler, alpha=0 means no upweighting of rare shells
        sample_fraction: float = 0.0,           # 0.05 → keep 5 % of files | 0.0 → keep all files (default, no sampling)
        enable_aug: bool = False,               # whether to apply augmentation transforms to the training set
        seed: int = 42                          # seed for reproducibility
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split  = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.pin_memory  = pin_memory
        self.use_sampler = use_sampler
        self.alpha = alpha
        self.sample_fraction = sample_fraction
        self.enable_aug = enable_aug
        self.seed = seed

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

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate", "test"):

            full_dataset = AEVolumes(self.data_dir)
            print(f"Loaded {len(full_dataset)} samples from {self.data_dir}")

            if self.sample_fraction != 0.0:
                _, full_dataset = create_train_val_split(
                    full_dataset,
                    val_fraction=self.sample_fraction,
                    seed=self.seed
                )

            self.train_dataset, self.val_dataset, self.test_dataset = self._three_way_split(full_dataset)
            print(f"Split into {len(self.train_dataset)} train, "
                  f"{len(self.val_dataset)} val, "
                  f"{len(self.test_dataset)} test samples.")

        if self.use_sampler:
            self.sampler = make_ae_sampler(self.train_dataset, alpha=self.alpha)
            print(f"Created sampler with {len(self.sampler)} samples, alpha={self.alpha}")
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
        dl = self._dl(self.train_dataset, self.sampler, shuffle=True)
        print(f"Created train DataLoader with {len(dl.dataset)} samples, "
              f"batch size {self.batch_size}, "
              f"{self.num_workers} workers, "
              f"pin_memory={self.pin_memory}")
        return self._dl(self.train_dataset, self.sampler, shuffle=True)

    def val_dataloader(self):
        print(f"Created val DataLoader with {len(self.val_dataset)} samples, "
              f"batch size {self.batch_size}, "
              f"{self.num_workers} workers, "
              f"pin_memory={self.pin_memory}")
        return self._dl(self.val_dataset)

    def test_dataloader(self):
        print(f"Created test DataLoader with {len(self.test_dataset)} samples, "
              f"batch size {self.batch_size}, "
              f"{self.num_workers} workers, "
              f"pin_memory={self.pin_memory}")
        return self._dl(self.test_dataset)


class TFDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir,                                                       # directory where the data is saved
            batch_size: int = 8,                                            # batch size for training / validation
            val_split: float = 0.05,                                        # fraction of the dataset to use for validation
            num_workers: int = 8,                                           # number of workers for DataLoader (Determine for the slurm script)
            pin_memory: bool = True,                                        # pin_memory=True keep batch in (pinned) RAM so that when you do batch.to("cuda"), this usually speeds up transfers. If you’re only on CPU, it has no effect.
            sampler_fn: Callable[[Dataset], WeightedRandomSampler] = None,  # Function that returns a sampler for the training dataset
            collate_fn: Callable[[list[Any]], Any] = collate_mdm,           # Function to collate the batch
            sample_fraction: float = 0.0,                                   # 0.05 → keep 5 % of files | 0.0 → keep all files (default, no sampling)
            seed: int = 42                                                  # seed for reproducibility
            ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sampler_fn = sampler_fn
        self.collate_fn = collate_fn
        self.sample_fraction = sample_fraction
        self.seed = seed

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Called on every process (GPU) and for every stage.
        We split the same full dataset into train/val once.
        """
        if stage in (None, "fit", "validate"):
            full_dataset = TFLatents(self.data_dir)

            if self.sample_fraction != 0.0:
                _, full_dataset = create_train_val_split(
                    full_dataset,
                    val_fraction=self.sample_fraction,
                    seed=self.seed
                )
            
            self.train_dataset, self.val_dataset = create_train_val_split(
                full_dataset,
                val_fraction=self.val_split,
                seed=self.seed
            )

        if self.sampler_fn is not None:
            self.sampler = self.sampler_fn(dataset=self.train_dataset)
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
            collate_fn=self.collate_fn
        )

    def train_dataloader(self):
        return self._dl(self.train_dataset, self.sampler, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.val_dataset)






        