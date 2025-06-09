# test_load_volume.py

import hydra
from omegaconf import OmegaConf
import torch

from big_brain.data.dataset     import AEVolumes
from big_brain.data.datamodule  import AEDataModule

def smoke_test_dataset(cache_root: str):
    # 1) Instantiate the low-level Dataset directly
    ds = AEVolumes(cache_root)
    print(f"Found {len(ds)} files in {cache_root!r}")
    
    # 2) Grab one sample and inspect its contents
    sample = ds[5]
    print("Single-sample keys:", list(sample.keys()))
    print("  vol.shape:", sample["vol"].shape)           # should be [1, X, Y, Z]
    print("  bval:", sample["bval"].item())
    # print("  bvec.shape:", sample["bvec"].shape)         # should be length 3 (or whatever)
    print("  affine.shape:", sample["affine"].shape)     # should be [4, 4]
    print("  patient:", sample["patient"])
    print("  session:", sample["session"])

def smoke_test_datamodule(cache_root: str):
    # 1) Instantiate the DataModule via its constructor
    dm = AEDataModule(
        cache_root=cache_root,
        batch_size=8,
        val_split=0.1,
        test_split=0.1,
        num_workers=0,
        pin_memory=False,
        use_weighted_sampler=False,
        alpha=0.0,
        seed=42,
    )
    
    # 2) Call setup() to split & build samplers
    dm.setup()
    
    # 3) Get one batch from train_dataloader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print("Batch keys:", list(batch.keys()))
    print("  batch['vol'].shape:", batch["vol"].shape)  # [batch_size, 1, X, Y, Z]
    print("  batch['bval'].shape:", batch["bval"].shape)
    # print("  batch['bvec'].shape:", batch["bvec"].shape)
    print("  batch['affine'].shape:", batch["affine"].shape)
    print("  batch['patient'][5]:", batch["patient"][5])
    print("  batch['session'][5]:", batch["session"][5])

if __name__ == "__main__":
    # Replace this path with the folder where your *.npz lives
    cache_root = "/home/spieterman/dev/big-brain-model/data/encoder"
    
    print("=== Testing AEVolumes Dataset ===")
    smoke_test_dataset(cache_root)
    
    print("\n=== Testing AEDataModule DataLoader ===")
    smoke_test_datamodule(cache_root)
