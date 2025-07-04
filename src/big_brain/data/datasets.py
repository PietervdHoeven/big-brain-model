# src/big_brain/data.py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class AEVolumes(Dataset):
    def __init__(self, cache_root: str):
        # 1) Find all cached gradient files
        self.files = sorted(Path(cache_root).rglob("*grad*.npz"))

        # 2) Preload metadata for sampling weights (if you ever need it)
        self.patients = []
        self.sessions = []
        self.bvals    = []
        for f in self.files:
            data = np.load(f)
            self.patients.append(data["patient"].item())  
            self.sessions.append(data["session"].item())  
            self.bvals.append(float(data["bval"].item()))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        # 2) Volume: [1, X, Y, Z]
        vol = torch.from_numpy(data["vol_data"]).unsqueeze(0).float()

        # 3) Acquisition metadata
        bval = torch.tensor(float(data["bval"].item()), dtype=torch.float32)
        bvec = torch.from_numpy(data["bvec"].astype(np.float32))

        # 4) Spatial metadata
        affine = torch.from_numpy(data["affine"].astype(np.float32))  # shape: [4,4]

        # 5) Patient/session tags
        patient = data["patient"].item()  # string
        session = data["session"].item()  # string

        return {
            "vol":     vol,
            "bval":    bval,
            "bvec":    bvec,
            "affine":  affine,
            "patient": patient,
            "session": session,
        }
    

class TFLatents(Dataset):
    def __init__(self, data_root: str = "data"):
        # Find all preprocessed and packed DWI sessions' latents
        self.files = sorted(Path(data_root).rglob("*latent.npz"))

        # Preload the files to get meta data for weighted sampling
        self.patients = []
        self.sessions = []
        self.lengths  = []
        for f in self.files:
            data = np.load(f)
            self.patients.append(data["patient"].item())  
            self.sessions.append(data["session"].item())
            self.lengths.append(data["g"].shape[0])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # all data in npz file
        data = np.load(self.files[idx])
        # latents (512)
        z = torch.from_numpy(data['z'])
        # gradient info (bval, bval, bval, bvec)
        g = torch.from_numpy(data['g'])
        # number of volumes in one session
        length = self.lengths[idx]
        return z, g, length