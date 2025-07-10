# src/big_brain/data.py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

from big_brain.data.utils import MAPPERS


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
    

class TFLabeledLatents(TFLatents):
    """
    Wrapper around TFLatents that adds labels for classification/regression.
    """
    def __init__(
            self,
            data_root: str = "data",                                # root directory where the latents are stored
            labels_path: str = "data/labels/labels.parquet",        # dictionary mapping (patient, session) to label
            column: str = "cdr",                                    # patient, session, cdr, gender, handedness, age
            task: str = "bin_cdr"                                   # task to perform: bin_cdr, tri_cdr, ord_cdr, gender, handedness, age
    ):
        super().__init__(data_root)                     # Initialize the base class. So we get the files, patients, sessions, and lengths.
        self.labels_df = pd.read_parquet(labels_path)   # Load the labels DataFrame from a parquet file.
        self.task = task                                 # The task to perform, which determines how labels are processed.
        self.mapper = MAPPERS[task]                     # A mapper function or dict to transform labels if needed

        # get the right column for the task
        if task in ["bin_cdr", "tri_cdr", "ord_cdr"]:
            column = "cdr"                                  # For CDR tasks, we use the 'cdr' column
        else:
            column = task                                   # For other tasks, we use the task name as the column name

        # Extract labels for each patient-session pair from the DataFrame
        self.labels = []                                # List to store labels for each (patient, session) pair.
        for p, s in zip(self.patients, self.sessions):
            label = self.labels_df.loc[(self.labels_df['patient'] == p) & (self.labels_df['session'] == s), column].iloc[0]
            if self.mapper is not None:
                label = self.mapper[label]
            self.labels.append(label)                   # We build a parallel list of labels for each patient and session.
        
    
    def __getitem__(self, idx):
        # Get the latent vector and gradient info from the base class
        z, g, length = super().__getitem__(idx)

        # Get the label for the current index
        label = self.labels[idx]
        if isinstance(label, str): label = label.strip().lower()  # Normalize string labels
        if self.task in "age":
            y = torch.tensor(label, dtype=torch.float32)  # Convert to tensor (typically float for regression)
        else:
            y = torch.tensor(label, dtype=torch.long)  # Convert to tensor (typically long for classification)
        
        return z, g, length, y
