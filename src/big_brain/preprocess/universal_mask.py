import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# --------------------------------------------------
# settings
# --------------------------------------------------
data_dir   = Path("/home/spieterman/dev/big-brain-model/data/preproc")
files      = sorted(data_dir.rglob("*_dwi_allruns.nii.gz"))
cache_path = Path("data/global_mask.npy")
flag_file  = Path("data/flagged_scans.txt")
thr_vox    = 20                          # max allowed shift (voxels)

# --------------------------------------------------
# helpers
# --------------------------------------------------
def load_mask(fn):
    vol = nib.load(fn, mmap=True).get_fdata(dtype=np.float32)
    vol = vol[..., 0]                   # first (b=0) volume
    return vol > 0                       # boolean mask

def center_of_mass(mask):
    return np.array(np.nonzero(mask)).mean(axis=1)      # get the mean position of non-zero voxels

def aligned(ref_mask, new_mask, thr=thr_vox):
    return np.all(np.abs(center_of_mass(ref_mask) - center_of_mass(new_mask)) <= thr)

# --------------------------------------------------
# main
# --------------------------------------------------
if cache_path.exists():
    print("Global mask already exists → loading")
    global_mask = np.load(cache_path)
    flagged     = []
else:
    print("Building global mask …")
    flagged     = []
    global_mask = None

    for file in tqdm(files, desc="scans"):
        mask = load_mask(file)

        # first volume becomes the reference
        if global_mask is None:
            global_mask = mask.copy()
            continue

        if aligned(global_mask, mask):
            global_mask |= mask            # in-place OR
        else:
            flagged.append(str(file))

    # save results
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(cache_path, global_mask.astype(np.uint8))
    if flagged:
        flag_file.write_text("\n".join(flagged))
        print(f"Flagged {len(flagged)} scans → {flag_file}")
    print(f"Union mask saved to {cache_path}")

# --------------------------------------------------
# quick visual check (axial mid-slice)
# --------------------------------------------------
z_mid = global_mask.shape[0] // 2          # axial mid-slice
plt.imshow(global_mask[z_mid], cmap="gray")
plt.axis("off")
plt.title(f"Union mask  (z={z_mid})")
plt.savefig("data/global_mask_mid_slice.png", bbox_inches="tight")
plt.close()
print("Preview saved to data/global_mask_mid_slice.png")
