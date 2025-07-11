import numpy as np, nibabel as nib
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- settings ----------------
data_dir   = Path("/home/spieterman/dev/big-brain-model/data/preproc")
files      = sorted(data_dir.rglob("*_dwi_allruns.nii.gz"))
cache_path = Path("data/global_mask.npy")
flag_file  = Path("data/flagged_scans.txt")
thr_vox    = 20                               # tolerance in voxels
n_jobs     = -1                               # use all cores

# ---------------- helpers ------------------
def load_mask_and_com(fn):
    vol = nib.load(fn, mmap=True).get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    mask = vol > 0
    if not mask.any():
        return None, None, str(fn)                  # empty scan flagged
    com = np.asarray(np.nonzero(mask)).mean(axis=1)
    return (mask, com, str(fn))

def is_aligned(com_ref, com_new, thr=thr_vox):
    return np.all(np.abs(com_ref - com_new) <= thr)

# ---------------- main ---------------------
if cache_path.exists():
    print("Global mask already exists → loading")
    global_mask = np.load(cache_path)
    flagged     = []
else:
    assert files, "No NIfTI files found!"
    print("Building union mask …")

    # --- first scan sets the reference --------------------
    first_mask, first_com, _ = load_mask_and_com(files[0])
    global_mask = first_mask.copy()
    ref_com     = first_com
    to_process  = files[1:]

    # --- parallel loop ------------------------------------
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(load_mask_and_com)(fn) for fn in tqdm(to_process, desc="scans")
    )

    flagged = []
    for mask, com, fn in results:
        if mask is None:               # already flagged as empty
            flagged.append(fn)
        elif is_aligned(ref_com, com):
            global_mask |= mask        # in-place OR
        else:
            flagged.append(fn)

    # --- save ---------------------------------------------
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(cache_path, global_mask.astype(np.uint8))
    if flagged:
        flag_file.write_text("\n".join(flagged))
        print(f"Flagged {len(flagged)} scans → {flag_file}")
    print(f"Union mask saved to {cache_path}")

# ---------------- Extract the bounding box ----------------
coords = np.array(np.nonzero(global_mask))
z0, y0, x0 = coords.min(axis=1)
z1, y1, x1 = coords.max(axis=1) + 1

bbox = dict(z0=int(z0), z1=int(z1),
            y0=int(y0), y1=int(y1),
            x0=int(x0), x1=int(x1))
crop_shape = (bbox['z1']-bbox['z0'],
              bbox['y1']-bbox['y0'],
              bbox['x1']-bbox['x0'])
print("Bounding box:", bbox)
print("crop shape =", crop_shape) 

# ---------------- quick preview ---------------------------
# three-view preview -------------------------------------------------
z_mid = global_mask.shape[0] // 2          # axial (Z)
y_mid = global_mask.shape[1] // 2          # coronal (Y)
x_mid = global_mask.shape[2] // 2          # sagittal (X)

fig, ax = plt.subplots(1, 3, figsize=(9, 3))

ax[0].imshow(global_mask[z_mid],      cmap="gray")  # axial
ax[1].imshow(global_mask[:, y_mid],   cmap="gray")  # coronal
ax[2].imshow(global_mask[:, :, x_mid], cmap="gray") # sagittal

for a, t in zip(ax, [f"Z={z_mid}", f"Y={y_mid}", f"X={x_mid}"]):
    a.set_title(t); a.axis("off")

plt.tight_layout()
plt.savefig("data/global_mask_orth.png", bbox_inches="tight")
plt.close()
print("Preview saved to data/global_mask_orth.png")

