# pad_mri.py
#
# Overwrite every *.nii[.gz] under SRC_DIR with a 128³-padded version.
#
#   python pad_to_128_overwrite.py /path/to/data  [n_jobs]
#
# Requires: nibabel, numpy, tqdm
# ----------------------------------------------------------------------

import os, sys, glob, numpy as np, nibabel as nib
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

TARGET = 128

def pad_and_save(fname: str):
    """
    Pad the spatial dimensions of a NIfTI file to 128³.
    • 3-D input  (X, Y, Z)    → (128, 128, 128)
    • 4-D input  (X, Y, Z, T) → (128, 128, 128, T)
    The file is overwritten in-place.
    """
    img   = nib.load(fname)
    data  = img.get_fdata(dtype=np.float32)

    # --- spatial shape & sanity check ---------------------------------------
    if data.ndim not in (3, 4):
        raise ValueError(f"{fname} has unsupported ndim {data.ndim}")

    dx, dy, dz = data.shape[:3]
    if dx > TARGET or dy > TARGET or dz > TARGET:
        raise ValueError(f"{fname} has spatial dims larger than {TARGET}")

    # --- compute symmetric padding ------------------------------------------
    pad_before = [(TARGET - dx) // 2,
                  (TARGET - dy) // 2,
                  (TARGET - dz) // 2]
    pad_after  = [TARGET - (d + b) for d, b in zip((dx, dy, dz), pad_before)]

    # pad spec for np.pad
    pad_spec = [(pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2])]

    if data.ndim == 4:                # keep diffusion/time axis unchanged
        pad_spec.append((0, 0))

    padded = np.pad(data, pad_width=pad_spec,
                    mode="constant", constant_values=0)

    # --- affine shift to keep world coords ----------------------------------
    affine = img.affine.copy()
    affine[:3, 3] -= affine[:3, :3] @ pad_before

    nib.save(nib.Nifti1Image(padded, affine, header=img.header), fname)
    return 1   # for tqdm counting

def main(src_dir, n_jobs=None):
    files = sorted(glob.glob(os.path.join(src_dir, "**", "*.nii*"), recursive=True))
    if not files:
        sys.exit("No NIfTI files found.")

    files = files[2:]  # for testing, remove this line in production

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        list(tqdm(ex.map(pad_and_save, files, chunksize=4),
                  total=len(files), desc="Padding to 128^3"))

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.exit("Usage: python pad_mri.py /data/dir [n_jobs]")
    src_dir = sys.argv[1]
    jobs = int(sys.argv[2]) if len(sys.argv) == 3 else None
    main(src_dir, jobs)
