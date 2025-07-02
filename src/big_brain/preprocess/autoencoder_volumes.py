from pathlib import Path
import numpy as np, nibabel as nib, json, argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def find_sessions(root: Path):
    return sorted(root.rglob("*_dwi_allruns.nii.gz"))

def normalise_dwi(dwi_data: np.ndarray, bvals: np.ndarray):
    """
    Load a 4-D DWI series and apply robust, session-level gain and z-score normalisation.

    Parameters
    ----------
    dwi_path : Path
        Path to the cleaned 4-D DWI NIfTI (shape [X,Y,Z,N]).
    bval_path : Path
        Path to the matching .bval file (one row of N b-values).

    Returns
    -------
    dwi_norm : ndarray (float32)
        The fully normalised DWI data (same shape as input).
    stats : dict
        {
          'gain': float,       # scale factor applied to align b0 median → 1.0
          'mean': float,       # session-wide mean after gain scaling
          'std':  float        # session-wide std  after gain scaling
        }
    """

    # 1) Identify all b0 volumes (b-value == 0)
    b0_indices  = np.where(bvals == 0)[0]         # e.g. array([0, 10, 20])

    # 2) Extract those b0 volumes and form a union-mask of nonzero voxels
    #    (handles slight mis-alignments: if *any* run has signal, we treat it as brain)
    b0_volumes = np.take(dwi_data, b0_indices, axis=3)  # shape (X, Y, Z, N_b0)
    mask       = np.any(b0_volumes > 0, axis=3)         # boolean mask [X,Y,Z]

    # 3) Remove any stray zero-intensity voxels inside the union mask
    #    (e.g. holes due to warping) before computing the gain
    b0_values  = b0_volumes[mask].ravel()
    b0_values  = b0_values[b0_values > 0]               # drop zeros

    # 4) SESSION-GAIN NORMALISATION:
    #    Anchor the median of all b0 tissue intensities to 1.0,
    #    removing scanner-/coil-level scale differences across sessions
    gain       = 1.0 / (np.median(b0_values) + 1e-12)
    dwi_scaled = dwi_data * gain

    # 5) SESSION-WIDE Z-SCORE NORMALISATION:
    #    Compute mean/std across all brain voxels in all volumes,
    #    giving zero-mean/unit-variance inputs for the autoencoder,
    #    yet preserving relative shell attenuation patterns.
    dwi_values = dwi_scaled[mask, ...].ravel()
    dwi_values = dwi_values[dwi_values > 0]              # drop any sneaky zeros
    mean       = dwi_values.mean()
    std        = dwi_values.std() + 1e-6
    dwi_norm   = (dwi_scaled - mean) / std

    # 6) Return the normalised data plus the stats for reproducibility
    return dwi_norm.astype(np.float32)

def crop_to_shape(x: np.ndarray, target_shape=(80, 96, 80)) -> np.ndarray:
    """
    Center-crop a 3D array x to the given target_shape.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (D, H, W), e.g. (91, 109, 91).
    target_shape : tuple of ints
        Desired output shape (d, h, w), e.g. (80, 96, 80).

    Returns
    -------
    np.ndarray
        Cropped array of shape target_shape.
    """
    D, H, W = x.shape
    td, th, tw = target_shape

    # compute start indices for cropping
    sd = (D - td) // 2
    sh = (H - th) // 2
    sw = (W - tw) // 2

    # compute end indices
    ed = sd + td
    eh = sh + th
    ew = sw + tw

    return x[sd:ed, sh:eh, sw:ew]


def pad_to_shape(arr: np.ndarray, target_shape=(96, 112, 96)) -> np.ndarray:
    """
    Symmetrically pad a 3D NumPy array with zeros so that its shape matches target_shape.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (d1, d2, d3).
    target_shape : tuple of three ints
        Desired output shape (t1, t2, t3). Each ti must be >= corresponding input size.

    Returns
    -------
    padded : np.ndarray
        Zero-padded array of shape target_shape.

    Raises
    ------
    ValueError
        If any target dimension is smaller than the input's.
    """
    if arr.ndim != 3:
        raise ValueError(f"Input array must be 3D, got {arr.ndim}D.")

    current_shape = arr.shape
    pad_width = []
    for cur, tgt in zip(current_shape, target_shape):
        if tgt < cur:
            raise ValueError(f"Target size {tgt} is smaller than current size {cur}.")
        total_padding = tgt - cur
        # split padding as evenly as possible
        pad_before = total_padding // 2
        pad_after = total_padding - pad_before
        pad_width.append((pad_before, pad_after))

    # pad with zeros
    padded = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
    return padded


def process_session(dwi_path: Path):
    """
    Process a single 4-D DWI session and cache its 3-D gradient volumes.

    Parameters
    ----------
    dwi_path : pathlib.Path
        Path to the cleaned 4-D DWI NIfTI file (shape [X, Y, Z, N]).

    Steps
    -----
    1) Derive the matching .bval and .bvec file paths.
    2) Parse patient ID (sub-XXX) and session ID (ses-YYY) from the filename.
    3) Create an output directory at CACHE/sub-XXX/ses-YYY/.
    4) Load the raw 4-D DWI data and corresponding b-values.
    5) Load the .bval file containing the b-values for each gradient.
    6) Normalize the entire volume (session-level gain + z-score).
    7) Split the normalized 4-D volume into N individual 3-D gradient arrays and save each gradient as a compressed .npz containing:
        - vol_data: 3-D image array
        - bval: single float b-value
        - bvec: 3-D b-vector (if available, currently not used)
        - affine: 4 x 4 spatial transform
        - patient, session tags
    """

    # 1) Derive file stems for .bval and .bvec
    #    We strip off the trailing ".nii.gz" by slicing off 7 chars
    base      = dwi_path.with_suffix("").with_suffix("")  # remove .nii.gz (with_suffix only removes one suffix at a time)
    bval_path = base.with_suffix(".bval")
    bvec_path = base.with_suffix(".bvec")

    # 2) Extract patient & session IDs from the BIDS-style filename
    #    Filename looks like "sub-XXX_ses-YYY_dwi_allruns.nii.gz"
    p_id = base.name.split("_")[0]  # e.g. "sub-0001"
    s_id = base.name.split("_")[1]  # e.g. "ses-01"

    # 3) Prepare the output folder for this session
    #    e.g. cache/sub-0001_ses-01/
    out_dir = CACHE / p_id / s_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) Load the 4-D DWI image (keeps affine + header for later)
    dwi_img = nib.load(dwi_path)                             # nibabel Nifti1Image
    dwi_raw = dwi_img.get_fdata().astype(np.float32)         # (X, Y, Z, N)

    # 5) Load acquisition metadata
    #    .bval: one row of N diffusion weightings
    #    .bvec: 3×N axis vectors
    bvals = np.loadtxt(bval_path)                            # shape (N,)
    bvecs = np.loadtxt(bvec_path)                            # shape (3, N)

    # 6) Normalize the 4-D data with our robust session-level function
    #    Returns the normalized array
    dwi_norm = normalise_dwi(dwi_raw, bvals)

    # 7) Split the normalized 4-D volume into individual 3-D gradient volumes
    #    and save each as a compressed .npz with all relevant metadata.
    for g in range(dwi_norm.shape[3]):
        vol_data = dwi_norm[..., g]  # 3-D array (X, Y, Z)

        # Pad volumes from 91, 109, 91 into shape 96, 112, 96
        vol_data = pad_to_shape(vol_data, target_shape=(96, 112, 96))

        out_file = out_dir / f"{p_id}_{s_id}_grad{g:03d}.npz"
        np.savez_compressed(
            out_file,
            vol_data=vol_data,                          # 3D gradient volume (X, Y, Z)
            bval=np.float32(bvals[g]),                  # single b-value for this gradient
            bvec=np.float32(bvecs[:, g]),               # 3D b-vector for this gradient
            affine=dwi_img.affine.astype(np.float32),   # preserves spatial orientation for nifti reconstruction
            patient=p_id,                               # for downstream grouping or sampling
            session=s_id                                # ditto
        )


def main(root: Path, cache: Path, n_jobs: int):
    # ensure CACHE is globally visible to process_session
    global CACHE
    CACHE = cache
    CACHE.mkdir(parents=True, exist_ok=True)

    sessions = find_sessions(root)
    print(f"Found {len(sessions)} sessions; processing with {n_jobs} workers.")

    # use imap_unordered for best throughput and immediate progress updates
    with Pool(n_jobs) as pool:
        for _ in tqdm(pool.imap_unordered(process_session, sessions),
                      total=len(sessions),
                      desc="Preprocessing"):
            pass

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch-process DWI sessions.")
    p.add_argument(
        "--root", type=Path,
        default=Path("/home/spieterman/dev/dwi-preprocessing/data/preproc"),
        help="Where to find *_dwi_allruns.nii.gz files."
    )
    p.add_argument(
        "--cache", type=Path,
        default=Path("/home/spieterman/dev/big-brain-model/data/encoder"),
        help="Where to write per-gradient .npz files."
    )
    p.add_argument(
        "--n-jobs", type=int, default=cpu_count(),
        help="Number of parallel worker processes."
    )
    args = p.parse_args()
    main(args.root, args.cache, args.n_jobs)
