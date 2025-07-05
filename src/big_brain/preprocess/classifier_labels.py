from pathlib import Path
import re
import pandas as pd
import numpy as np

CDR_CSV = Path("data/labels/raw/OASIS3_cdr.csv")
DEMO_SCV = Path("data/labels/raw/OASIS3_demographics.csv")
LATENT_DIR = Path("data/transformer")
pd.set_option('display.max_rows', 500)

# 1. Parse the patient and session information from the latent files

npz_glob = "*_latent.npz"

sub_rx = re.compile(r"sub-(OAS3\d{4})")     # matches "sub-OAS30001" etc.
ses_rx = re.compile(r"ses-(d\d{4})")        # matches "ses-d0001" etc.

rows = []
for path in LATENT_DIR.rglob(npz_glob):
    # path example: /data/transformer/sub-OAS30001/ses-0001/sub-OAS30001_ses-d0001_latent.npz
    p_str = path.as_posix()

    # Extract patient and session from the path
    sub_match = sub_rx.search(p_str)
    ses_match = ses_rx.search(p_str)
    if not sub_match or not ses_match:
        raise ValueError(f"Invalid path format: {p_str}")
    
    patient = sub_match.group(1)  # e.g., "OAS30001"
    session = ses_match.group(1)  # e.g., "d0001"
    scan_day = int(session.lstrip("d"))  # Convert "d0001" to 1

    rows.append({
        "patient": patient,
        "session": session,
        "scan_day": scan_day
    })

# Create a DataFrame from the collected rows
latents_df = pd.DataFrame(rows, dtype=str)
latents_df["scan_day"] = latents_df["scan_day"].astype(int)

# 2. match the scan session days with the nearest CDR days

# Read the CDR CSV file and rename columns for clarity
cdr_df = pd.read_csv(CDR_CSV, dtype=str).rename(columns={
    "OASISID":             "patient",
    "days_to_visit":       "cdr_day",       # day in the study when CDR was recorded
    "CDRTOT":              "cdr_global",    # global CDR score
    }).assign(
        cdr_day=lambda df: df["cdr_day"].astype(int),
        cdr_global=lambda df: df["cdr_global"].astype(float),
        )

# Take only the relevant columns
cdr_df = cdr_df[["patient", "cdr_day", "cdr_global"]]

# Sort cdr_df and latents_df by patient and scan day
latents_df = (
    latents_df
    .sort_values(
        by=["patient", "scan_day"],
        ascending=[True, True],
        kind="mergesort"     # ← stable sort
    )
    .reset_index(drop=True)
)

cdr_df = (
    cdr_df
    .sort_values(
        by=["patient", "cdr_day"],
        ascending=[True, True],
        kind="mergesort"
    )
    .reset_index(drop=True)
)

# Merge the CDR scores with the latent scan days
merged_parts = []

for patient_id, sub_df in latents_df.groupby("patient"):
    # grab only that patient’s CDR rows, and sort both pieces
    cdr_sub = (
        cdr_df[cdr_df["patient"] == patient_id]
        .sort_values("cdr_day")
    )
    ses_sub = (
        sub_df
        .sort_values("scan_day")
    )
    # merge just this patient
    merged_parts.append(
        pd.merge_asof(
            ses_sub,
            cdr_sub,
            left_on  = "scan_day",
            right_on = "cdr_day",
            direction= "nearest",
            suffixes=("", "_cdr"),
        )
    )

# Concatenate all the merged parts into a single DataFrame
merged_df = pd.concat(merged_parts, ignore_index=True)

# Calculate the absolute difference between scan day and cdr day
merged_df["delta_day"] = (merged_df["scan_day"] - merged_df["cdr_day"]).abs()

# Take only the relevant columns
merged_df = merged_df[["patient", "session", "scan_day", "cdr_day", "delta_day", "cdr_global"]]

# 3. Enforce that the CDR score is monotonic increasing

# Sort merged_df by patient and scan day
merged_df = merged_df.sort_values(['patient', 'scan_day']).reset_index(drop=True)

# Always take the maximum CDR score up to each scan day
merged_df['cdr_monotonic'] = (
    merged_df
    .groupby('patient')['cdr_global']
    .cummax()
)

# 4. Read the demographics file and merge it with the CDR data

# Read the demographics CSV file and rename columns for clarity
demo_df = pd.read_csv(DEMO_SCV, dtype=str).rename(columns={
    "OASISID": "patient",
    "GENDER": "gender",
    "AgeatEntry": "entry_age",  # age at entry
    "HAND": "handedness",
})

# Convert entry_age to float
demo_df["entry_age"] = demo_df["entry_age"].astype(float)

# Clean up the demographics DataFrame
demo_df["gender"] = demo_df["gender"].map({"1": "male", "2": "female"})
demo_df["handedness"] = demo_df["handedness"].map({"R": "right", "L": "left", "B": "both", " ": "right", "": "right"})

# Take only the relevant columns
demo_df = demo_df[["patient", "gender", "handedness", "entry_age"]]

# Merge demographics with the merged DataFrame
merged_df = pd.merge(
    merged_df,
    demo_df,
    on="patient",
    how="left",
)

# Calculate age at each scan session
merged_df["age_at_scan"] = merged_df["entry_age"] + merged_df["scan_day"] / 365.25

# Take only the relevant columns for the final DataFrame
final_df = merged_df[[
    "patient",
    "session",
    "cdr_monotonic",
    "gender",
    "handedness",
    "age_at_scan",]]

# Rename columns for clarity
final_df = final_df.rename(columns={
    "cdr_monotonic": "cdr",
    "age_at_scan": "age",
})

# print the final DataFrame for verification
print(final_df.head())

# 

