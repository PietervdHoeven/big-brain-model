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

# Fix NaN values in the handedness column
final_df["handedness"] = final_df["handedness"].fillna("right")

# prepend sub- and ses- to patient and session columns
final_df["patient"] = "sub-" + final_df["patient"]
final_df["session"] = "ses-" + final_df["session"]

# print the final DataFrame for verification
print(final_df.head())

# print all rows with NaN values (final check)
print(final_df[final_df.isna().any(axis=1)].sort_values("patient"))

# save the final DataFrame to a parquet file
final_df.to_parquet("data/labels/labels.parquet", index=False)

# print the unique values of the cdr, gender, and handedness columns. Also print the number of unique values in each column.
print("Unique CDR values:", final_df["cdr"].unique())
print("CDR value counts:\n", final_df["cdr"].value_counts())
print("Unique gender values:", final_df["gender"].unique())
print("Gender value counts:\n", final_df["gender"].value_counts())
print("Unique handedness values:", final_df["handedness"].unique())
print("Handedness value counts:\n", final_df["handedness"].value_counts())

# plot the distribution of the age column
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
final_df["age"].hist(bins=30, edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(False)

# save the plot in outputs/figures (mkdir if it doesn't exist)
output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(output_dir / "age_distribution.png")



"""
        patient    session  cdr  gender handedness        age
0  sub-OAS30001  ses-d0757  0.0  female      right  67.267053
1  sub-OAS30001  ses-d2430  0.0  female      right  71.847477
2  sub-OAS30001  ses-d3132  0.0  female      right  73.769449
3  sub-OAS30001  ses-d4467  0.0  female      right  77.424479
4  sub-OAS30002  ses-d1680  0.0    male      right  71.851689

Unique CDR values: [0.  0.5 1.  2. ]
CDR value counts:
 cdr
0.0    1365
0.5     283
1.0      84
2.0       6

Unique gender values: ['female' 'male']
Gender value counts:
 gender
female    956
male      782

Unique handedness values: ['right' 'left' 'both']
Handedness value counts:
 handedness
right    1581
left      151
both        6

"""