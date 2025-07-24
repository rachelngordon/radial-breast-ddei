import pandas as pd
import json
import re

# --- Load files ---
# Path to files
split_path = "data_split.json"
mapping_path = "DROSubID_vs_fastMRIbreastID.csv"
demographics_path = "breast_fastMRI_final.xlsx"

# Load data split
with open(split_path, "r") as f:
    splits = json.load(f)

# Load mapping file (fastMRIbreast <-> DRO)
mapping_df = pd.read_csv(mapping_path)
dro_to_fmri = dict(zip(mapping_df["DRO"], mapping_df["fastMRIbreast"]))

# Convert test_dro and val_dro to fastMRI IDs
def extract_dro_id(s):
    match = re.search(r"sub(\d+)", s)
    return int(match.group(1)) if match else None

test_fmri_ids = [f"fastMRI_breast_{dro_to_fmri[extract_dro_id(dro_id)]:03d}" for dro_id in splits["test_dro"]]
val_fmri_ids = [f"fastMRI_breast_{dro_to_fmri[extract_dro_id(dro_id)]:03d}" for dro_id in splits["val_dro"]]
train_fmri_ids = splits["train"]

# --- Load demographics ---
demo_df = pd.read_excel(demographics_path)

# Ensure consistent column names
demo_df.columns = demo_df.columns.str.strip()
demo_df["Patient Coded Name"] = demo_df["Patient Coded Name"].str.strip()

# Helper to compute stats
def compute_counts(df_subset):
    total = len(df_subset)
    no_lesion = (df_subset["Lesion status (0 = negative, 1= malignancy, 2= benign)"] == 0).sum()
    benign = (df_subset["Lesion status (0 = negative, 1= malignancy, 2= benign)"] == 2).sum()
    malignant = (df_subset["Lesion status (0 = negative, 1= malignancy, 2= benign)"] == 1).sum()

    malignant_right = ((df_subset["Lesion status (0 = negative, 1= malignancy, 2= benign)"] == 1) & (df_subset["Laterality (1=right, 2=left)"] == 1)).sum()
    malignant_left = ((df_subset["Lesion status (0 = negative, 1= malignancy, 2= benign)"] == 1) & (df_subset["Laterality (1=right, 2=left)"] == 2)).sum()

    return {
        "Total Patients": total,
        "No Lesion": no_lesion,
        "Benign": benign,
        "Malignant": malignant,
        "Malignant Right Breast": malignant_right,
        "Malignant Left Breast": malignant_left,
    }

# Filter by split
train_df = demo_df[demo_df["Patient Coded Name"].isin(train_fmri_ids)]
val_df = demo_df[demo_df["Patient Coded Name"].isin(val_fmri_ids)]
test_df = demo_df[demo_df["Patient Coded Name"].isin(test_fmri_ids)]

# Compute stats
train_stats = compute_counts(train_df)
val_stats = compute_counts(val_df)
test_stats = compute_counts(test_df)

# --- Print summary ---
summary = {
    "Train": train_stats,
    "Validation": val_stats,
    "Test": test_stats
}

for split, stats in summary.items():
    print(f"\n--- {split} ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
