import pandas as pd
import random
import json

# Set seed for reproducibility
random.seed(42)

# Load mapping CSV
csv_path = "DROSubID_vs_fastMRIbreastID.csv"
df = pd.read_csv(csv_path)

# Convert to dict: fastMRI ID -> DRO ID
fastmri_to_dro = dict(zip(df["fastMRIbreast"], df["DRO"]))

# All fastMRI IDs from 1 to 300, excluding 6 and 279
all_ids = [i for i in range(1, 301) if i not in [6, 279]]

# Overlapping fastMRI IDs with DRO
dro_overlap_ids = list(set(fastmri_to_dro.keys()) & set(all_ids))

# Randomly select 25 for test and 15 for val
test_ids = random.sample(dro_overlap_ids, 25)
remaining_dro_ids = list(set(dro_overlap_ids) - set(test_ids))
val_ids = random.sample(remaining_dro_ids, 15)

# Training set: all remaining IDs not in test or val
train_ids = sorted(list(set(all_ids) - set(test_ids) - set(val_ids)))

# Format fastMRI IDs
def format_fmri_id(i):
    return f"fastMRI_breast_{i:03d}"

# Format DRO sample
def format_dro_id(i):
    dro_id = fastmri_to_dro[i]
    return f"sample_{dro_id:03d}_sub{dro_id}"

# Build dictionary
split = {
    "train": [format_fmri_id(i) for i in sorted(train_ids)],
    "val": [format_fmri_id(i) for i in sorted(val_ids)],
    "val_dro": [format_dro_id(i) for i in sorted(val_ids)],
    "test_dro": [format_dro_id(i) for i in sorted(test_ids)],
}

# Save to JSON
with open("data_split.json", "w") as f:
    json.dump(split, f, indent=4)

print("Split written to data_split.json")
