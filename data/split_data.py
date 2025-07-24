import pandas as pd
import json
import random

# Load CSV
csv_path = "DROSubID_vs_fastMRIbreastID.csv"
df = pd.read_csv(csv_path)

# Extract test set (from DRO mappings)
test_ids = set(df['fastMRIbreast'].tolist())

# Create the full set of fastMRI breast patient IDs from 1 to 300
all_ids = set(range(1, 301))

# Remaining IDs for train/val
train_val_ids = list(all_ids - test_ids)

# Shuffle for randomness
random.seed(42)  # For reproducibility
random.shuffle(train_val_ids)

# 80/20 split
split_idx = int(0.8 * len(train_val_ids))
train_ids = train_val_ids[:split_idx]
val_ids = train_val_ids[split_idx:]

# Helper to format IDs with zero-padding
def format_id(pid):
    return f"fastMRI_breast_{pid:03d}"

# Format all sets
splits = {
    "train": [format_id(pid) for pid in sorted(train_ids)],
    "val": [format_id(pid) for pid in sorted(val_ids)],
    "test": [format_id(pid) for pid in sorted(test_ids)],
}

# Save to JSON
with open("patient_splits.json", "w") as f:
    json.dump(splits, f, indent=4)

# Print counts
print(f"Train set size: {len(splits['train'])}")
print(f"Validation set size: {len(splits['val'])}")
print(f"Test set size: {len(splits['test'])}")
