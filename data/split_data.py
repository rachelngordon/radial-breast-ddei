import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Path to the Excel file
excel_path = "/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/process_data/data_splits/breast_fastMRI_final.xlsx"

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_path)

# Assume the columns are named exactly as given:
# "Patient Coded Name", "Data split (0=training, 1=testing)"
patient_col = "Patient Coded Name"
split_col = "Data split (0=training, 1=testing)"

# 1. Separate patient IDs into train vs. test according to the "Data split" column
train_df = df[df[split_col] == 0]
test_df  = df[df[split_col] == 1]

train_ids = train_df[patient_col].tolist()
test_ids  = test_df[patient_col].tolist()

# 2. Further split the train_ids into train vs. val (80/20 split)
#    Use a fixed random_state for reproducibility
train_ids_final, val_ids = train_test_split(
    train_ids,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

# 3. Package everything into a dictionary
splits = {
    "train": train_ids_final,
    "val": val_ids,
    "test": test_ids
}

print("Number of train patients: ", len(train_ids_final))
print("Number of val patients: ", len(val_ids))
print("Number of test patients: ", len(test_ids))

# 4. Save to a JSON file so you can load it easily later
output_path = "patient_splits.json"
with open(output_path, "w") as fp:
    json.dump(splits, fp, indent=4)

print(f"Saved train/val/test splits to {output_path}")
