import os

# Set the path to your directory
directory = "/ess/scratch/scratch1/rachelgordon/dce-8tf/binned_kspace"

# List all files in the directory
files = os.listdir(directory)

# Extract patient IDs from filenames like: fastMRI_breast_XXX_2.h5
present_ids = set()
for filename in files:
    if filename.startswith("fastMRI_breast_") and filename.endswith("_2.h5"):
        try:
            id_str = filename.split('_')[2]
            patient_id = int(id_str)
            present_ids.add(patient_id)
        except (IndexError, ValueError):
            pass  # Skip files not matching expected pattern

# Compare with expected IDs 1 to 300
expected_ids = set(range(1, 301))
missing_ids = sorted(expected_ids - present_ids)

# Print the missing IDs
print(f"Missing {len(missing_ids)} IDs:")
print(missing_ids)
