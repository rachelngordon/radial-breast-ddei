import json
import pandas as pd

# --- Configuration ---

# 1. Path to your patient splits JSON file
splits_file_path = "/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/patient_splits.json"

# 2. Path to your CSV file containing the DRO and fastMRI ID mapping
overlap_csv_path = "DROSubID_vs_fastMRIbreastID.csv"

# --- Main Logic ---

def find_dro_samples_in_splits(splits_path, csv_path):
    """
    Loads patient splits (train/val/test) and an overlap CSV to find corresponding DRO samples.

    Args:
        splits_path (str): Path to the patient_splits.json file.
        csv_path (str): Path to the CSV file mapping DRO to fastMRI IDs.

    Returns:
        dict: A dictionary containing all results, or None if an error occurs.
              Keys: 'total_train_patients', 'total_val_patients', 'total_test_patients',
                    'dro_train', 'dro_val', 'dro_test', 'dro_unused'.
    """
    # --- Step 1: Load all patient splits and get total counts ---
    try:
        with open(splits_path, 'r') as f:
            patient_splits = json.load(f)
        
        # Load train, validation, and test sets
        train_patients = patient_splits.get('train', [])
        val_patients = patient_splits.get('val', patient_splits.get('validation', []))
        test_patients = patient_splits.get('test', [])
        
        # Capture the total number of patients in each split
        total_train_count = len(train_patients)
        total_val_count = len(val_patients)
        total_test_count = len(test_patients)
        
        print(f"Successfully loaded {total_train_count} train, {total_val_count} val, and {total_test_count} test patients from splits file.")
    except FileNotFoundError:
        print(f"Error: The JSON splits file '{splits_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{splits_path}'. Check the file format.")
        return None

    # --- Step 2: Load the overlap data from the CSV file ---
    try:
        overlap_df = pd.read_csv(csv_path)
        print(f"Successfully loaded the overlap data from '{csv_path}'.")
    except FileNotFoundError:
        print(f"Error: The CSV overlap file '{csv_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # --- Step 3: Create a lookup map and get a set of all DRO IDs ---
    id_col = 'fastMRIbreast'
    dro_col = 'DRO'

    fastmri_to_dro_map = {}
    for _, row in overlap_df.iterrows():
        fastmri_id_num = row[id_col]
        dro_id = row[dro_col]
        full_fastmri_id = f"fastMRI_breast_{int(fastmri_id_num):03d}"
        fastmri_to_dro_map[full_fastmri_id] = dro_id

    all_dro_ids = set(overlap_df[dro_col].unique())
    print(f"Created a lookup map with {len(fastmri_to_dro_map)} entries.")
    print(f"Found {len(all_dro_ids)} unique DRO samples in the overlap CSV.")

    # --- Step 4: Find the corresponding DRO samples for each split ---
    dro_train = sorted([fastmri_to_dro_map[p] for p in train_patients if p in fastmri_to_dro_map])
    dro_val = sorted([fastmri_to_dro_map[p] for p in val_patients if p in fastmri_to_dro_map])
    dro_test = sorted([fastmri_to_dro_map[p] for p in test_patients if p in fastmri_to_dro_map])
    
    # --- Step 5: Find DRO samples not used in any split ---
    found_dro_ids = set(dro_train + dro_val + dro_test)
    unused_dro_ids = all_dro_ids - found_dro_ids

    # --- Step 6: Return all results in a single dictionary ---
    results = {
        "total_train_patients": total_train_count,
        "total_val_patients": total_val_count,
        "total_test_patients": total_test_count,
        "dro_train": dro_train,
        "dro_val": dro_val,
        "dro_test": dro_test,
        "dro_unused": sorted(list(unused_dro_ids)),
    }
    return results


# --- Execute the script and print results ---
if __name__ == "__main__":
    results = find_dro_samples_in_splits(splits_file_path, overlap_csv_path)

    if results:
        print("\n" + "="*60)
        print("                     RESULTS SUMMARY")
        print("="*60)
        
        # --- Training Set Information ---
        print("\n--- Training Set ---")
        print(f"Total patients in original training split: {results['total_train_patients']}")
        print(f"Mapped DRO Samples in Training Set ({len(results['dro_train'])} found):")
        print(results['dro_train'])

        # --- Validation Set Information ---
        print("\n--- Validation Set ---")
        print(f"Total patients in original validation split: {results['total_val_patients']}")
        print(f"Mapped DRO Samples in Validation Set ({len(results['dro_val'])} found):")
        print(results['dro_val'])
        
        # --- Test Set Information ---
        print("\n--- Test Set ---")
        print(f"Total patients in original test split: {results['total_test_patients']}")
        print(f"Mapped DRO Samples in Test Set ({len(results['dro_test'])} found):")
        print(results['dro_test'])

        # --- Unused DRO Samples Information ---
        print("\n--- Unused DRO Samples ---")
        print(f"DRO Samples from CSV not found in any split ({len(results['dro_unused'])}):")
        print(results['dro_unused'])

        print("\n" + "="*60)