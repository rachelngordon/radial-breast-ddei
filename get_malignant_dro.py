#!/usr/bin/env python3
import re
import pandas as pd

# --- paths (edit as needed) ---
MAP_CSV   = "data/DROSubID_vs_fastMRIbreastID.csv"
META_XLSX = "data/breast_fastMRI_final.xlsx"
SHEET     = 0   # or the sheet name if needed, e.g., "Sheet1"

# 1) Load files
df_map = pd.read_csv(MAP_CSV)  # expects columns: "DRO", "fastMRIbreast"
df_meta = pd.read_excel(META_XLSX, sheet_name=SHEET)

# 2) Clean types in mapping
df_map = df_map.rename(columns={c: c.strip() for c in df_map.columns})
df_map["DRO"] = df_map["DRO"].astype(int)
df_map["fastMRIbreast"] = df_map["fastMRIbreast"].astype(int)

# 3) Extract numeric patient ID from "Patient Coded Name" like "fastMRI_breast_001" -> 1
#    and find the lesion status column (its header is long)
name_col = "Patient Coded Name"
if name_col not in df_meta.columns:
    raise ValueError(f"Couldn't find '{name_col}' in Excel columns: {list(df_meta.columns)}")

# find lesion status column by prefix match to be robust to the long header text
lesion_col = None
for c in df_meta.columns:
    if str(c).lower().startswith("lesion status"):
        lesion_col = c
        break
if lesion_col is None:
    raise ValueError("Couldn't find 'Lesion status (...)' column in Excel.")

def extract_id(s):
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else None

df_meta = df_meta.copy()
df_meta["fastMRI_id"] = df_meta[name_col].apply(extract_id)

# 4) Keep only rows with a valid numeric ID
df_meta = df_meta.dropna(subset=["fastMRI_id"])
df_meta["fastMRI_id"] = df_meta["fastMRI_id"].astype(int)

# 5) Merge mapping -> metadata
#    df_map.fastMRIbreast (int) should match df_meta.fastMRI_id (int)
df = df_map.merge(df_meta[["fastMRI_id", lesion_col]], left_on="fastMRIbreast", right_on="fastMRI_id", how="left")

# 6) Build DRO lists
# Lesion status legend: 0=negative, 1=malignancy, 2=benign
malignant_only = df.loc[df[lesion_col] == 1, "DRO"].dropna().astype(int).sort_values().tolist()
any_lesion     = df.loc[df[lesion_col].isin([1, 2]), "DRO"].dropna().astype(int).sort_values().tolist()

# 7) Save and print
pd.Series(malignant_only, name="DRO_malignant").to_csv("dro_ids_malignant_only.csv", index=False)
pd.Series(any_lesion, name="DRO_any_lesion").to_csv("dro_ids_any_lesion.csv", index=False)

print(f"# DRO with malignancy (Lesion status == 1): {len(malignant_only)}")
print(malignant_only)
print()
print(f"# DRO with any lesion (Lesion status in [1, 2]): {len(any_lesion)}")
print(any_lesion)

# Optional: show which mappings didn't find metadata (fastMRI id missing from Excel)
missing = df[df[lesion_col].isna()][["DRO", "fastMRIbreast"]]
if not missing.empty:
    print("\n# WARNING: fastMRI IDs in the mapping not found in Excel metadata:")
    print(missing.to_string(index=False))
