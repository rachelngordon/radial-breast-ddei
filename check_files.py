import os
import numpy as np

DIR = "/net/scratch2/rachelgordon/"   # <-- CHANGE THIS

bad_files = []

for root, _, files in os.walk(DIR):
    for f in files:
        if f.endswith(".npy") or f.endswith(".npz"):
            path = os.path.join(root, f)
            try:
                np.load(path)
            except Exception as e:
                print(f"[BAD] {path} -> {type(e).__name__}: {e}")
                bad_files.append(path)

print("\n=== SUMMARY ===")
print(f"Total bad files: {len(bad_files)}")
for b in bad_files:
    print(b)
