import glob, numpy as np


bad=[]


for p in glob.glob(f"/net/scratch2/rachelgordon/zf_data_192_slices/cs_maps/**/*.npy", recursive=True):
    try: np.load(p)
    except Exception as e:
        print("BAD:", p, e); bad.append(p)
print("Total bad:", len(bad))