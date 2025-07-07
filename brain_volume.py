# nohup python brain_volume.py > logs/out_brain_volumes.log 2>&1 &

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats as st
from pathlib import Path
from tqdm.auto import tqdm


label_df = pd.read_csv(Path("labels/label_index.csv"), usecols=["IDX"])

exclude_ids = set(label_df["IDX"].tail(2))
# exclude_ids.add(0)

root_dir = "/nfs2/harmonization/BIDS/WRAPnew/derivatives/"
pattern = "sub-*/ses-*/SLANT-TICVv1.2/post/FinalResult/*_T1w_seg.nii.gz"

paths = list(Path(root_dir).glob(pattern))
rows = []

for seg_path in tqdm(paths, desc="Computing Brain Volume", total=len(paths)):
    subject = seg_path.parents[4].name
    session = seg_path.parents[3].name
    sid = f"{subject}_{session}"

    img = nib.load(seg_path)
    arr = img.get_fdata().astype(np.int32)
    vox = np.prod(img.header.get_zooms()[:3])
    brain_voxels = np.isin(arr, list(exclude_ids), invert=True).sum()
    brain_ml = brain_voxels * vox / 1e3

    rows.append({"subject": sid, "BrainVol_ml": brain_ml})

df = pd.DataFrame(rows).sort_values("subject")
df["zscore"] = st.zscore(df["BrainVol_ml"])
df.to_csv("stats_csv/brain_volume_summary.csv", index=False)

print(df.head())
print(f"\nCSV saved: brain_volume_summary.csv")
