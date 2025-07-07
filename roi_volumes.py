import nibabel as nib
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm

root_dir = "/nfs2/harmonization/BIDS/WRAPnew/derivatives/"
pattern = "sub-*/ses-*/SLANT-TICVv1.2/post/FinalResult/*_T1w_seg.nii.gz"

pairs = pd.read_csv("labels/slant_ai_pairs.csv", usecols=["RightID", "LeftID"])
LABEL_LIST = sorted(set(pairs["RightID"]).union(pairs["LeftID"]))

paths = list(Path(root_dir).glob(pattern))
vrows = []

for seg_path in tqdm(paths, desc="Computing ROI volumes", total=len(paths)):
    subject = seg_path.parents[4].name
    session = seg_path.parents[3].name
    sid = f"{subject}_{session}"

    img = nib.load(seg_path)
    seg = img.get_fdata().astype(np.int32)
    vox = np.prod(img.header.get_zooms()[:3]) / 1e3

    vol_dict = {"subject": sid}
    for lab in LABEL_LIST:
        vol_dict[f"L{lab}"] = (seg == lab).sum() * vox
    vrows.append(vol_dict)

vdf = pd.DataFrame(vrows).set_index("subject").sort_index()
vdf.to_csv("stats_csv/roi_volumes.csv")

zdf = vdf.apply(lambda col: (col - col.mean()) / col.std(), axis=0)
zdf.to_csv("stats_csv/roi_volumes_zscore.csv")

print(vdf.iloc[:5, :8])
outliers = (zdf.abs() > 3).sum().sum()
print(f"\nTotal ROI outlier cells (|Z|>3) = {outliers}")
print("CSV saved: roi_volumes.csv, roi_volumes_zscore.csv")
