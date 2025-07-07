import nibabel as nib
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm

root_dir = "/nfs2/harmonization/BIDS/WRAPnew/derivatives/"
pattern = "sub-*/ses-*/SLANT-TICVv1.2/post/FinalResult/*_T1w_seg.nii.gz"


pairs = pd.read_csv("labels/slant_ai_pairs.csv", usecols=["RightID", "LeftID"])
PAIRS = [tuple(row) for row in pairs.to_numpy()]

paths = list(Path(root_dir).glob(pattern))
ai_rows = []

for seg_path in tqdm(paths, desc="Computing Asymmetry Index", total=len(paths)):
    subject = seg_path.parents[4].name
    session = seg_path.parents[3].name
    sid = f"{subject}_{session}"

    img = nib.load(seg_path)
    seg = img.get_fdata().astype(np.int32)
    vox = np.prod(img.header.get_zooms()[:3]) / 1e3

    row = {"subject": sid}
    for L, R in PAIRS:
        vL = (seg == L).sum() * vox
        vR = (seg == R).sum() * vox
        ai = (vL - vR) / (vL + vR + 1e-6)
        row[f"AI_{L}-{R}"] = ai
    ai_rows.append(row)

aidf = pd.DataFrame(ai_rows).set_index("subject").sort_index()
aidf.to_csv("stats_csv/asymmetry_index.csv")

ai_outliers = (aidf.abs() > 0.15).sum().sum()
print(aidf.iloc[:5, :8])
print(f"\nTotal AI outlier cells (|AI|>0.15) = {ai_outliers}")
print("CSV saved: asymmetry_index.csv")
