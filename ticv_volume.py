import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats as st

from pathlib import Path
from tqdm.auto import tqdm

root_dir = "/nfs2/harmonization/BIDS/WRAPnew/derivatives/"
pattern = "sub-*/ses-*/SLANT-TICVv1.2/post/FinalResult/*_T1w_seg.nii.gz"

paths = list(Path(root_dir).glob(pattern))
rows = []

for seg_path in tqdm(paths, desc="Computing TICV"):
    subject = seg_path.parents[4].name
    session = seg_path.parents[3].name
    sid = f"{subject}_{session}"

    img = nib.load(seg_path)
    arr = img.get_fdata().astype(np.int32)
    vox = np.prod(img.header.get_zooms()[:3])
    ticv_ml = (arr == 208).sum() * vox / 1e3

    rows.append({"subject": sid, "TICV_ml": ticv_ml})

df = pd.DataFrame(rows).sort_values("subject")
df["zscore"] = st.zscore(df["TICV_ml"])
df.to_csv("ticv_summary.csv", index=False)
