"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm


class ROIVolumeCalculator:
    def __init__(
        self,
        slant_ticv_root: str | Path,
        out_dir: str | Path,
        pattern: str = "sub-*/ses-*/SLANT-TICVv1.2*/post/FinalResult/*_T1w_seg.nii.gz",
        label_index: str = "labels/label_index.csv",
    ):
        self.root_dir = Path(slant_ticv_root)
        self.pattern = pattern
        self.label_index = Path(label_index)
        self.out_csv = Path(f"{out_dir}/stats_csv/roi_volumes.csv")
        self.out_z_csv = Path(f"{out_dir}/stats_csv/roi_volumes_zscore.csv")
        label_df = pd.read_csv(self.label_index, usecols=["IDX"])
        self.LABEL_LIST = sorted(label_df["IDX"].astype(int))

    def compute_volumes(self) -> pd.DataFrame:
        paths = list(self.root_dir.glob(self.pattern))
        vrows = []
        for seg_path in tqdm(paths, desc="Computing ROI volumes", total=len(paths)):
            sid = seg_path.name[: -len("_T1w_seg.nii.gz")]
            img = nib.load(seg_path)
            seg = img.get_fdata().astype(np.int32)
            vox = np.prod(img.header.get_zooms()[:3]) / 1e3
            vol_dict = {"subject": sid}
            for lab in self.LABEL_LIST:
                vol_dict[f"L{lab}"] = (seg == lab).sum() * vox
            vrows.append(vol_dict)
        vdf = pd.DataFrame(vrows).set_index("subject").sort_index()
        return vdf

    def save_volumes(self, vdf: pd.DataFrame):
        self.out_csv.parent.mkdir(exist_ok=True, parents=True)
        vdf.to_csv(self.out_csv)
        print(f"Saved ROI volumes to {self.out_csv}")

    def compute_zscores(self, vdf: pd.DataFrame) -> pd.DataFrame:
        zdf = vdf.apply(lambda col: (col - col.mean()) / col.std(), axis=0)
        return zdf

    def save_zscores(self, zdf: pd.DataFrame):
        self.out_z_csv.parent.mkdir(exist_ok=True, parents=True)
        zdf.to_csv(self.out_z_csv)
        print(f"Saved ROI z-scores to {self.out_z_csv}")

    def run(self):
        vdf = self.compute_volumes()
        self.save_volumes(vdf)
        zdf = self.compute_zscores(vdf)
        self.save_zscores(zdf)
