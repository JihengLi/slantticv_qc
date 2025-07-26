"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

import pandas as pd
import scipy.stats as st

from typing import Union
from pathlib import Path
from visualization import plot_distribution


class VolumeQC:
    def __init__(
        self,
        out_dir: Union[str, Path],
    ):
        self.out_dir = Path(out_dir)
        roi_volumes_csv = Path(out_dir) / "stats_csv" / "roi_volumes.csv"
        self.vdf = pd.read_csv(roi_volumes_csv, index_col="subject")

    def compute_brain(self) -> pd.DataFrame:
        # exclude background (L0), TICV and PosteriorFossa
        exclude_lbls = ["L0", "L208", "L209"]
        # sum all columns not in exclude
        cols = [c for c in self.vdf.columns if c not in exclude_lbls]
        brain_ml = self.vdf[cols].sum(axis=1)
        df = pd.DataFrame({"subject": brain_ml.index, "BrainVol_ml": brain_ml.values})
        df["zscore"] = st.zscore(df["BrainVol_ml"])
        self.brain_df = df
        return df

    def compute_ticv(self) -> pd.DataFrame:
        # TICV -> L208
        col = "L208"
        if col not in self.vdf.columns:
            raise KeyError(f"{col} not found in ROI volumes")
        df = pd.DataFrame({"subject": self.vdf.index, "TICV_ml": self.vdf[col].values})
        df["zscore"] = st.zscore(df["TICV_ml"])
        self.ticv_df = df
        return df

    def save_brainvol_csv(self):
        brainvol_csv = self.out_dir / "stats_csv" / "brain_volume.csv"
        brainvol_csv.parent.mkdir(parents=True, exist_ok=True)
        self.brain_df.to_csv(brainvol_csv, index=False)
        print(f"Saved brain volumes to {brainvol_csv}")

    def run(self):
        self.compute_brain()
        self.compute_ticv()
        self.save_brainvol_csv()
        plot_distribution(
            self.brain_df["BrainVol_ml"].values,
            name="Brain Volume",
            ylabel="BrainVol (mL)",
            png_name="BrainVol.png",
            out_dir=self.out_dir / "stats_png",
        )
        plot_distribution(
            self.ticv_df["TICV_ml"].values,
            name="TICV",
            ylabel="TICV (mL)",
            png_name="TICV.png",
            out_dir=self.out_dir / "stats_png",
        )
