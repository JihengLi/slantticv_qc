"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path


class VolumeQC:
    def __init__(
        self,
        out_dir: str | Path,
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

    def plot_distribution(
        self, series: np.ndarray, name: str, ylabel: str, png_name: str
    ):
        fig = plt.figure(constrained_layout=True, figsize=(15, 8))
        gs = GridSpec(1, 2, figure=fig)

        # Violin + box
        ax1 = fig.add_subplot(gs[0, 0])
        parts = ax1.violinplot(series, showmeans=False, showmedians=True, widths=0.6)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
        ax1.boxplot(series, widths=0.15, vert=True, showfliers=False, positions=[1])
        ax1.set_ylabel(ylabel)
        ax1.set_xticks([])
        ax1.set_title(f"{name}: Violin + Box")

        # Histogram + KDE
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(series, bins=30, alpha=0.6, density=True, label="Histogram")
        kde_x = np.linspace(series.min(), series.max(), 200)
        kde_y = st.gaussian_kde(series)(kde_x)
        ax2.plot(kde_x, kde_y, linewidth=2, label="KDE")
        ax2.axvline(series.mean(), linestyle="--", label="Mean")
        ax2.set_xlabel(ylabel)
        ax2.set_ylabel("Density")
        ax2.set_title(f"{name}: Histogram + KDE")
        ax2.legend()

        fig.suptitle(f"{name} Distribution QC", fontsize=14)
        png_dir = self.out_dir / "stats_png"
        png_dir.mkdir(parents=True, exist_ok=True)
        out_png = png_dir / png_name
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"Saved distribution plot to {out_png}")

    def run(self):
        # compute metrics
        self.compute_brain()
        self.compute_ticv()
        # save tables
        self.save_brainvol_csv()
        # plot distributions
        self.plot_distribution(
            self.brain_df["BrainVol_ml"].values,
            name="Brain Volume",
            ylabel="BrainVol (mL)",
            png_name="brainvol_violin_hist.png",
        )
        self.plot_distribution(
            self.ticv_df["TICV_ml"].values,
            name="TICV",
            ylabel="TICV (mL)",
            png_name="ticv_violin_hist.png",
        )
