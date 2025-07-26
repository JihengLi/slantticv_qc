"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

import pandas as pd
import matplotlib
import imageio.v2 as iio

from pathlib import Path
from typing import Union
from visualization import *

matplotlib.use("Agg")


class ZScoreAnalyzer:
    def __init__(
        self,
        slant_ticv_root: Union[str, Path],
        out_dir: Union[str, Path],
        label_index: str = "labels/label_index.csv",
        cell_thr: float = 3.0,
        sub_thr: int = 5,
        roi_frac_thr: float = 0.05,
        max_roi_show: int = 135,
    ):
        self.root_dir = Path(slant_ticv_root)
        self.out_dir = Path(out_dir)
        self.label_index = Path(label_index)
        self.roi_volume_csv = self.out_dir / "stats_csv" / "roi_volumes.csv"
        self.roi_zscore_csv = self.out_dir / "stats_csv" / "roi_volumes_zscore.csv"
        self.cell_thr = cell_thr
        self.sub_thr = sub_thr
        self.roi_frac_thr = roi_frac_thr
        self.max_roi_show = max_roi_show
        self.prob_subjectid = []
        self.prob_rois = []

        vdf = pd.read_csv(self.roi_volume_csv)
        zdf = pd.read_csv(self.roi_zscore_csv)
        self.vdf = vdf.set_index("subject").apply(pd.to_numeric)
        self.zdf = zdf.set_index("subject").apply(pd.to_numeric)
        self.idx2labeldf = pd.read_csv(
            self.label_index, usecols=["IDX", "LABEL"]
        ).set_index("IDX")["LABEL"]

    def _cell_level_str(self) -> str:
        mask = self.zdf.abs() >= self.cell_thr
        total = mask.values.sum()
        pct = mask.values.mean()
        return (
            "===== Cell-level Summary =====\n"
            f"Cells with |z| ≥ {self.cell_thr}: {total} / {self.zdf.size} "
            f"({pct:.2%})\n"
        )

    def _session_level_str(self) -> str:
        mask = self.zdf.abs() >= self.cell_thr
        counts = mask.sum(axis=1).sort_values(ascending=False)
        lines = [
            "\n===== Session-level =====",
            f"(sessions with ≥ {self.sub_thr} ROIs |z| ≥ {self.cell_thr})",
        ]
        for sid, cnt in counts.items():
            if cnt >= self.sub_thr:
                rois = self.zdf.columns[mask.loc[sid]]
                zs = self.zdf.loc[sid, rois]
                roi_str = ", ".join(
                    f"{r}({zs[r]:+.1f})" for r in rois[: self.max_roi_show]
                )
                tail = " ..." if len(rois) > self.max_roi_show else ""
                lines.append(f"{sid:30s} {cnt:3d} ROIs → {roi_str}{tail}")
                self.prob_subjectid.append(sid)
        return "\n".join(lines) + "\n"

    def _roi_level_str(self) -> str:
        mask = self.zdf.abs() >= self.cell_thr
        frac = mask.mean().sort_values(ascending=False)
        lines = [
            "\n===== ROI-level =====",
            f"(ROIs in ≥ {self.roi_frac_thr:.0%} subjects with |z| ≥ {self.cell_thr})",
        ]
        for roi, f in frac.items():
            if f >= self.roi_frac_thr:
                n = int(f * len(self.zdf))
                lines.append(f"{roi:25s} {n:3d}/{len(self.zdf):3d} subjects ({f:.1%})")
                self.prob_rois.append(roi)
        return "\n".join(lines) + "\n"

    def generate_report(self) -> str:
        parts = [
            f"Z-Score Analysis Report\n{'='*30}",
            self._cell_level_str(),
            self._session_level_str(),
            self._roi_level_str(),
        ]
        return "\n".join(parts)

    def visualize_subjects(self) -> None:
        matplotlib.use("Agg")
        out_path = self.out_dir / "prob_sessions"
        out_path.mkdir(parents=True, exist_ok=True)

        for subjectid in self.prob_subjectid:
            print(f"\n=== Visualizing subject: {subjectid} ===")

            viz_kwargs = {
                "subjectid": subjectid,
                "root": self.root_dir,
                "bg_t1_file": True,
                "show_img": False,
            }
            arr1 = visualize_slant_subjectid(**viz_kwargs)
            viz2_kwargs = {k: v for k, v in viz_kwargs.items() if k != "bg_t1_file"}
            arr2 = visualize_t1w_subjectid(**viz2_kwargs)

            combined = np.vstack([arr1, arr2])
            out_file = out_path / f"{subjectid}.png"
            iio.imwrite(out_file, combined)
            print(f"Saved combined image for {subjectid} at {out_file}")

    def plot_roi_distribution(self) -> None:
        out_path = self.out_dir / "prob_rois"
        out_path.mkdir(parents=True, exist_ok=True)

        for roi in self.prob_rois:
            print(f"\n=== Plotting Regions of Interest: {roi} ===")
            if roi not in self.vdf.columns:
                print(f"[WARN] ROI '{roi}' not in vdf, skip.")
                continue
            roi_name = self.idx2labeldf[int(roi[1:])]
            plot_distribution(
                self.vdf[roi].values,
                name=roi_name,
                ylabel=f"{roi_name} (mL)",
                png_name=f"{roi_name}.png",
                out_dir=out_path,
            )
            print(f"Saved distribution image for {roi_name} at {out_path}")

    def run(self):
        out_txt = self.out_dir / "outliers.txt"
        out_txt.parent.mkdir(exist_ok=True, parents=True)
        report = self.generate_report()
        out_txt.write_text(report, encoding="utf-8")
        print(f"Report saved to {out_txt}")
