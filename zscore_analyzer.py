"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

import pandas as pd
import re

from pathlib import Path
from typing import Union
from visualization import *


class ZScoreAnalyzer:
    def __init__(
        self,
        out_dir: Union[str, Path],
        cell_thr: float = 3.0,
        sub_thr: int = 5,
        roi_frac_thr: float = 0.05,
        max_roi_show: int = 135,
    ):
        """
        zscore_csv: path to z-score CSV (must have 'subject' column and ROI columns)
        cell_thr: |z| threshold for individual cells
        sub_thr: minimum number of ROIs per session to flag
        roi_frac_thr: minimum fraction of subjects per ROI to flag
        max_roi_show: how many ROIs to display per session
        """
        self.out_dir = Path(out_dir)
        self.roi_zscore_csv = Path(out_dir) / "stats_csv" / "roi_volumes_zscore.csv"
        self.cell_thr = cell_thr
        self.sub_thr = sub_thr
        self.roi_frac_thr = roi_frac_thr
        self.max_roi_show = max_roi_show
        self.prob_subjectid = []

        df = pd.read_csv(self.roi_zscore_csv)
        self.zdf = df.set_index("subject").apply(pd.to_numeric)

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
        return "\n".join(lines) + "\n"

    def generate_report(self) -> str:
        parts = [
            f"Z-Score Analysis Report\n{'='*30}",
            self._cell_level_str(),
            self._session_level_str(),
            self._roi_level_str(),
        ]
        return "\n".join(parts)

    def run(self):
        out_txt = self.out_dir / "outliers.txt"
        out_txt.parent.mkdir(exist_ok=True, parents=True)
        report = self.generate_report()
        out_txt.write_text(report, encoding="utf-8")
        print(f"Report saved to {out_txt}")

    def visualize_subjects(self, slant_root: Union[str, Path]) -> None:
        out_path = self.out_dir / "prob_images"
        out_path.mkdir(parents=True, exist_ok=True)

        for subjectid in self.prob_subjectid:
            print(f"\n=== Visualizing subject: {subjectid} ===")
            viz_kwargs = {
                "subjectid": subjectid,
                "root": slant_root,
                "bg_t1_file": True,
                "show_img": False,
            }
            fig1 = visualize_slant_subjectid(**viz_kwargs)
            viz2_kwargs = {k: v for k, v in viz_kwargs.items() if k != "bg_t1_file"}
            fig2 = visualize_t1w_subjectid(**viz2_kwargs)
            arr1 = self._fig_to_array(fig1)
            arr2 = self._fig_to_array(fig2)
            combined_fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(12, 8),
                constrained_layout=True,
            )
            ax1.imshow(arr1)
            ax1.set_title("Segmentation Overlay", fontsize=10)
            ax1.axis("off")
            ax2.imshow(arr2)
            ax2.set_title("T1w Anatomy", fontsize=10)
            ax2.axis("off")

            combined_file = out_path / f"{subjectid}.png"
            combined_fig.savefig(combined_file, dpi=300, bbox_inches="tight")
            plt.close(combined_fig)
            plt.close(fig1)
            plt.close(fig2)

            print(f"Saved combined image for {subjectid} at {combined_file}")

    @staticmethod
    def _fig_to_array(fig: plt.Figure) -> np.ndarray:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)
        return buf
