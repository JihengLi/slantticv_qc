#!/usr/bin/env python3
"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

import argparse
import logging
import sys
from pathlib import Path

from roi_volume_calculator import ROIVolumeCalculator
from volume_qc import VolumeQC
from zscore_analyzer import ZScoreAnalyzer


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="ROI QA pipeline")
    p.add_argument(
        "--slant-root",
        type=Path,
        required=True,
        help="Root directory of SLANT-TICV outputs",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where CSVs, report, and QA figures will be written",
    )
    p.add_argument(
        "--label-index",
        type=Path,
        default=Path("labels/label_index.csv"),
        help="CSV of ROI labels (must contain column 'IDX')",
    )
    p.add_argument(
        "--cell-thr",
        type=float,
        default=3.0,
        help="Threshold for |z|-score at the cell level",
    )
    p.add_argument(
        "--sub-thr",
        type=int,
        default=5,
        help="Min number of ROIs over cell-thr to flag a session",
    )
    p.add_argument(
        "--roi-frac-thr",
        type=float,
        default=0.05,
        help="Min fraction of subjects per ROI to flag at ROI level",
    )
    p.add_argument(
        "--max-roi-show",
        type=int,
        default=135,
        help="Max number of ROIs to list per session in the report",
    )
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    if not args.slant_root.is_dir():
        logging.error(f"SLANT root not found: {args.slant_root}")
        sys.exit(1)
    if args.slant_root.name != "derivatives":
        logging.error(
            f"SLANT root must be the 'derivatives' folder, "
            f"but got: '{args.slant_root.name}'"
        )
        sys.exit(1)
    if not args.label_index.is_file():
        logging.error(f"Label index CSV not found: {args.label_index}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("---> Computing ROI volumes …")
    calc = ROIVolumeCalculator(
        slant_ticv_root=args.slant_root,
        out_dir=args.out_dir,
        pattern="sub-*/ses-*/SLANT-TICVv1.2*/post/FinalResult/*_T1w_seg.nii.gz",
        label_index=args.label_index,
    )
    calc.run()

    logging.info("---> Running z-score analysis …")
    analyzer = ZScoreAnalyzer(
        out_dir=args.out_dir,
        cell_thr=args.cell_thr,
        sub_thr=args.sub_thr,
        roi_frac_thr=args.roi_frac_thr,
        max_roi_show=args.max_roi_show,
    )
    analyzer.run()

    logging.info("---> Visualizing flagged subjects …")
    analyzer.visualize_subjects(args.slant_root)

    logging.info("---> Computing brain & TICV QC …")
    volqc = VolumeQC(
        out_dir=args.out_dir,
    )
    volqc.run()
    logging.info("---> Pipeline finished.")


if __name__ == "__main__":
    main()
