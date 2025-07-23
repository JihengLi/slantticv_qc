# Statistical test methods for SLANT-TICV

This repository provides a command-line tool to perform a complete QA workflow on SLANT-TICV segmentation outputs, including:

1. **ROI Volume Computation** using `ROIVolumeCalculator`.
2. **Z-Score Analysis and outlier reporting** via `ZScoreAnalyzer`.
3. **Subject-Level Visualization** of segmentation overlays.
4. **Brain Volume & Total Intracranial Volume (TICV) QC** using `VolumeQC`.

Author: Jiheng Li (jiheng.li.1@vanderbilt.edu)

---

### Requirements

- Python 3.10+
- nibabel
- numpy
- pandas
- scipy
- tqdm
- matplotlib

## Installation

1. Clone this repository:

```
git clone <repo-url>
cd <repo-dir>
```

2. (Optional) Create and activate a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```
pip install nibabel numpy pandas scipy tqdm matplotlib
```

## Usage

slant_root and out_dir are **required** inputs. Other options are optional and have defaults.

```
python slant‑ticv_qc.py \
  --slant_root /path/to/Dataset/derivatives \
  --out_dir    /path/to/output \
  [--label_index labels/label_index.csv] \
  [--cell-thr 3.0] [--sub-thr 5] [--roi-frac-thr 0.05] [--max-roi-show 12]
```

- `--slant_root`: **required.** Path to the derivatives/ folder of a BIDS‑style dataset containing SLANT‑TICV outputs:
  ```
  Dataset/
    sub-<label>/
      ses-<label>/
        anat/…              # raw T1w images here
    derivatives/
      sub-<label>/
        ses-<label>/
          SLANT-TICVv1.2…/…_T1w_seg.nii.gz
  ```
- `--out_dir`: **required.** Directory where all CSVs, reports, and figures will be written.

- `--label_index`: Path to ROI label CSV (must contain column IDX; default: labels/label_index.csv).

- `--cell-thr`: Threshold for |z|-score at the cell level (default: 3.0).

- `--sub-thr`: Min number of ROIs above cell-thr to flag a session (default: 5).

- `--roi-frac-thr`: Min fraction of subjects per ROI to include in ROI‑level report (default: 0.05).

- `--max-roi-show`: Max number of ROIs to list per session in the report (default: 135).

## Outputs

After running, `out_dir` will contain:

```
out_dir/
├─ stats_csv/
│ ├─ roi_volumes.csv # raw ROI volumes by subject
│ ├─ roi_volumes_zscore.csv # z-scored ROI table
│ └─ brain_volume.csv # total brain volume table
│
├─ stats_png/
│ ├─ brainvol_violin_hist.png # brain volume distribution plot
│ └─ ticv_violin_hist.png # TICV distribution plot
│
├─ prob_images/ # segmentation overlays for outliers
└─ outliers.txt # three-level outlier report
```

## Extensibility

- Adjust thresholds and ROI‑label inputs via command‑line flags.

- Customize the glob `pattern` for locating segmentation files in `ROIVolumeCalculator`.

- Extend `VolumeQC` to add more QC metrics or plots.

---
