import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from pathlib import Path

root_dir = "/nfs2/harmonization/BIDS/WRAPnew/derivatives/"
lut_addr = "labels/slant.label"


def load_lut(lut_path: str, make_bg_transparent: bool = True):
    idx_list, rgba_list = [], []
    with open(lut_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            idx = int(parts[0])
            r, g, b = map(int, parts[1:4])
            a = float(parts[4])
            idx_list.append(idx)
            rgba_list.append((r / 255.0, g / 255.0, b / 255.0, a))

    max_idx = max(idx_list)
    full_rgba = [(0, 0, 0, 0)] * (max_idx + 1)
    for i, idx in enumerate(idx_list):
        full_rgba[idx] = rgba_list[i]

    if make_bg_transparent and 0 <= max_idx:
        full_rgba[0] = (0, 0, 0, 0)

    cmap = ListedColormap(full_rgba, name="slant_lut")
    bounds = np.arange(max_idx + 2) - 0.5
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
    return cmap, norm


def visualize_slant(seg_file: str, lut_file: str = lut_addr):
    img_obj = nib.load(seg_file)
    data = img_obj.get_fdata()
    cmap, norm = load_lut(lut_file, False)

    x_mid = data.shape[0] // 2
    y_mid = data.shape[1] // 2
    z_mid = data.shape[2] // 2

    slice_sagittal = data[x_mid, :, :]
    slice_coronal = data[:, y_mid, :]
    slice_axial = data[:, :, z_mid]

    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        np.rot90(slice_sagittal), cmap=cmap, norm=norm, interpolation="nearest"
    )
    axes[0].set_title(f"Sagittal (X = {x_mid})")
    axes[0].axis("off")

    axes[1].imshow(
        np.rot90(slice_coronal), cmap=cmap, norm=norm, interpolation="nearest"
    )
    axes[1].set_title(f"Coronal (Y = {y_mid})")
    axes[1].axis("off")

    axes[2].imshow(np.rot90(slice_axial), cmap=cmap, norm=norm, interpolation="nearest")
    axes[2].set_title(f"Axial (Z = {z_mid})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_slant_subjectid(subjectid: str):
    sub, ses = subjectid.split("_")
    seg_file = (
        Path(root_dir)
        / sub
        / ses
        / "SLANT-TICVv1.2/post/FinalResult"
        / f"{subjectid}_T1w_seg.nii.gz"
    )
    # seg_file = (
    #     Path(root_dir) / sub / ses /
    #     "SLANT-TICVv1.2/post/FinalResult/5000_fusion27_mv/finetune_out/seg_output/epoch_0034/seg_orig_final" /
    #     f"{subject}_T1w_seg.nii.gz"
    # )
    visualize_slant(seg_file, lut_addr)
