"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

import nibabel as nib
import numpy as np
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from pathlib import Path
from collections.abc import Sequence
from typing import List, Union
from nibabel.orientations import (
    io_orientation,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
)
from bids_addr import *

lut_addr = "labels/slant.label"


def load_lut(lut_path: str, bg_transparent: bool = True):
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

    if bg_transparent and 0 <= max_idx:
        full_rgba[0] = (0, 0, 0, 0)

    cmap = ListedColormap(full_rgba, name="slant_lut")
    bounds = np.arange(max_idx + 2) - 0.5
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
    return cmap, norm


def _normalize_slices(
    val: int | str | Sequence[int | str], size: int
) -> tuple[int, ...]:
    if isinstance(val, (str, int)):
        val = (val,)
    idxs = []
    for v in val:
        idx = size // 2 if v == "mid" else int(v)
        if not 0 <= idx < size:
            raise ValueError(f"Slice index {idx} out of bounds (0‥{size-1})")
        idxs.append(idx)
    return tuple(idxs)


def _keep_roi(arr: np.ndarray, roi: List) -> np.ndarray:
    mask = np.isin(arr, roi)
    out = np.where(mask, arr, 0)
    return out


def _load_as_ras(path):
    img = nib.load(path, mmap=True)
    data = img.get_fdata(dtype=np.float32)
    src = io_orientation(img.affine)
    tgt = axcodes2ornt(("R", "A", "S"))
    trf = ornt_transform(src, tgt)
    return apply_orientation(data, trf)


def visualize_slant(
    seg_file: str | Path,
    lut_file: str | Path,
    sagittal_slices: int | str | Sequence[int | str] = "mid",
    coronal_slices: int | str | Sequence[int | str] = "mid",
    axial_slices: int | str | Sequence[int | str] = "mid",
    keep_roi_list: List | None = None,
    auto_slice: bool = False,
    t1_file: str | Path | None = None,
    alpha_seg: float = 0.6,
    save_path: Path | None = None,
    show_img: bool = True,
) -> plt.Figure:
    seg_data = _load_as_ras(seg_file)
    cmap, norm = load_lut(lut_file, bg_transparent=t1_file != None)

    if t1_file:
        t1_data = _load_as_ras(t1_file)
        if t1_data.shape != seg_data.shape:
            raise ValueError("T1 shape ≠ seg shape")
    else:
        t1_data = None
        alpha_seg = 1.0

    if auto_slice:
        if keep_roi_list is None:
            raise ValueError("auto_slice=True requires keep_roi_list to be provided")
        if sagittal_slices != "mid" or coronal_slices != "mid" or axial_slices != "mid":
            warnings.warn("auto_slice=True: provided slices ignored.", UserWarning)

        def _best_slice_along_axis(axis):
            """Return a dict {roi: best_idx} and a fallback idx (max area)."""
            best_for_roi = {}
            max_area_idx, max_area = -1, -1

            for i in range(seg_data.shape[axis]):
                if axis == 0:
                    slc = seg_data[i, :, :]
                elif axis == 1:
                    slc = seg_data[:, i, :]
                else:
                    slc = seg_data[:, :, i]

                roi_counts = {
                    roi: np.count_nonzero(slc == roi) for roi in keep_roi_list
                }
                total = sum(roi_counts.values())

                if total > max_area:
                    max_area, max_area_idx = total, i

                for roi, cnt in roi_counts.items():
                    if cnt == 0:
                        continue
                    if roi not in best_for_roi or cnt > best_for_roi[roi][1]:
                        best_for_roi[roi] = (i, cnt)

            best_for_roi = {roi: idx_cnt[0] for roi, idx_cnt in best_for_roi.items()}
            return best_for_roi, max_area_idx

        best_x_roi, fallback_x = _best_slice_along_axis(axis=0)
        best_y_roi, fallback_y = _best_slice_along_axis(axis=1)
        best_z_roi, fallback_z = _best_slice_along_axis(axis=2)

        def _pick(best_map, fallback):
            if len(best_map) == len(keep_roi_list):
                idx = max(
                    best_map.values(),
                    key=lambda i: (
                        np.count_nonzero(
                            np.isin(
                                (
                                    seg_data[i, :, :]
                                    if best_map is best_x_roi
                                    else (
                                        seg_data[:, i, :]
                                        if best_map is best_y_roi
                                        else seg_data[:, :, i]
                                    )
                                ),
                                keep_roi_list,
                            )
                        )
                    ),
                )
            elif best_map:
                counts = {
                    idx: np.count_nonzero(
                        np.isin(
                            (
                                seg_data[idx, :, :]
                                if best_map is best_x_roi
                                else (
                                    seg_data[:, idx, :]
                                    if best_map is best_y_roi
                                    else seg_data[:, :, idx]
                                )
                            ),
                            keep_roi_list,
                        )
                    )
                    for idx in best_map.values()
                }
                idx = max(counts, key=counts.get)
            else:
                idx = fallback
            return int(idx)

        best_x = _pick(best_x_roi, fallback_x)
        best_y = _pick(best_y_roi, fallback_y)
        best_z = _pick(best_z_roi, fallback_z)

        sagittal_slices = [best_x]
        coronal_slices = [best_y]
        axial_slices = [best_z]

    x_idxs = _normalize_slices(sagittal_slices, seg_data.shape[0])
    y_idxs = _normalize_slices(coronal_slices, seg_data.shape[1])
    z_idxs = _normalize_slices(axial_slices, seg_data.shape[2])

    combos = [(x, y, z) for x in x_idxs for y in y_idxs for z in z_idxs]
    n_rows = len(combos)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, 3)

    for row_idx, (x, y, z) in enumerate(combos):
        slices_seg = (
            np.rot90(seg_data[x, :, :]),
            np.rot90(seg_data[:, y, :]),
            np.rot90(seg_data[:, :, z]),
        )
        if keep_roi_list:
            slices_seg = tuple(_keep_roi(slc, keep_roi_list) for slc in slices_seg)
        if t1_data is not None:
            slices_t1 = (
                np.rot90(t1_data[x, :, :]),
                np.rot90(t1_data[:, y, :]),
                np.rot90(t1_data[:, :, z]),
            )
        titles = (
            f"Sagittal X={x}",
            f"Coronal  Y={y}",
            f"Axial    Z={z}",
        )
        for col_idx, (seg_slc, title) in enumerate(zip(slices_seg, titles)):
            ax = axes[row_idx, col_idx]
            if t1_data is not None:
                ax.imshow(slices_t1[col_idx], cmap="gray", interpolation="nearest")
            ax.imshow(
                seg_slc, cmap=cmap, norm=norm, interpolation="nearest", alpha=alpha_seg
            )
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_img:
        plt.show()
    else:
        plt.close(fig)
    return fig


def visualize_slant_subjectid(
    subjectid: str,
    root: str | Path,
    lut_file: str | Path = lut_addr,
    sagittal_slices: int | str | Sequence[int | str] = "mid",
    coronal_slices: int | str | Sequence[int | str] = "mid",
    axial_slices: int | str | Sequence[int | str] = "mid",
    keep_roi_list: List | None = None,
    auto_slice: bool = False,
    bg_t1_file: bool = False,
    alpha_seg: float = 0.6,
    save_path: Path | None = None,
    show_img: bool = True,
) -> plt.Figure:
    seg_file = find_slant_addr_subjectid(subjectid, Path(root))[0]
    t1_file = None
    if bg_t1_file:
        t1_file = find_t1w_addr_subjectid(subjectid, Path(root).parent)[0]
    return visualize_slant(
        seg_file,
        lut_file,
        sagittal_slices,
        coronal_slices,
        axial_slices,
        keep_roi_list,
        auto_slice,
        t1_file,
        alpha_seg,
        save_path,
        show_img,
    )


def visualize_t1w(
    t1_file: str | Path,
    sagittal_slices: int | str | Sequence[int | str] = "mid",
    coronal_slices: int | str | Sequence[int | str] = "mid",
    axial_slices: int | str | Sequence[int | str] = "mid",
    perc: tuple[float, float] = (1, 99),
    save_path: Path | None = None,
    show_img: bool = True,
) -> plt.Figure:
    t1_file = Path(t1_file)
    if not t1_file.exists():
        raise FileNotFoundError(t1_file)

    data_ras = _load_as_ras(t1_file)

    x_idxs = _normalize_slices(sagittal_slices, data_ras.shape[0])
    y_idxs = _normalize_slices(coronal_slices, data_ras.shape[1])
    z_idxs = _normalize_slices(axial_slices, data_ras.shape[2])

    combos = [(x, y, z) for x in x_idxs for y in y_idxs for z in z_idxs]
    n_rows = len(combos)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, 3)
    vmin, vmax = np.percentile(data_ras, perc)

    for row_idx, (x, y, z) in enumerate(combos):
        slices = (
            np.rot90(data_ras[x, :, :]),
            np.rot90(data_ras[:, y, :]),
            np.rot90(data_ras[:, :, z]),
        )
        titles = (
            f"Sagittal X={x}",
            f"Coronal  Y={y}",
            f"Axial    Z={z}",
        )
        for col_idx, (slc, title) in enumerate(zip(slices, titles)):
            ax = axes[row_idx, col_idx]
            ax.imshow(slc, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_img:
        plt.show()
    else:
        plt.close(fig)
    return fig


def visualize_t1w_subjectid(
    subjectid: str,
    root: str | Path,
    sagittal_slices: int | str | Sequence[int | str] = "mid",
    coronal_slices: int | str | Sequence[int | str] = "mid",
    axial_slices: int | str | Sequence[int | str] = "mid",
    perc: tuple[float, float] = (1, 99),
    save_path: Path | None = None,
    show_img: bool = True,
) -> plt.Figure:
    t1_file = find_t1w_addr_subjectid(subjectid, Path(root).parent)[0]
    return visualize_t1w(
        t1_file,
        sagittal_slices,
        coronal_slices,
        axial_slices,
        perc,
        save_path,
        show_img,
    )


def visualize_ct(
    ct_file: Union[str, Path],
    sagittal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    coronal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    axial_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    window: tuple[int, int] = (-1024, 1024),
    save_path: Union[str, Path] | None = None,
    show_img: bool = True,
) -> plt.Figure:
    ct_file = Path(ct_file)
    if not ct_file.exists():
        raise FileNotFoundError(f"CT file not found: {ct_file}")

    img = nib.load(ct_file)
    img = nib.as_closest_canonical(img)
    data_ras = _load_as_ras(ct_file)
    sx, sy, sz = img.header.get_zooms()[:3]

    x_idxs = _normalize_slices(sagittal_slices, data_ras.shape[0])
    y_idxs = _normalize_slices(coronal_slices, data_ras.shape[1])
    z_idxs = _normalize_slices(axial_slices, data_ras.shape[2])
    combos = [(x, y, z) for x in x_idxs for y in y_idxs for z in z_idxs]
    n = len(combos)

    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    axes = np.asarray(axes).reshape(n, 3)
    vmin, vmax = window

    orient_titles = ("Sagittal", "Coronal", "Axial")
    for i, (x, y, z) in enumerate(combos):
        sl_sag = np.rot90(data_ras[x, :, :])
        sl_cor = np.rot90(data_ras[:, y, :])
        sl_axi = np.rot90(data_ras[:, :, z])
        slices = (sl_sag, sl_cor, sl_axi)

        for j, slc in enumerate(slices):
            ax = axes[i, j]
            ax.imshow(slc, cmap="bone", vmin=vmin, vmax=vmax, interpolation="nearest")
            if j == 0:
                aspect = sz / sy
            elif j == 1:
                aspect = sz / sx
            else:
                aspect = sy / sx

            ax.set_aspect(aspect)
            ax.set_title(
                f"{orient_titles[j]} ({['X','Y','Z'][j]}={[x,y,z][j]})", fontsize=9
            )
            ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_img:
        plt.show()
    else:
        plt.close(fig)
    return fig
