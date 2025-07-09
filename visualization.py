import nibabel as nib
import numpy as np
import yaml

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from pathlib import Path
from collections.abc import Sequence

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

root_dir = cfg["root_dir"]
t1_root_dir = cfg["t1_root_dir"]
lut_addr = cfg["lut_addr"]


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


def visualize_slant(
    seg_file: str | Path,
    lut_file: str | Path,
    sagittal_slices: int | str | Sequence[int | str] = "mid",
    coronal_slices: int | str | Sequence[int | str] = "mid",
    axial_slices: int | str | Sequence[int | str] = "mid",
    t1_file: str | Path | None = None,
    alpha_seg: float = 0.6,
    save_path: Path | None = None,
) -> None:
    seg_img = nib.load(seg_file, mmap=True)
    seg_data = seg_img.dataobj
    cmap, norm = load_lut(lut_file, bg_transparent=t1_file != None)

    if t1_file:
        t1_data = nib.load(t1_file, mmap=True).dataobj
        if t1_data.shape != seg_data.shape:
            raise ValueError("T1 shape ≠ seg shape")
    else:
        t1_data = None
        alpha_seg = 1.0

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
        plt.close(fig)
    else:
        plt.show()


def visualize_slant_subjectid(
    subjectid: str,
    sagittal_slices: int | str | Sequence[int | str] = "mid",
    coronal_slices: int | str | Sequence[int | str] = "mid",
    axial_slices: int | str | Sequence[int | str] = "mid",
    bg_t1_file: bool = False,
    alpha_seg: float = 0.6,
    save_path: Path | None = None,
):
    sub, ses = subjectid.split("_")
    seg_file = (
        Path(root_dir)
        / sub
        / ses
        / "SLANT-TICVv1.2/post/FinalResult"
        / f"{subjectid}_T1w_seg.nii.gz"
    )
    t1_file = None
    if bg_t1_file:
        t1_file = Path(t1_root_dir) / sub / ses / "anat" / f"{subjectid}_T1w.nii.gz"
    visualize_slant(
        seg_file,
        lut_addr,
        sagittal_slices,
        coronal_slices,
        axial_slices,
        t1_file,
        alpha_seg,
        save_path,
    )


def visualize_t1w_subjectid(
    subjectid: str,
    root: str | Path = t1_root_dir,
    sagittal_slices: int | str | Sequence[int | str] = "mid",
    coronal_slices: int | str | Sequence[int | str] = "mid",
    axial_slices: int | str | Sequence[int | str] = "mid",
    perc: tuple[float, float] = (1, 99),
    save_path: Path | None = None,
) -> None:
    root = Path(root)
    sub, ses = subjectid.split("_")
    t1_file = root / sub / ses / "anat" / f"{subjectid}_T1w.nii.gz"
    if not t1_file.exists():
        raise FileNotFoundError(t1_file)

    img = nib.load(t1_file, mmap=True)
    dataobj = img.dataobj

    x_idxs = _normalize_slices(sagittal_slices, dataobj.shape[0])
    y_idxs = _normalize_slices(coronal_slices, dataobj.shape[1])
    z_idxs = _normalize_slices(axial_slices, dataobj.shape[2])

    combos = [(x, y, z) for x in x_idxs for y in y_idxs for z in z_idxs]
    n_rows = len(combos)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, 3)
    vmin, vmax = np.percentile(dataobj, perc)

    for row_idx, (x, y, z) in enumerate(combos):
        slices = (
            np.rot90(dataobj[x, :, :]),
            np.rot90(dataobj[:, y, :]),
            np.rot90(dataobj[:, :, z]),
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
        plt.close(fig)
    else:
        plt.show()
