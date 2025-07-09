# nohup python brain_volume.py > logs/out_brain_volumes.log 2>&1 &

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats as st
import yaml

from pathlib import Path
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

label_df = pd.read_csv(Path(cfg["label_index"]), usecols=["IDX"])

exclude_ids = set(label_df["IDX"].tail(2))
exclude_ids.add(0)

root_dir = cfg["root_dir"]
pattern = cfg["pattern"]
brainvol_csv = cfg["brainvol_csv"]

paths = list(Path(root_dir).glob(pattern))
rows = []

for seg_path in tqdm(paths, desc="Computing Brain Volume", total=len(paths)):
    subject = seg_path.parents[4].name
    session = seg_path.parents[3].name
    sid = f"{subject}_{session}"

    img = nib.load(seg_path)
    arr = img.get_fdata().astype(np.int32)
    vox = np.prod(img.header.get_zooms()[:3])
    brain_voxels = np.isin(arr, list(exclude_ids), invert=True).sum()
    brain_ml = brain_voxels * vox / 1e3

    rows.append({"subject": sid, "BrainVol_ml": brain_ml})

df = pd.DataFrame(rows).sort_values("subject")
df["zscore"] = st.zscore(df["BrainVol_ml"])
df.to_csv("stats_csv/brain_volume.csv", index=False)

print(df.head())
print(f"\nCSV saved: brain_volume.csv")

df = pd.read_csv(brainvol_csv)
brainvol_series = df["BrainVol_ml"].values

fig = plt.figure(constrained_layout=True, figsize=(15, 8))
gs = GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
parts = ax1.violinplot(brainvol_series, showmeans=False, showmedians=True, widths=0.6)
for pc in parts["bodies"]:
    pc.set_alpha(0.6)
ax1.boxplot(brainvol_series, widths=0.15, vert=True, showfliers=False, positions=[1])
ax1.set_ylabel("brainvol (mL)")
ax1.set_xticks([])
ax1.set_title("Violin")


ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(brainvol_series, bins=30, alpha=0.6, density=True, label="Histogram")
kde_x = np.linspace(brainvol_series.min(), brainvol_series.max(), 200)
kde_y = st.gaussian_kde(brainvol_series)(kde_x)
ax2.plot(kde_x, kde_y, linewidth=2, label="KDE")
ax2.axvline(brainvol_series.mean(), linestyle="--", label="Mean")
ax2.set_xlabel("brainvol (mL)")
ax2.set_ylabel("Density")
ax2.set_title("Histogram")
ax2.legend()
fig.suptitle("brainvol Distribution QC", fontsize=14)
fig.savefig("stats_png/brainvol_violin_hist.png", dpi=300)
plt.close(fig)
