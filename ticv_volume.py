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

root_dir = cfg["root_dir"]
pattern = cfg["pattern"]

paths = list(Path(root_dir).glob(pattern))
rows = []

for seg_path in tqdm(paths, desc="Computing TICV"):
    subject = seg_path.parents[4].name
    session = seg_path.parents[3].name
    sid = f"{subject}_{session}"

    img = nib.load(seg_path)
    arr = img.get_fdata().astype(np.int32)
    vox = np.prod(img.header.get_zooms()[:3])
    ticv_ml = (arr == 208).sum() * vox / 1e3

    rows.append({"subject": sid, "TICV_ml": ticv_ml})

df = pd.DataFrame(rows).sort_values("subject")
df["zscore"] = st.zscore(df["TICV_ml"])
df.to_csv(cfg["ticv_csv"], index=False)

ticv_series = df["TICV_ml"].values

fig = plt.figure(constrained_layout=True, figsize=(15, 8))
gs = GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
parts = ax1.violinplot(ticv_series, showmeans=False, showmedians=True, widths=0.6)
for pc in parts["bodies"]:
    pc.set_alpha(0.6)
ax1.boxplot(ticv_series, widths=0.15, vert=True, showfliers=False, positions=[1])
ax1.set_ylabel("TICV (mL)")
ax1.set_xticks([])
ax1.set_title("Violin")


ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(ticv_series, bins=30, alpha=0.6, density=True, label="Histogram")
kde_x = np.linspace(ticv_series.min(), ticv_series.max(), 200)
kde_y = st.gaussian_kde(ticv_series)(kde_x)
ax2.plot(kde_x, kde_y, linewidth=2, label="KDE")
ax2.axvline(ticv_series.mean(), linestyle="--", label="Mean")
ax2.set_xlabel("TICV (mL)")
ax2.set_ylabel("Density")
ax2.set_title("Histogram")
ax2.legend()
fig.suptitle("TICV Distribution QC", fontsize=14)
fig.savefig("stats_png/ticv_violin_hist.png", dpi=300)
plt.close(fig)
