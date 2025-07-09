import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

root_dir = cfg["root_dir"]
pattern = cfg["pattern"]
ai_pairs_csv = cfg["ai_pairs_csv"]
ticv_csv = cfg["ticv_csv"]

pairs = pd.read_csv("labels/slant_ai_pairs.csv", usecols=["RightID", "LeftID"])
PAIRS = [tuple(row) for row in pairs.to_numpy()]

paths = list(Path(root_dir).glob(pattern))
ai_rows = []

for seg_path in tqdm(paths, desc="Computing Asymmetry Index", total=len(paths)):
    subject = seg_path.parents[4].name
    session = seg_path.parents[3].name
    sid = f"{subject}_{session}"

    img = nib.load(seg_path)
    seg = img.get_fdata().astype(np.int32)
    vox = np.prod(img.header.get_zooms()[:3]) / 1e3

    row = {"subject": sid}
    for L, R in PAIRS:
        vL = (seg == L).sum() * vox
        vR = (seg == R).sum() * vox
        ai = (vL - vR) / (vL + vR + 1e-6)
        row[f"AI_{L}-{R}"] = ai
    ai_rows.append(row)

aidf = pd.DataFrame(ai_rows).set_index("subject").sort_index()
aidf.to_csv("stats_csv/asymmetry_index.csv")

ai_outliers = (aidf.abs() > 0.15).sum().sum()
print(aidf.iloc[:5, :8])
print(f"\nTotal AI outlier cells (|AI|>0.15) = {ai_outliers}")
print("CSV saved: asymmetry_index.csv")

# Visualization Part
# ==============================================================
pairs_df = pd.read_csv(
    ai_pairs_csv,
    usecols=["RightID", "LeftID", "ROI"],
)

map_dict = {f"AI_{r}-{l}": roi for r, l, roi in pairs_df.itertuples(index=False)}

aidf = aidf.rename(columns=map_dict)
aidf = aidf[[c for c in map_dict.values() if c in aidf.columns]]

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(aidf.values, aspect="auto", cmap="coolwarm", vmin=-0.2, vmax=0.2)
plt.colorbar(im, ax=ax, label="AI")

ax.set_ylabel("Subject")
ax.set_xticks(np.arange(aidf.shape[1]))
ax.set_xticklabels(aidf.columns, rotation=90, fontsize=6)
ax.set_xlabel("ROI (Left vs. Right)")

plt.tight_layout()
plt.savefig("stats_png/ai_heatmap.png", dpi=300)
plt.close()

# ==============================================================
sub_flags = (aidf.abs() > 0.15).sum(axis=1)
sub_flags.sort_values(ascending=False).head(20).plot.bar(figsize=(9, 4))
plt.axhline(5, color="r", ls="--")
plt.ylabel("# ROI |AI|>0.15")
plt.title("Top subjects by AI outlier count")
plt.tight_layout()
plt.savefig("stats_png/ai_subject_bar.png", dpi=300)
plt.close()

# ==============================================================
top_cols = aidf.abs().mean().sort_values(ascending=False).head(10).index
sns.violinplot(
    data=aidf[top_cols].melt(var_name="ROI", value_name="AI"),
    x="ROI",
    y="AI",
    inner="box",
)
plt.axhline(0.15, color="r", ls="--")
plt.axhline(-0.15, color="r", ls="--")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("stats_png/ai_top_roi_violin.png", dpi=300)
plt.close()

# ==============================================================
ticv = pd.read_csv(ticv_csv).set_index("subject")["zscore"].reindex(aidf.index)
plt.scatter(ticv, aidf.abs().mean(axis=1), alpha=0.7)
plt.axhline(0.15, color="r", ls="--")
plt.axvline(2.8, color="r", ls="--")
plt.axvline(-2.8, color="r", ls="--")
plt.xlabel("TICV z-score")
plt.ylabel("Mean |AI| across ROI")
plt.title("Mean |AI| vs TICV z-score")
plt.tight_layout()
plt.savefig("stats_png/ai_vs_ticv_scatter.png", dpi=300)
plt.close()
