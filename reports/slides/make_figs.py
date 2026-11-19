"""簡報用的圖：UMAP 上的熱門書分布（灰點 + 橘色熱門書，標幾個類型群）。

輸入 embedding/out/atlas.parquet（UMAP 座標、tags、top10），輸出 reports/slides/img/umap_hits.png。
類型標籤放在該 tag 的書最密集的格子（40 × 40 的 2D 直方圖），軸範圍切掉兩端 0.5% 的離群點。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
GRAY, ACCENT, INK = "#BFBFBF", "#D9480F", "#111111"
LABELS = {"tag_LitRPG": "LitRPG", "tag_Romance": "Romance", "tag_Martial_Arts": "Martial arts", "tag_Horror": "Horror", "tag_Sci_fi": "Sci-fi"}

a = pd.read_parquet(ROOT / "embedding/out/atlas.parquet", columns=["umap_x", "umap_y", "top10", *LABELS])
a["top10"] = a.top10.eq("前 10%")  # atlas 的 top10 是顯示用的中文標籤
lo, hi = a[["umap_x", "umap_y"]].quantile(0.005), a[["umap_x", "umap_y"]].quantile(0.995)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(a.umap_x[~a.top10], a.umap_y[~a.top10], s=1.2, c=GRAY, alpha=0.35, linewidths=0, rasterized=True)
ax.scatter(a.umap_x[a.top10], a.umap_y[a.top10], s=2, c=ACCENT, alpha=0.6, linewidths=0, rasterized=True)
rng2 = [[lo.umap_x, hi.umap_x], [lo.umap_y, hi.umap_y]]
for col, name in LABELS.items():
    s = a[a[col]]
    H, xe, ye = np.histogram2d(s.umap_x, s.umap_y, bins=40, range=rng2)
    i, j = np.unravel_index(H.argmax(), H.shape)
    ax.text((xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2, name, fontsize=14, fontweight="bold", color=INK, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
ax.set_xlim(lo.umap_x, hi.umap_x); ax.set_ylim(lo.umap_y, hi.umap_y)
ax.set_axis_off()
ax.scatter([], [], s=40, c=ACCENT, label=f"Future top-10% serials ({a.top10.sum():,})")
ax.scatter([], [], s=40, c=GRAY, label=f"Other serials ({(~a.top10).sum():,})")
ax.legend(loc="lower right", frameon=False, fontsize=12)
(HERE / "img").mkdir(exist_ok=True)
plt.savefig(HERE / "img" / "umap_hits.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", HERE / "img" / "umap_hits.png")
