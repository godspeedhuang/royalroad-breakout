"""每一季前 10% 的中心點在 PCA 平面上的軌跡（靜態圖）。

PCA 是線性的，2D 座標的平均就是 1536 維中心點的投影，所以中心點位置是正確的。
輸出：out/figs/pca_quarter_trajectory.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "out"
plt.rcParams.update({"font.sans-serif": ["PingFang TC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"], "axes.unicode_minus": False})

a = pd.read_parquet(OUT / "atlas.parquet", columns=["quarter", "year", "top10", "pca_x", "pca_y"])
ev = json.load(open(OUT / "projection_summary.json"))["pca_explained"]
axis = json.load(open(OUT / "drift_summary.json"))["axis_tags"]

def path(df):
    return df.groupby("quarter").agg(x=("pca_x", "mean"), y=("pca_y", "mean"), n=("pca_x", "size")).sort_index()

top, allb = path(a[a.top10 == "前 10%"]), path(a)
t = np.linspace(0, 1, len(top))

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
ax.hexbin(a.pca_x, a.pca_y, gridsize=70, cmap="Greys", bins="log", mincnt=1, alpha=.6)
ax.plot(top.x, top.y, color="#c8553d", lw=1.5); ax.plot(allb.x, allb.y, color="#2f6f73", lw=1.5)
ax.set_title("全部書（灰）與兩條中心點軌跡"); ax.set_xlabel(f"PC1（{ev[0]:.1%}）"); ax.set_ylabel(f"PC2（{ev[1]:.1%}）")

ax = axes[1]   # 放大看軌跡
sc = ax.scatter(top.x, top.y, c=t, cmap="viridis", s=18 + top.n / 6, zorder=3, label="每季前 10% 中心點（大小 = 本數）")
for i in range(len(top) - 1):
    ax.annotate("", xy=(top.x.iloc[i + 1], top.y.iloc[i + 1]), xytext=(top.x.iloc[i], top.y.iloc[i]),
                arrowprops=dict(arrowstyle="->", color=plt.cm.viridis(t[i]), lw=1.2))
ax.plot(allb.x, allb.y, color="#2f6f73", lw=1, ls="--", alpha=.7, label="每季全部書的中心點")
for q in top.index:
    if q.endswith("Q1") or q == top.index[-1]:
        ax.annotate(q[2:], (top.loc[q, "x"], top.loc[q, "y"]), fontsize=9, xytext=(5, 4), textcoords="offset points")
fig.colorbar(sc, ax=ax, label="時間（2016Q1 → 2024Q4）", ticks=[])
ax.set_title("放大：每季前 10% 的中心點怎麼移動"); ax.legend(fontsize=8, loc="best")
ax.set_xlabel("PC1：高 = " + "、".join(list(axis["PC1"]["高的一端"])[:3]) + "；低 = " + "、".join(list(axis["PC1"]["低的一端"])[:3]))
ax.set_ylabel("PC2：高 = " + "、".join(list(axis["PC2"]["高的一端"])[:2]) + "；低 = " + "、".join(list(axis["PC2"]["低的一端"])[:2]))
plt.tight_layout()
(OUT / "figs").mkdir(exist_ok=True)
plt.savefig(OUT / "figs" / "pca_quarter_trajectory.png", dpi=140, bbox_inches="tight")
print("saved; 前 10% 中心點 2016Q1→2024Q4 移動：", (top[["x", "y"]].iloc[-1] - top[["x", "y"]].iloc[0]).round(3).to_dict())
