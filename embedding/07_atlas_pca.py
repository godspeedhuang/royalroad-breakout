"""PCA 版的 Embedding Atlas：在書的點之外，加上每一季前 10% 的中心點。

PCA 是線性的，所以「2D 座標取平均」剛好等於「1536 維中心點的 PCA 投影」，中心點的位置是正確的。
（UMAP 是非線性的，直接平均 2D 座標沒有這個性質，所以中心點只做在 PCA 版。）

輸出（out/）：
  atlas_pca.parquet         書 + 36 個季中心點（kind 欄位區分）
  atlas_pca_labels.parquet  中心點的文字標籤（x、y、text），取代自動分群標籤

用法：python 07_atlas_pca.py，再執行 08_atlas_pca.sh
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "out"

a = pd.read_parquet(OUT / "atlas.parquet")
a.insert(0, "kind", "書")

top = a[a.top10 == "前 10%"]
cent = top.groupby("quarter").agg(pca_x=("pca_x", "mean"), pca_y=("pca_y", "mean"), n=("fiction_id", "size"),
                                  year=("year", "first")).reset_index()
cent["label"] = cent.quarter.str.slice(2, 4) + cent.quarter.str.slice(4)   # 2024Q4 → 24Q4

rows = pd.DataFrame({
    "kind": "季中心點（前 10%）",
    "title": "中心點 " + cent.quarter + "（" + cent.n.astype(str) + " 本）",
    "year": cent.year.astype(int), "quarter": cent.quarter, "top10": "季中心點",
    "pca_x": cent.pca_x.astype("float32"), "pca_y": cent.pca_y.astype("float32"),
    "neighbors": [{"ids": [], "distances": []} for _ in range(len(cent))],
})
out = pd.concat([a, rows], ignore_index=True)   # 中心點接在最後，書的列號不變，最近鄰的 id 仍然正確
out.to_parquet(OUT / "atlas_pca.parquet", index=False)

labels = pd.DataFrame({"x": cent.pca_x.astype(float), "y": cent.pca_y.astype(float), "text": cent.label.astype(object), "level": 0,
                       "priority": np.arange(len(cent))})   # 越新的季度優先顯示
labels.to_parquet(OUT / "atlas_pca_labels.parquet", index=False)

spread = a[["pca_x", "pca_y"]].std()
move = cent[["pca_x", "pca_y"]].iloc[-1] - cent[["pca_x", "pca_y"]].iloc[0]
print(f"季中心點 {len(cent)} 個，每季前 10% 本數 {cent.n.min()}–{cent.n.max()}")
print(f"書在 PCA 平面的標準差 x={spread.pca_x:.3f} y={spread.pca_y:.3f}；"
      f"2016Q1→2024Q4 中心點移動 x={move.pca_x:+.3f} y={move.pca_y:+.3f}")
