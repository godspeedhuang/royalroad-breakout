"""PCA version of the Embedding Atlas: on top of the books, add each quarter's top-10% centroid.

PCA is linear, so averaging the 2D coordinates equals projecting the 1,536-d centroid, and the centroid positions are correct.
(UMAP is nonlinear, so averaging its 2D coordinates does not have this property; centroids are only drawn in the PCA version.)

Outputs (out/):
  atlas_pca.parquet         books + 36 quarterly centroids (told apart by the kind column)
  atlas_pca_labels.parquet  text labels for the centroids (x, y, text), replacing automatic cluster labels

Usage: python 07_atlas_pca.py, then run 08_atlas_pca.sh
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "out"

a = pd.read_parquet(OUT / "atlas.parquet")
# atlas.parquet files written before the code was translated (including the copy on the HF dataset) store this label in Chinese
a["top10"] = a["top10"].replace({"\u524d 10%": "top 10%", "\u5176\u4ed6": "other"})
a.insert(0, "kind", "book")

top = a[a.top10 == "top 10%"]
cent = top.groupby("quarter").agg(pca_x=("pca_x", "mean"), pca_y=("pca_y", "mean"), n=("fiction_id", "size"),
                                  year=("year", "first")).reset_index()
cent["label"] = cent.quarter.str.slice(2, 4) + cent.quarter.str.slice(4)   # 2024Q4 → 24Q4

rows = pd.DataFrame({
    "kind": "quarterly centroid (top 10%)",
    "title": "centroid " + cent.quarter + " (" + cent.n.astype(str) + " books)",
    "year": cent.year.astype(int), "quarter": cent.quarter, "top10": "quarterly centroid",
    "pca_x": cent.pca_x.astype("float32"), "pca_y": cent.pca_y.astype("float32"),
    "neighbors": [{"ids": [], "distances": []} for _ in range(len(cent))],
})
out = pd.concat([a, rows], ignore_index=True)   # centroids go last, so book row numbers and neighbor ids stay correct
out.to_parquet(OUT / "atlas_pca.parquet", index=False)

labels = pd.DataFrame({"x": cent.pca_x.astype(float), "y": cent.pca_y.astype(float), "text": cent.label.astype(object), "level": 0,
                       "priority": np.arange(len(cent))})   # newer quarters are shown first
labels.to_parquet(OUT / "atlas_pca_labels.parquet", index=False)

spread = a[["pca_x", "pca_y"]].std()
move = cent[["pca_x", "pca_y"]].iloc[-1] - cent[["pca_x", "pca_y"]].iloc[0]
print(f"{len(cent)} quarterly centroids, top-10% books per quarter {cent.n.min()}–{cent.n.max()}")
print(f"spread of books on the PCA plane x={spread.pca_x:.3f} y={spread.pca_y:.3f}; "
      f"centroid move 2016Q1→2024Q4 x={move.pca_x:+.3f} y={move.pca_y:+.3f}")
