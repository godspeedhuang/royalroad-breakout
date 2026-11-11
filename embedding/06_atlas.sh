#!/bin/sh
# 用 Embedding Atlas 開啟互動探索介面：http://localhost:5055
cd "$(dirname "$0")"
uvx embedding-atlas out/atlas.parquet --text chapter1_excerpt --x umap_x --y umap_y \
  --neighbors neighbors --disable-projection --port 5055
