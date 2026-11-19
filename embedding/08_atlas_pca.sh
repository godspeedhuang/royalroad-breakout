#!/bin/sh
# PCA version of the Embedding Atlas (with each quarter's top-10% centroid): http://localhost:5057
cd "$(dirname "$0")"
uvx embedding-atlas out/atlas_pca.parquet --text chapter1_excerpt --x pca_x --y pca_y \
  --neighbors neighbors --disable-projection --port 5057 < /dev/null
