#!/bin/sh
# 匯出公開版 Embedding Atlas（靜態網頁）到 docs/atlas/，給 GitHub Pages 用。
# 公開版和本機版（06、08）的差別：拿掉作者名、第一章節錄縮到 300 字、加上原站連結。
#   docs/atlas/pca/   PCA 座標 + 每季前 10% 中心點
#   docs/atlas/umap/  UMAP 座標（只有書）
set -e
cd "$(dirname "$0")"

uv run python - <<'EOF'
import pandas as pd

EXCERPT = 300  # 公開版節錄字數上限
a = pd.read_parquet("out/atlas_pca.parquet").drop(columns=["author"])
a["chapter1_excerpt"] = a["chapter1_excerpt"].str.slice(0, EXCERPT)
a["url"] = "https://www.royalroad.com/fiction/" + a["fiction_id"].astype("Int64").astype("string")
assert "author" not in a.columns and a["chapter1_excerpt"].str.len().max() <= EXCERPT
a.to_parquet("out/atlas_public_pca.parquet", index=False)
a[a["kind"] == "書"].to_parquet("out/atlas_public_umap.parquet", index=False)
print(f"public atlas: {len(a):,} rows (pca), {(a['kind'] == '書').sum():,} rows (umap)")
EOF

for v in pca umap; do
  rm -rf "../docs/atlas/$v"
  uvx embedding-atlas "out/atlas_public_$v.parquet" --text chapter1_excerpt \
    --x "${v}_x" --y "${v}_y" --neighbors neighbors --disable-projection \
    --export-application "../docs/atlas/$v" < /dev/null
done
du -sh ../docs/atlas/*
