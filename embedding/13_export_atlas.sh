#!/bin/sh
# Export the public Embedding Atlas (static site) to docs/atlas/ for GitHub Pages.
# Differences from the local versions (06, 08): no author names, chapter-1 excerpt cut to 300 characters, link to the original page.
#   docs/atlas/pca/   PCA coordinates + each quarter's top-10% centroid
#   docs/atlas/umap/  UMAP coordinates (books only)
set -e
cd "$(dirname "$0")"

uv run python - <<'EOF'
import pandas as pd

EXCERPT = 300  # excerpt length limit for the public version
a = pd.read_parquet("out/atlas_pca.parquet").drop(columns=["author"])
# atlas_pca.parquet files written before the code was translated (including the copy on the HF dataset) store these labels in Chinese
a["kind"] = a["kind"].replace({"\u66f8": "book", "\u5b63\u4e2d\u5fc3\u9ede\uff08\u524d 10%\uff09": "quarterly centroid (top 10%)"})
a["top10"] = a["top10"].replace({"\u524d 10%": "top 10%", "\u5176\u4ed6": "other", "\u5b63\u4e2d\u5fc3\u9ede": "quarterly centroid"})
a["title"] = a["title"].str.replace(r"^\u4e2d\u5fc3\u9ede (\S+)\uff08(\d+) \u672c\uff09$", r"centroid \1 (\2 books)", regex=True)
a["chapter1_excerpt"] = a["chapter1_excerpt"].str.slice(0, EXCERPT)
a["url"] = "https://www.royalroad.com/fiction/" + a["fiction_id"].astype("Int64").astype("string")
assert "author" not in a.columns and a["chapter1_excerpt"].str.len().max() <= EXCERPT
a.to_parquet("out/atlas_public_pca.parquet", index=False)
a[a["kind"] == "book"].to_parquet("out/atlas_public_umap.parquet", index=False)
print(f"public atlas: {len(a):,} rows (pca), {(a['kind'] == 'book').sum():,} rows (umap)")
EOF

for v in pca umap; do
  rm -rf "../docs/atlas/$v"
  uvx embedding-atlas "out/atlas_public_$v.parquet" --text chapter1_excerpt \
    --x "${v}_x" --y "${v}_y" --neighbors neighbors --disable-projection \
    --export-application "../docs/atlas/$v" < /dev/null
done
du -sh ../docs/atlas/*
