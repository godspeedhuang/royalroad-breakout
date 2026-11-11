#!/bin/sh
# 把要花錢才算得出來的向量檔，上傳到公開的 Hugging Face dataset，讓隊友和評分者不用重打 API。
# 用法：./embedding/14_publish_hf.sh <hf-username>/royalroad-breakout
# 前提：uv run hf auth login（token 要有 write 權限）
# HF 上的路徑和這個 repo 一樣（embedding/out/...），所以下載時 --local-dir . 就會放到對的位置。
set -e
REPO="${1:?用法：$0 <hf-username>/<repo-name>}"
cd "$(dirname "$0")/.."

for f in books.parquet books_meta.parquet atlas.parquet atlas_pca.parquet; do
  uv run hf upload "$REPO" "embedding/out/$f" "embedding/out/$f" --repo-type dataset \
    --commit-message "Upload $f"
done

CARD="$(mktemp)"
cat > "$CARD" <<EOF
---
license: mit
language: [en]
pretty_name: Royal Road Breakout (embeddings)
size_categories: [10K<n<100K]
source_datasets: [OmniAICreator/RoyalRoad-1.61M]
---

# Royal Road Breakout: first-three-chapter embeddings

Derived from [OmniAICreator/RoyalRoad-1.61M](https://huggingface.co/datasets/OmniAICreator/RoyalRoad-1.61M) (MIT) for MIT 15.071 Project 1.
Code and documentation: the project's GitHub repository.

45,245 web serials first released on royalroad.com in 2016–2024 with at least 3 chapters, one row per book.
No chapter text is included.

| File | Rows | Contents |
|---|---|---|
| \`embedding/out/books.parquet\` | 45,245 | Book metadata, \`top10\` label, 2D coordinates, and three 1,536-d vectors: \`book_vec\` (chapters 1–3), \`chap1_vec\`, \`desc_vec\` (blurb) |
| \`embedding/out/books_meta.parquet\` | 45,245 | Same metadata without vectors |
| \`embedding/out/atlas.parquet\` | 45,245 | Input for Apple Embedding Atlas (UMAP and PCA coordinates, nearest neighbours, 2,000-char chapter 1 excerpt) |
| \`embedding/out/atlas_pca.parquet\` | 45,281 | Same plus 36 rows of per-quarter top-10% centroids |

- Embeddings: \`text-embedding-3-small\`. Each chapter is embedded separately (chapters over 8,000 tokens split evenly and re-averaged), books capped at 20,000 tokens, book vector = token-weighted mean of chapters 1–3, L2-normalised.
- \`top10\`: followers at or above the 90th percentile among books first released in the same calendar quarter. Followers are a single snapshot taken around 2025-06-16.
- Known issue: about 6,500 books contain anti-piracy sentences in their chapter text, which also shift their vectors. The list is in the GitHub repo (\`embedding/out/watermark_books.csv\`).

\`\`\`python
import pandas as pd
books = pd.read_parquet("hf://datasets/$REPO/embedding/out/books.parquet")
\`\`\`
EOF
uv run hf upload "$REPO" "$CARD" README.md --repo-type dataset --commit-message "Dataset card"
rm "$CARD"
echo "https://huggingface.co/datasets/$REPO"
