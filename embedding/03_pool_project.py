"""步驟 5–6：把塊向量合併成章與書的向量，再算 PCA / UMAP 2D 座標與最近鄰。

輸出（out/）：
  chapters.parquet  一章一列：原文、tags、warnings、章節向量
  books.parquet     一本一列：書向量、第一章向量、簡介向量、2D 座標、所有欄位
  atlas.parquet     給 Embedding Atlas 的輕量版（沒有 1536 維向量，只有座標、篩選欄位與摘錄）

用法：python 03_pool_project.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA

HERE = Path(__file__).parent
OUT = HERE / "out"
DIM = 1536
K_NEIGHBORS = 15
TAG_FLAGS = ["LitRPG", "Progression", "Portal Fantasy / Isekai", "Reincarnation", "Romance", "Female Lead", "Male Lead",
             "Slice of Life", "Sci-fi", "Dungeon", "GameLit", "Martial Arts", "Harem", "Horror", "Mystery", "Grimdark"]


def normalize(m):
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def weighted_group_mean(keys, vecs, w):
    """依 keys 分組，對 vecs 做加權平均後正規化。keys 需已排序。"""
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    summed = np.add.reduceat(vecs * w[:, None], starts, axis=0)
    return keys[starts], normalize(summed), np.add.reduceat(w, starts)


def vec_col(m):
    return pa.FixedSizeListArray.from_arrays(pa.array(np.ascontiguousarray(m, dtype=np.float32).ravel()), DIM)


def main():
    parts = pq.read_table(list((OUT / "emb_parts").glob("*.parquet")))
    uid = np.array(parts.column("chunk_uid").to_pylist())
    V = parts.column("vector").combine_chunks().flatten().to_numpy().reshape(-1, DIM).astype(np.float32)
    meta = pd.read_parquet(OUT / "chunks_input.parquet", columns=["chunk_uid", "fiction_id", "chapter_idx", "n_tokens"])
    meta = meta.set_index("chunk_uid").loc[uid].reset_index()
    missing = len(pd.read_parquet(OUT / "chunks_input.parquet", columns=["chunk_uid"])) - len(uid)
    print(f"塊向量 {len(uid):,}（缺 {missing}）")

    # ---- 章節向量：同一章的塊依 token 數加權 ----
    meta["ckey"] = meta.fiction_id.astype(np.int64) * 10 + meta.chapter_idx
    order = np.argsort(meta.ckey.values, kind="stable")
    ckeys, CV, ctok = weighted_group_mean(meta.ckey.values[order], V[order], meta.n_tokens.values[order].astype(np.float32))
    cfid, cidx = ckeys // 10, ckeys % 10
    chapvec = pd.DataFrame({"fiction_id": cfid, "chapter_idx": cidx, "row": np.arange(len(ckeys))})

    # ---- 書向量：第 1–3 章依 token 數加權；另存第一章、簡介 ----
    body = chapvec[chapvec.chapter_idx > 0]
    bkeys, BV, _ = weighted_group_mean(body.fiction_id.values, CV[body.row.values], ctok[body.row.values])
    books = pd.read_parquet(OUT / "books_meta.parquet").set_index("fiction_id").loc[bkeys].reset_index()
    def pick(idx):
        m = chapvec[chapvec.chapter_idx == idx].set_index("fiction_id").row
        out = np.full((len(bkeys), DIM), np.nan, dtype=np.float32)
        hit = pd.Series(bkeys).map(m)
        ok = hit.notna().values
        out[ok] = CV[hit[ok].astype(int).values]
        return out
    C1, DV = pick(1), pick(0)

    # ---- 降維（所有年份一起擬合）----
    pca = PCA(n_components=50, random_state=42).fit(BV)
    P = pca.transform(BV)
    books["pca_x"], books["pca_y"] = P[:, 0], P[:, 1]
    ev = pca.explained_variance_ratio_
    print(f"PCA 前兩軸解釋 {ev[:2].sum():.1%}（PC1 {ev[0]:.1%}、PC2 {ev[1]:.1%}），前 50 軸 {ev.sum():.1%}")
    import umap
    U = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(P)
    books["umap_x"], books["umap_y"] = U[:, 0], U[:, 1]

    # ---- 最近鄰（cosine，分批矩陣乘法）----
    nb_ids, nb_dist = [], []
    for s in range(0, len(BV), 2000):
        sim = BV[s:s + 2000] @ BV.T
        np.put_along_axis(sim, (np.arange(sim.shape[0]) + s)[:, None], -np.inf, axis=1)  # 排除自己
        idx = np.argpartition(-sim, K_NEIGHBORS, axis=1)[:, :K_NEIGHBORS]
        srt = np.take_along_axis(sim, idx, axis=1).argsort(axis=1)[:, ::-1]
        idx = np.take_along_axis(idx, srt, axis=1)
        nb_ids += idx.tolist(); nb_dist += (1 - np.take_along_axis(sim, idx, axis=1)).round(4).tolist()

    # ---- 寫檔 ----
    tbl = pa.Table.from_pandas(books, preserve_index=False)
    tbl = tbl.append_column("book_vec", vec_col(BV)).append_column("chap1_vec", vec_col(np.nan_to_num(C1))) \
             .append_column("desc_vec", vec_col(np.nan_to_num(DV)))
    pq.write_table(tbl, OUT / "books.parquet")

    chapters = pd.read_parquet(OUT / "chapters_meta.parquet")
    chapters = chapters.merge(books[["fiction_id", "tags", "warnings", "year", "quarter", "top10", "pct_in_quarter"]], on="fiction_id")
    cv = chapvec[chapvec.chapter_idx > 0].set_index(["fiction_id", "chapter_idx"]).row
    rows = chapters.set_index(["fiction_id", "chapter_idx"]).index.map(cv)
    ok = ~pd.isna(rows)
    chapters = chapters[ok].reset_index(drop=True)
    ctbl = pa.Table.from_pandas(chapters, preserve_index=False).append_column("vector", vec_col(CV[np.asarray(rows[ok], dtype=int)]))
    pq.write_table(ctbl, OUT / "chapters.parquet")

    # Embedding Atlas 用的輕量檔
    first = chapters[chapters.chapter_idx == 1].set_index("fiction_id").text
    tags = books.tags.map(lambda s: json.loads(s) if isinstance(s, str) and s else [])
    atlas = pd.DataFrame({
        "fiction_id": books.fiction_id, "title": books.title, "author": books.author, "author_id": books.author_id,
        "first_release": pd.to_datetime(books.first_release, utc=True).dt.tz_localize(None),
        "year": books.year.astype(int), "quarter": books.quarter, "top10": np.where(books.top10, "前 10%", "其他"),
        "pct_in_quarter": books.pct_in_quarter.round(4), "followers": books.followers, "log_followers": books.log_followers.round(3),
        "tags": tags.map(", ".join), "n_tags": tags.map(len), "is_original": books.is_original,
        "first3_tokens": books.first3_tokens, "first3_truncated": books.first3_truncated,
        "description": books.description.fillna("").str.slice(0, 800),
        "chapter1_excerpt": books.fiction_id.map(first).fillna("").str.slice(0, 2000),
        "umap_x": books.umap_x.round(4), "umap_y": books.umap_y.round(4), "pca_x": books.pca_x.round(4), "pca_y": books.pca_y.round(4),
        "neighbors": [{"ids": i, "distances": d} for i, d in zip(nb_ids, nb_dist)],
    })
    for t in TAG_FLAGS:
        atlas["tag_" + t.split(" /")[0].replace(" ", "_").replace("-", "_")] = tags.map(lambda l, t=t: t in l)
    atlas.to_parquet(OUT / "atlas.parquet", index=False)
    json.dump({"pca_explained": ev[:10].round(4).tolist(), "n_books": len(books), "missing_chunks": int(missing)},
              open(OUT / "projection_summary.json", "w"), indent=1)
    print("寫出 books", len(books), "chapters", len(chapters), "atlas", atlas.shape)


if __name__ == "__main__":
    main()
