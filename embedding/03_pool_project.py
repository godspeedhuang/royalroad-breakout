"""Steps 5–6: pool chunk vectors into chapter and book vectors, then compute PCA / UMAP 2D coordinates and nearest neighbors.

Outputs (out/):
  chapters.parquet  one row per chapter: text, tags, warnings, chapter vector
  books.parquet     one row per book: book vector, chapter-1 vector, blurb vector, 2D coordinates, all fields
  atlas.parquet     light version for Embedding Atlas (no 1,536-d vectors; coordinates, filter fields and excerpts only)

Usage: python 03_pool_project.py
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
    """Group by keys, take the weighted mean of vecs, then normalize. keys must be sorted."""
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
    print(f"chunk vectors {len(uid):,} ({missing} missing)")

    # ---- Chapter vectors: chunks of the same chapter, weighted by token count ----
    meta["ckey"] = meta.fiction_id.astype(np.int64) * 10 + meta.chapter_idx
    order = np.argsort(meta.ckey.values, kind="stable")
    ckeys, CV, ctok = weighted_group_mean(meta.ckey.values[order], V[order], meta.n_tokens.values[order].astype(np.float32))
    cfid, cidx = ckeys // 10, ckeys % 10
    chapvec = pd.DataFrame({"fiction_id": cfid, "chapter_idx": cidx, "row": np.arange(len(ckeys))})

    # ---- Book vectors: chapters 1–3 weighted by token count; chapter 1 and the blurb are stored separately ----
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

    # ---- Dimensionality reduction (fitted on all years together) ----
    pca = PCA(n_components=50, random_state=42).fit(BV)
    P = pca.transform(BV)
    books["pca_x"], books["pca_y"] = P[:, 0], P[:, 1]
    ev = pca.explained_variance_ratio_
    print(f"PCA: first two axes explain {ev[:2].sum():.1%} (PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}), first 50 axes {ev.sum():.1%}")
    import umap
    U = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(P)
    books["umap_x"], books["umap_y"] = U[:, 0], U[:, 1]

    # ---- Nearest neighbors (cosine, batched matrix products) ----
    nb_ids, nb_dist = [], []
    for s in range(0, len(BV), 2000):
        sim = BV[s:s + 2000] @ BV.T
        np.put_along_axis(sim, (np.arange(sim.shape[0]) + s)[:, None], -np.inf, axis=1)  # exclude the book itself
        idx = np.argpartition(-sim, K_NEIGHBORS, axis=1)[:, :K_NEIGHBORS]
        srt = np.take_along_axis(sim, idx, axis=1).argsort(axis=1)[:, ::-1]
        idx = np.take_along_axis(idx, srt, axis=1)
        nb_ids += idx.tolist(); nb_dist += (1 - np.take_along_axis(sim, idx, axis=1)).round(4).tolist()

    # ---- Write files ----
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

    # Light file for Embedding Atlas
    first = chapters[chapters.chapter_idx == 1].set_index("fiction_id").text
    tags = books.tags.map(lambda s: json.loads(s) if isinstance(s, str) and s else [])
    atlas = pd.DataFrame({
        "fiction_id": books.fiction_id, "title": books.title, "author": books.author, "author_id": books.author_id,
        "first_release": pd.to_datetime(books.first_release, utc=True).dt.tz_localize(None),
        "year": books.year.astype(int), "quarter": books.quarter, "top10": np.where(books.top10, "top 10%", "other"),
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
    print("wrote books", len(books), "chapters", len(chapters), "atlas", atlas.shape)


if __name__ == "__main__":
    main()
