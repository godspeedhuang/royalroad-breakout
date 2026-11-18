"""Shared code for modeling/: load data, build features, time split, metrics.

Inputs: embedding/out/books.parquet (one row per book), embedding/out/watermark_books.csv
Y     : top10 = whether followers are at or above the 90th percentile among books first released in the same quarter

Feature blocks (all use only information available when chapter 3 is published):
  S structured: tags, warnings, is_original, length of chapters 1–3 and days between them
  A author experience: first book or not, number of earlier books (only books first released earlier)
  T first-three-chapter vector book_vec (1,536-d)
  C chapter-1 vector chap1_vec (1,536-d)

Leak-prone blocks, used only as an upper bound:
  P earlier books' success: whether an earlier book reached its quarter's top 10%. Earlier books' followers are a 2025 snapshot, and a new hit pulls readers back to them.
  D blurb vector desc_vec (1,536-d) + blurb length. The blurb is the 2025 version; authors edit it after a book takes off
    (adding words like Amazon, Kindle, Patreon, Audible, "Book 2"). b["desc_marker"] flags these books.

Not usable as features (only known afterwards): followers, total_views, average_views, favorites, ratings,
n_chapters、pct_in_quarter、log_followers、status、cover_url、pca_*、umap_*
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).parent
EMB = HERE.parent / "embedding" / "out"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

# Time split. 2016 is not used: 75% of its books carry anti-piracy watermarks (see embedding/11_watermark_check.py).
# 2024 has been observed too briefly for its top 10% to settle; supplementary only.
TRAIN_YEARS, VAL_YEAR, TEST_YEAR, RECENT_YEAR = (2017, 2021), 2022, 2023, 2024
VECS = {"D": "desc_vec", "T": "book_vec", "C": "chap1_vec"}
# Words in blurbs that only appear after a book takes off (incomplete; used only for a sensitivity check)
DESC_MARKERS = ["amazon", "kindle", "patreon", "audible", "audiobook", "stubbed", "book 2", "book two", "volume 2", "discord"]


def load():
    """Returns (books DataFrame, {block name: 2D float32 array}) with matching row order."""
    t = pq.read_table(EMB / "books.parquet")
    b = t.drop(list(VECS.values())).to_pandas()
    X = {k: np.nan_to_num(t.column(c).combine_chunks().flatten().to_numpy().reshape(len(b), -1).astype(np.float32))
         for k, c in VECS.items()}
    wm = set(pd.read_csv(EMB / "watermark_books.csv").fiction_id)
    b["wm"] = b.fiction_id.isin(wm)
    desc = b.description.fillna("").str.lower()
    b["desc_marker"] = np.logical_or.reduce([desc.str.contains(t, regex=False).values for t in DESC_MARKERS])

    tags = b.tags.map(json.loads)
    warns = b.warnings.map(json.loads)
    S = pd.concat([
        _multihot(tags, "tag"), _multihot(warns, "warn"),
        pd.DataFrame({
            "is_original": b.is_original, "n_tags": tags.map(len),
            "log_first3_tokens": np.log1p(b.first3_tokens), "first3_truncated": b.first3_truncated,
            "log_first3_span_days": np.log1p(b.first3_span_days),
        }),
    ], axis=1).astype(np.float32)
    A = _author_history(b)
    b["first_time_author"] = (A.has_prior == 0).values
    X["S"], X["A"], X["P"] = S.values, A[["has_prior", "log_prior_n"]].values, A[["prior_any_top10"]].values
    X["D"] = np.hstack([X["D"], np.log1p(desc.str.len()).values[:, None].astype(np.float32)])
    b.attrs["cols"] = {"S": list(S.columns), "A": ["has_prior", "log_prior_n"], "P": ["prior_any_top10"]}
    return b, X


def _multihot(lists, prefix):
    names = sorted({x for l in lists for x in l})
    return pd.DataFrame({f"{prefix}_{n}": lists.map(lambda l, n=n: n in l) for n in names})


def _author_history(b):
    """Earlier books by the same author (first released earlier).

    ponytail: only books in the sample count (2016–2024, at least 3 chapters); earlier books, or books with fewer than 3 chapters, are not counted;
    to include them, compute from data/interim/fictions_47.parquet.
    """
    o = b.sort_values(["first_release", "fiction_id"])
    g = o.groupby("author_id", sort=False)
    n = g.cumcount()
    hit = g.top10.transform(lambda s: s.astype(float).shift().cummax())
    A = pd.DataFrame({"has_prior": n > 0, "log_prior_n": np.log1p(n), "prior_any_top10": hit.fillna(0)}).astype(np.float32)
    return A.loc[b.index]


def splits(b):
    y = b.year.values
    return {"train": (y >= TRAIN_YEARS[0]) & (y <= TRAIN_YEARS[1]), "val": y == VAL_YEAR,
            "test": y == TEST_YEAR, "recent": y == RECENT_YEAR}


def block(X, names):
    """Concatenate feature blocks, e.g. block(X, "SAD")."""
    return np.hstack([X[k] for k in names])


def recall_at_slots(y, score, quarter):
    """Each quarter gets as many slots as it has true hits (about 10%); returns the share of all slots that go to a hit.

    The number of slots equals the number of hits, so this is both recall and precision. Random picks score about 0.10.
    """
    d = pd.DataFrame({"y": np.asarray(y, bool), "s": score, "q": quarter})
    hits = sum(g.nlargest(int(g.y.sum()), "s").y.sum() for _, g in d.groupby("q"))
    return hits / d.y.sum()


def metrics(y, score, quarter):
    return {"auc": float(roc_auc_score(y, score)), "ap": float(average_precision_score(y, score)),
            "recall_at_slots": float(recall_at_slots(y, score, quarter))}


if __name__ == "__main__":  # self-check: python modeling/features.py
    b, X = load()
    sp = splits(b)
    for k, m in sp.items():
        print(f"{k:6s} {m.sum():6,d} books  top10 {b.top10[m].mean():.3f}  watermark {b.wm[m].mean():.3f}  "
              f"blurb with after-success words {b.desc_marker[m].mean():.3f}  first-time author {b.first_time_author[m].mean():.3f}")
    print({k: v.shape for k, v in X.items()})
    # Author history must not peek ahead: each author's first book has no earlier books
    first = b.sort_values(["first_release", "fiction_id"]).groupby("author_id").head(1).index
    assert (X["A"][b.index.get_indexer(first)] == 0).all() and (X["P"][b.index.get_indexer(first)] == 0).all()
    # The most prolific author: earlier-book counts must be 0, 1, 2, ...; earlier-book success must match a hand computation
    a = b.author_id.value_counts().index[0]
    o = b[b.author_id == a].sort_values(["first_release", "fiction_id"])
    i = b.index.get_indexer(o.index)
    assert np.allclose(np.expm1(X["A"][i, 1]), np.arange(len(o)), atol=1e-3)
    assert np.allclose(X["P"][i, 0], o.top10.astype(float).shift().cummax().fillna(0).values)
    # Metric: perfect ranking = 1, reversed ranking = 0
    y = b.top10.values[sp["test"]]; q = b.quarter.values[sp["test"]]
    assert recall_at_slots(y, y.astype(float), q) == 1 and recall_at_slots(y, -y.astype(float), q) == 0
    r = np.random.default_rng(0).random(len(y))
    print("random scores:", {k: round(v, 3) for k, v in metrics(y, r, q).items()})
    print("ok")
