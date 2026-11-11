"""modeling/ 共用：讀資料、做特徵、時間切分、指標。

輸入：embedding/out/books.parquet（一本一列）、embedding/out/watermark_books.csv
Y   ：top10 = 同一季開始連載的書裡，followers 是否 ≥ 90 百分位

特徵區塊（都只用第 3 章發布時拿得到的資訊）：
  S 結構化：tags、warnings、is_original、前三章長度與發布間隔
  A 作者經驗：是不是第一次寫書、以前寫過幾本（只算更早開始連載的書）
  T 前三章向量 book_vec（1,536 維）
  C 第一章向量 chap1_vec（1,536 維）

有 leak 疑慮、只當上限參考的區塊：
  P 前作表現：前作有沒有進同季前 10%。前作的 followers 是 2025 年快照，這本書紅了會回頭拉高前作。
  D 簡介向量 desc_vec（1,536 維）+ 簡介長度。簡介是 2025 年的版本，作者紅了之後會改
    （加上 Amazon、Kindle、Patreon、Audible、"Book 2" 之類的字）。b["desc_marker"] 標出這些書。

不能當特徵（事後才有）：followers、total_views、average_views、favorites、ratings、
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

# 時間切分。2016 年不用：75% 的書含防盜浮水印（見 embedding/11_watermark_check.py）。
# 2024 年觀察時間太短，「前 10%」還不穩，只當補充。
TRAIN_YEARS, VAL_YEAR, TEST_YEAR, RECENT_YEAR = (2017, 2021), 2022, 2023, 2024
VECS = {"D": "desc_vec", "T": "book_vec", "C": "chap1_vec"}
# 簡介裡「紅了之後才會出現」的字（不完整，只用來做敏感度檢查）
DESC_MARKERS = ["amazon", "kindle", "patreon", "audible", "audiobook", "stubbed", "book 2", "book two", "volume 2", "discord"]


def load():
    """回傳 (books DataFrame, {區塊名: 2D float32 array})，列順序一致。"""
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
    """同一作者更早開始連載的書。

    ponytail: 只算樣本內的書（2016–2024、至少 3 章），更早的或不到 3 章的前作沒算進來；
    要補的話從 data/interim/fictions_47.parquet 算。
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
    """把幾個特徵區塊接起來，例如 block(X, "SAD")。"""
    return np.hstack([X[k] for k in names])


def recall_at_slots(y, score, quarter):
    """每一季的推薦名額 = 該季真正的熱門書數量（約 10%）；回傳所有名額裡挑中熱門書的比例。

    名額數等於熱門書數，所以這個數字同時是 recall 和 precision。隨機挑大約是 0.10。
    """
    d = pd.DataFrame({"y": np.asarray(y, bool), "s": score, "q": quarter})
    hits = sum(g.nlargest(int(g.y.sum()), "s").y.sum() for _, g in d.groupby("q"))
    return hits / d.y.sum()


def metrics(y, score, quarter):
    return {"auc": float(roc_auc_score(y, score)), "ap": float(average_precision_score(y, score)),
            "recall_at_slots": float(recall_at_slots(y, score, quarter))}


if __name__ == "__main__":  # 自我檢查：python modeling/features.py
    b, X = load()
    sp = splits(b)
    for k, m in sp.items():
        print(f"{k:6s} {m.sum():6,d} 本  top10 {b.top10[m].mean():.3f}  含浮水印 {b.wm[m].mean():.3f}  "
              f"簡介有事後字眼 {b.desc_marker[m].mean():.3f}  第一次寫書 {b.first_time_author[m].mean():.3f}")
    print({k: v.shape for k, v in X.items()})
    # 作者歷史不能偷看：每位作者最早的那本書一定沒有前作
    first = b.sort_values(["first_release", "fiction_id"]).groupby("author_id").head(1).index
    assert (X["A"][b.index.get_indexer(first)] == 0).all() and (X["P"][b.index.get_indexer(first)] == 0).all()
    # 寫最多本的作者：前作本數要是 0, 1, 2, ...；前作表現要等於手算的值
    a = b.author_id.value_counts().index[0]
    o = b[b.author_id == a].sort_values(["first_release", "fiction_id"])
    i = b.index.get_indexer(o.index)
    assert np.allclose(np.expm1(X["A"][i, 1]), np.arange(len(o)), atol=1e-3)
    assert np.allclose(X["P"][i, 0], o.top10.astype(float).shift().cummax().fillna(0).values)
    # 指標：完美排序 = 1，反向排序 = 0
    y = b.top10.values[sp["test"]]; q = b.quarter.values[sp["test"]]
    assert recall_at_slots(y, y.astype(float), q) == 1 and recall_at_slots(y, -y.astype(float), q) == 0
    r = np.random.default_rng(0).random(len(y))
    print("隨機分數：", {k: round(v, 3) for k, v in metrics(y, r, q).items()})
    print("ok")
