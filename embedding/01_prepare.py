"""步驟 1–3：挑書、取前三章、算 Y 與欄位、切塊。

輸出（都在 out/）：
  books_meta.parquet     一本一列：Y、篩選欄位、事前與事後欄位
  chapters_meta.parquet  一章一列：前三章原文與 token 數
  chunks_input.parquet   要送去 embedding 的每一塊（含簡介）

用法：python 01_prepare.py
"""
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import tiktoken

HERE = Path(__file__).parent
RAW = HERE.parent / "data" / "raw" / "data"
INTERIM = HERE.parent / "data" / "interim"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

YEARS = (2016, 2024)
BOOK_CAP = 20_000   # 前三章合計上限（token）
CHUNK_MAX = 8_000   # 單塊上限，模型上限約 8,191
DESC_MAX = 8_000

enc = tiktoken.get_encoding("cl100k_base")
ntok = lambda s: len(enc.encode(s, disallowed_special=()))


def split_even(text, n_tokens, k):
    """把一章平均切成 k 塊，盡量在段落邊界；單一段落太長才硬切。"""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    target = n_tokens / k
    chunks, cur, cur_n = [], [], 0
    for p in paras:
        pn = ntok(p)
        if cur and cur_n + pn > target and len(chunks) < k - 1:
            chunks.append("\n\n".join(cur)); cur, cur_n = [], 0
        cur.append(p); cur_n += pn
    if cur:
        chunks.append("\n\n".join(cur))
    out = []
    for c in chunks:                       # 保證每塊都在上限內
        ids = enc.encode(c, disallowed_special=())
        for i in range(0, len(ids), CHUNK_MAX):
            out.append(enc.decode(ids[i:i + CHUNK_MAX]))
    return out


def main():
    # ---- 選書：用 EDA 已建好的章節索引 ----
    ch = pd.read_parquet(INTERIM / "chapters_47.parquet", columns=["fiction_id", "chapter_id", "release_datetime"])
    ch["t"] = pd.to_datetime(ch.release_datetime, utc=True, errors="coerce")
    ch = ch.sort_values(["fiction_id", "t", "chapter_id"])
    g = ch.groupby("fiction_id")
    book = pd.DataFrame({"first_release": g.t.min(), "n_chapters": g.size()})
    book = book[(book.n_chapters >= 3) & book.first_release.dt.year.between(*YEARS)]
    first3 = ch[ch.fiction_id.isin(book.index)].groupby("fiction_id").head(3).copy()
    first3["chapter_idx"] = first3.groupby("fiction_id").cumcount() + 1
    print("books", len(book), "chapters", len(first3))

    # ---- 取前三章原文 ----
    dset = ds.dataset(sorted(RAW.glob("*.parquet")))
    ids = first3.chapter_id.dropna().astype("float64").tolist()
    txt = dset.to_table(columns=["fiction_id", "chapter_id", "chapter_title", "text"],
                        filter=ds.field("chapter_id").isin(ids)).to_pandas()
    chap = first3.merge(txt, on=["fiction_id", "chapter_id"], how="left")
    chap["text"] = chap.text.fillna("")

    # ---- 小說層級欄位 ----
    fi = pd.read_parquet(INTERIM / "fictions_47.parquet").set_index("fiction_id")
    au = (dset.to_table(columns=["fiction_id", "author_url"], filter=ds.field("fiction_id").isin(book.index.tolist()))
          .to_pandas().drop_duplicates("fiction_id").set_index("fiction_id"))
    b = book.join(fi[["title", "author", "status", "is_original", "tags", "warnings", "description", "cover_url",
                      "followers", "total_views", "average_views", "favorites", "ratings"]]).join(au)
    b["author_id"] = b.author_url.fillna(b.author).map(lambda s: hashlib.sha1(str(s).encode()).hexdigest()[:12])
    b["year"] = b.first_release.dt.year
    b["quarter"] = b.first_release.dt.tz_localize(None).dt.to_period("Q").astype(str)
    b["pct_in_quarter"] = b.groupby("quarter").followers.rank(pct=True)
    b["top10"] = b.followers >= b.groupby("quarter").followers.transform(lambda s: s.quantile(.9))
    b["log_followers"] = np.log10(1 + b.followers)

    # ---- 截斷與切塊 ----
    chap["n_tokens_raw"] = chap.text.map(ntok)
    rows, kept = [], []
    for fid, grp in chap.groupby("fiction_id", sort=False):
        budget = BOOK_CAP
        for r in grp.itertuples():
            text, n = r.text, r.n_tokens_raw
            truncated = False
            if n > budget:
                text = enc.decode(enc.encode(text, disallowed_special=())[:max(budget, 0)])
                n, truncated = budget, True
            budget -= n
            kept.append((fid, r.chapter_idx, n, truncated))
            if n == 0:
                continue
            k = math.ceil(n / CHUNK_MAX)
            parts = [text] if k == 1 else split_even(text, n, k)
            for j, ptxt in enumerate(parts):
                rows.append((f"{fid}-c{r.chapter_idx}-p{j}", fid, r.chapter_idx, j, len(parts), ntok(ptxt), ptxt))
    kept = pd.DataFrame(kept, columns=["fiction_id", "chapter_idx", "n_tokens", "truncated"])
    chap = chap.merge(kept, on=["fiction_id", "chapter_idx"])

    # 簡介當成第 0 章
    for fid, desc in b.description.fillna("").items():
        if desc.strip():
            ids_ = enc.encode(desc, disallowed_special=())[:DESC_MAX]
            rows.append((f"{fid}-c0-p0", fid, 0, 0, 1, len(ids_), enc.decode(ids_)))
    chunks = pd.DataFrame(rows, columns=["chunk_uid", "fiction_id", "chapter_idx", "part", "n_parts", "n_tokens", "text"])

    # ---- 書層級的前三章摘要欄位 ----
    agg = chap.groupby("fiction_id").agg(first3_tokens=("n_tokens", "sum"), first3_truncated=("truncated", "any"),
                                         first3_last=("t", "max"))
    b = b.join(agg)
    b["first3_span_days"] = (b.first3_last - b.first_release).dt.total_seconds() / 86400
    b = b.drop(columns=["first3_last"]).reset_index().rename(columns={"index": "fiction_id"})

    b.to_parquet(OUT / "books_meta.parquet", index=False)
    chap[["fiction_id", "chapter_idx", "chapter_id", "chapter_title", "t", "n_tokens_raw", "n_tokens", "truncated", "text"]] \
        .rename(columns={"t": "release_datetime"}).to_parquet(OUT / "chapters_meta.parquet", index=False)
    chunks.to_parquet(OUT / "chunks_input.parquet", index=False)

    print("chunks", len(chunks), "tokens (M)", round(chunks.n_tokens.sum() / 1e6, 1))
    print("chapters split", int((chap.n_tokens > CHUNK_MAX).sum()), "books truncated", int(b.first3_truncated.sum()))
    print("empty chapters", int((chap.n_tokens_raw == 0).sum()), "top10 share", round(b.top10.mean(), 3))
    assert chunks.n_tokens.max() <= CHUNK_MAX and chunks.chunk_uid.is_unique


if __name__ == "__main__":
    main()
