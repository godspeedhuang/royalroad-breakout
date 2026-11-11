"""檢查 Royal Road 防盜浮水印對趨勢結果的影響。

Royal Road 會在章節裡插入防盜句子（例如 "Report any appearances on Amazon."）。
2016 年的章節有一半含這種句子，其他年份約 5–7%，可能造成假的「2016 年很不一樣」。
這裡把前三章含浮水印的書排除，重算主要數字，和全部書比較。

輸出：out/watermark_check.json、out/watermark_books.csv（含浮水印的書的 fiction_id）
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = Path(__file__).parent / "out"
TERMS = ["amazon", "unlawfully", "unauthorized", "royal road", "royalroad"]
EARLY, LATE = (2016, 2018), (2021, 2023)
YEARS = list(range(2016, 2025))
rng = np.random.default_rng(0)
unit = lambda v: v / np.linalg.norm(v)
cos = lambda a, c: float(unit(a) @ unit(c))

c = pd.read_parquet(OUT / "chapters_meta.parquet", columns=["fiction_id", "text"])
low = c.text.str.lower()
c["wm"] = np.logical_or.reduce([low.str.contains(t, regex=False).values for t in TERMS])
flag = c.groupby("fiction_id").wm.any()
pd.DataFrame({"fiction_id": flag.index[flag.values]}).to_csv(OUT / "watermark_books.csv", index=False)  # 給 modeling/ 用

t = pq.read_table(OUT / "books.parquet")
b = t.drop(["book_vec", "chap1_vec", "desc_vec"]).to_pandas()
V = t.column("book_vec").combine_chunks().flatten().to_numpy().reshape(len(b), -1).astype(np.float32)
b["wm"] = b.fiction_id.map(flag).fillna(False).values
yr, top = b.year.values, b.top10.values


def stats(keep):
    m = lambda mask: V[mask & keep].mean(0)
    cents = {y: m(yr == y) for y in YEARS}
    dirs = {y: m((yr == y) & top) - m((yr == y) & ~top) for y in YEARS}

    def rel(y):
        idx = np.flatnonzero((yr == y) & keep); vals = []
        for _ in range(60):
            p = rng.permutation(idx); h = len(p) // 2; a, d = p[:h], p[h:]
            vals.append(cos(V[a][top[a]].mean(0) - V[a][~top[a]].mean(0), V[d][top[d]].mean(0) - V[d][~top[d]].mean(0)))
        r = float(np.mean(vals)); return 2 * r / (1 + r)
    R = {y: rel(y) for y in YEARS}
    gap = {g: float(np.mean([cos(dirs[y], dirs[y + g]) / np.sqrt(R[y] * R[y + g]) for y in YEARS if y + g in YEARS])) for g in range(1, 9)}
    e, l = (yr >= EARLY[0]) & (yr <= EARLY[1]), (yr >= LATE[0]) & (yr <= LATE[1])
    d_sup, d_hot = m(l) - m(e), m(l & top) - m(e & top)
    d_pref = (m(l & top) - m(l & ~top)) - (m(e & top) - m(e & ~top))
    return {"n_by_year": {y: int(((yr == y) & keep).sum()) for y in YEARS},
            "supply_from_2016": {y: round(1 - cos(cents[2016], cents[y]), 4) for y in YEARS},
            "supply_from_2017": {y: round(1 - cos(cents[2017], cents[y]), 4) for y in YEARS[1:]},
            "dir_by_gap_disatt": {k: round(v, 3) for k, v in gap.items()},
            "cos_supply_hot": round(cos(d_sup, d_hot), 3), "cos_supply_pref": round(cos(d_sup, d_pref), 3),
            "norm": {"supply": round(float(np.linalg.norm(d_sup)), 4), "hot": round(float(np.linalg.norm(d_hot)), 4), "pref": round(float(np.linalg.norm(d_pref)), 4)}}


res = {"share_books_with_watermark_by_year": b.groupby("year").wm.mean().round(3).to_dict(),
       "all_books": stats(np.ones(len(b), bool)), "no_watermark": stats(~b.wm.values)}
(OUT / "watermark_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps(res, ensure_ascii=False, indent=0, default=str))
