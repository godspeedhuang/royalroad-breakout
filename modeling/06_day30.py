"""第 30 天版本：在第 3 章的主模型（S+A+T）上，再加開張後 30 天的更新節奏（V）。

給推薦團隊的第二輪篩選：第 3 章先推一批，第 30 天用更新節奏重排。預測時間點是第一章後 30 天
（第 3 章比第 30 天還晚的書，就在第 3 章時預測），V 只用這段時間內發布的章節，不會偷看未來。

V（7 個，計數取 log1p）：第一天、前 7 天、前 30 天各發幾章；前 30 天有幾天有更新；
最長斷更幾天；第 30 天時離上次更新幾天；前 30 天總字數。來源 data/interim/chapters_47.parquet
（全部 86,959 本書每一章的發布時間，eda/01_eda.ipynb 產生）。

另外在驗證年比較「完整作者歷史」（F）：用全部書（含不到 3 章、2016 年以前）重算作者經驗，
只看這本開始之前：前作本數、寫不到 3 章就停的前作數、前作總章數、寫作年資、開張前 30 天有沒有在寫別本。

流程和 01_ml.py 一樣：2017–2021 訓練、2022 選 C，2017–2022 重新訓練，打 2023 與 2024 一次。
第 3 章的對照用 01_ml.py 的 lr_SAT 與 rule_hot_genre 原本的分數。

ponytail: 有些書紅了以後把後面章節下架（stubbed），通常保留前面幾十章，前 30 天大多還在，
影響偏保守；沒紅的作者刪章則會讓節奏看起來比實際差，這個資料集查不到。

輸出：out/day30.json、out/preds_day30.parquet（不進 git）
"""
import glob
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import OUT, RECENT_YEAR, TEST_YEAR, block, load, metrics, recall_at_slots, splits

HERE_DATA = OUT.parent.parent / "data"
CS, B = [3e-4, 1e-3, 3e-3, 1e-2], 500
DAY = pd.Timedelta(days=1)
rng = np.random.default_rng(0)

b, X = load()
sp = splits(b)
y, q = b.top10.values, b.quarter.values
tr, va, ev = sp["train"], sp["val"], sp["test"] | sp["recent"]

# ---- 章節發布時間（全部書） ----
ch = pd.read_parquet(HERE_DATA / "interim" / "chapters_47.parquet", columns=["fiction_id", "release_datetime", "n_chars"])
ch["t"] = pd.to_datetime(ch.release_datetime, utc=True, errors="coerce")
ch = ch.dropna(subset=["t"])
ch["t0"] = ch.groupby("fiction_id").t.transform("min")
ch["d"] = (ch.t - ch.t0) / DAY  # 距第一章幾天（用 Timedelta 算，不受時間解析度影響）

# ---- V：前 30 天更新節奏 ----
w = ch[ch.d <= 30].sort_values(["fiction_id", "t"])
w["gap"] = w.groupby("fiction_id").d.diff()
g = w.groupby("fiction_id")
V = pd.DataFrame({
    "n_day1": w[w.d < 1].groupby("fiction_id").size(), "n_7d": w[w.d <= 7].groupby("fiction_id").size(), "n_30d": g.size(),
    "active_days_30d": g.d.apply(lambda s: np.unique(np.floor(s)).size), "max_gap_30d": g.gap.max(),
    "days_since_last_at_30d": 30 - g.d.max(), "chars_30d": g.n_chars.sum(),
}).reindex(b.fiction_id).fillna({"n_day1": 0, "n_7d": 0, "max_gap_30d": 30})
med30 = {"top10": float(V.n_30d[y & (tr | va)].median()), "others": float(V.n_30d[~y & (tr | va)].median())}
for c in ["n_day1", "n_7d", "n_30d", "active_days_30d", "max_gap_30d", "chars_30d"]:
    V[c] = np.log1p(V[c])
X["V"] = V.values.astype(np.float32)

# ---- F：完整作者歷史（只看這本開始之前） ----
au = pd.concat([pd.read_parquet(f, columns=["fiction_id", "author", "author_url"]).drop_duplicates("fiction_id")
                for f in sorted(glob.glob(str(HERE_DATA / "raw" / "data" / "*.parquet")))]).drop_duplicates("fiction_id")
au["author_id"] = au.author_url.fillna(au.author).map(lambda s: hashlib.sha1(str(s).encode()).hexdigest()[:12])  # 同 embedding/01_prepare.py
cc = ch[["fiction_id", "t"]].merge(au[["fiction_id", "author_id"]], on="fiction_id")
tgt = b[["fiction_id", "author_id"]].merge(ch.groupby("fiction_id").t0.first().reset_index(), on="fiction_id")
m = tgt.merge(cc.rename(columns={"fiction_id": "pf"}), on="author_id")
m = m[(m.pf != m.fiction_id) & (m.t < m.t0)]
pb = m.groupby(["fiction_id", "pf"]).size().rename("nb").reset_index()
F = pd.DataFrame({
    "prior_books_all": pb.groupby("fiction_id").size(), "prior_short": pb[pb.nb < 3].groupby("fiction_id").size(),
    "prior_chapters": m.groupby("fiction_id").size(),
    "tenure_days": (tgt.set_index("fiction_id").t0 - m.groupby("fiction_id").t.min()) / DAY,
    "active_30d_before": m[m.t >= m.t0 - 30 * DAY].groupby("fiction_id").size(),
}).reindex(b.fiction_id).fillna(0)
hidden_prior = float((F.prior_books_all[b.first_time_author.values] > 0).mean())
X["F"] = np.log1p(F.clip(lower=0)).values.astype(np.float32)


# ---- 驗證年選 C，再用 2017–2022 重新訓練 ----
def fit(Z, rows):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_, max_iter=3000)).fit(Z[rows], y[rows])


ft = b.first_time_author.values
res = {"val": {}, "selected_C": {}}
for name in ["SAT", "SAFT", "SATV", "V"]:
    Z = block(X, name)
    val = {}
    for C_ in CS:
        s = fit(Z, tr).decision_function(Z)
        val[C_] = {"auc": metrics(y[va], s[va], q[va])["auc"], "recall": recall_at_slots(y[va], s[va], q[va]),
                   "recall_first_time": recall_at_slots(y[va & ft], s[va & ft], q[va & ft])}
    C_ = max(val, key=lambda c: val[c]["auc"])
    res["val"][f"lr_{name}"] = {k: round(float(v), 4) for k, v in val[C_].items()}
    res["selected_C"][f"lr_{name}"] = C_
    if name == "SATV":
        Z_main, C_day30 = Z, C_
    print(f"val lr_{name:5s} C={C_:g} {res['val'][f'lr_{name}']}", flush=True)

C_ = C_day30
day30 = fit(Z_main, tr | va).decision_function(Z_main)
preds = b.loc[ev, ["fiction_id", "year", "quarter", "top10", "first_time_author", "followers"]].copy()
preds["day30"] = day30[ev]
preds = preds.merge(pd.read_parquet(OUT / "preds_ml.parquet")[["fiction_id", "lr_SAT", "rule_hot_genre"]], on="fiction_id")
preds.to_parquet(OUT / "preds_day30.parquet", index=False)

# ---- 測試年與 2024：全部／第一次寫書／有前作 ----
MODELS = ["rule_hot_genre", "lr_SAT", "day30"]
r4 = lambda x: round(float(x), 4)
for key, yr in [("test", TEST_YEAR), ("recent_2024", RECENT_YEAR)]:
    d = preds[preds.year == yr]
    groups = {"all": np.ones(len(d), bool), "first_time_author": d.first_time_author.values, "returning_author": ~d.first_time_author.values}
    res[key] = {gname: {"n": int(gm.sum()), **{mm: {k: r4(v) for k, v in metrics(d.top10.values[gm], d[mm].values[gm], d.quarter.values[gm]).items()}
                                               for mm in MODELS}} for gname, gm in groups.items()}

# ---- 成對 bootstrap：第 30 天 − 第 3 章（百分點），在每一季裡重抽書 ----
test = preds[preds.year == TEST_YEAR].reset_index(drop=True)
res["test_day30_minus_main_pp"] = {}
for gname, gm in {"all": np.ones(len(test), bool), "first_time_author": test.first_time_author.values}.items():
    d = test[gm].reset_index(drop=True)
    by_q = [np.flatnonzero(d.quarter.values == qq) for qq in d.quarter.unique()]
    diff = np.empty(B)
    for i in range(B):
        s = d.iloc[np.concatenate([rng.choice(ix, len(ix)) for ix in by_q])]
        diff[i] = recall_at_slots(s.top10.values, s.day30.values, s.quarter.values) - recall_at_slots(s.top10.values, s.lr_SAT.values, s.quarter.values)
    point = recall_at_slots(d.top10.values, d.day30.values, d.quarter.values) - recall_at_slots(d.top10.values, d.lr_SAT.values, d.quarter.values)
    res["test_day30_minus_main_pp"][gname] = {"diff": round(100 * point, 1), "ci95": [round(100 * x, 1) for x in np.percentile(diff, [2.5, 97.5])]}

nq = test.quarter.nunique()
hits = lambda mm: sum(gq.nlargest(int(gq.top10.sum()), mm).top10.sum() for _, gq in test.groupby("quarter")) / nq
res["impact_test"] = {"slots_per_quarter": round(test.top10.sum() / nq), "hits_per_quarter": {mm: round(float(hits(mm)), 1) for mm in MODELS}}
res["describe_trainval"] = {"median_chapters_first_30d": med30, "first_time_by_sample_but_has_prior_book": round(hidden_prior, 4)}
(OUT / "day30.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ["test_day30_minus_main_pp", "impact_test", "describe_trainval"]}, indent=1))
print("test all:", res["test"]["all"], "\n2024 all:", res["recent_2024"]["all"])
