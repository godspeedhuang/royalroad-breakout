"""規則 baseline、logistic regression、gradient boosting。

選模流程（每個模型都一樣）：
  1. 用 2017–2021 訓練，在 2022 上選超參數（看 AUC）
  2. 選好後用 2017–2022 重新訓練
  3. 在 2023（測試）和 2024（補充）打分數，測試年完全不參與選模

輸出：out/preds_ml.parquet（每本書、每個模型的分數）、out/selection_ml.json（選到的超參數與驗證分數）
比較與信賴區間在 03_compare.py。
"""
import itertools
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import OUT, block, load, metrics, splits

b, X = load()
sp = splits(b)
y, q = b.top10.values, b.quarter.values
tr, va = sp["train"], sp["val"]
trva, ev = tr | va, sp["test"] | sp["recent"]
preds = b.loc[ev, ["fiction_id", "year", "quarter", "top10", "followers", "first_time_author", "wm", "desc_marker"]].copy()
sel = {}
col = lambda blk, name: X[blk][:, b.attrs["cols"][blk].index(name)]

# ---- 規則 baseline：不用模型，編輯憑直覺也做得到的排法 ----
tag_cols = [i for i, c in enumerate(b.attrs["cols"]["S"]) if c.startswith("tag_")]
T = X["S"][:, tag_cols]
tag_rate = (T[trva] * y[trva, None]).sum(0) / np.maximum(T[trva].sum(0), 1)  # 每個 tag 過去進前 10% 的比例
rules = {
    "rule_fast_update": -b.first3_span_days.values,          # 更新越快越優先
    "rule_hot_genre": (T @ tag_rate) / np.maximum(T.sum(1), 1),  # 熱門類型優先
    "rule_known_author": col("A", "log_prior_n"),            # 老作者優先：以前寫過越多本越優先
}
for name, s in rules.items():
    preds[name] = s[ev]
    sel[name] = {"val": metrics(y[va], s[va], q[va])}

# ---- logistic regression：一次加一個特徵區塊，看每一塊多帶來多少 ----
# 主模型是 SAT。P（前作表現）和 D（簡介）有 leak 疑慮，加上去只當上限參考（見 features.py）。
CS = [1e-4, 1e-3, 1e-2, 1e-1]
lr = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
for names in ["S", "SA", "SAT", "SATP", "SATPD", "T", "D"]:
    Z = block(X, names)
    val = {C: metrics(y[va], lr(C).fit(Z[tr], y[tr]).decision_function(Z[va]), q[va]) for C in CS}
    C = max(val, key=lambda c: val[c]["auc"])
    preds[f"lr_{names}"] = lr(C).fit(Z[trva], y[trva]).decision_function(Z[ev])
    sel[f"lr_{names}"] = {"C": C, "val": val[C], "val_auc_by_C": {str(c): round(v["auc"], 4) for c, v in val.items()}}
    print(f"lr_{names:5s} C={C:<7g} val AUC {val[C]['auc']:.3f}  recall {val[C]['recall_at_slots']:.3f}", flush=True)

# ---- gradient boosting：結構化 + 作者 + 前三章向量先用 PCA 降到 32 維，避免 1,536 維淹沒少量但強的特徵 ----
def gb_matrix(fit):
    return np.hstack([X["S"], X["A"], PCA(32, random_state=0).fit(X["T"][fit]).transform(X["T"])])

Z = gb_matrix(tr)
gb = lambda lr_, leaves: HistGradientBoostingClassifier(learning_rate=lr_, max_leaf_nodes=leaves, max_iter=300,
                                                         l2_regularization=1.0, random_state=0)
val = {p: metrics(y[va], gb(*p).fit(Z[tr], y[tr]).decision_function(Z[va]), q[va])
       for p in itertools.product([0.03, 0.1], [8, 31])}
p = max(val, key=lambda k: val[k]["auc"])
Z = gb_matrix(trva)
preds["gb_SAT"] = gb(*p).fit(Z[trva], y[trva]).decision_function(Z[ev])
sel["gb_SAT"] = {"learning_rate": p[0], "max_leaf_nodes": p[1], "val": val[p],
                  "val_auc_by_param": {str(k): round(v["auc"], 4) for k, v in val.items()}}
print(f"gb_SAT  {p} val AUC {val[p]['auc']:.3f}  recall {val[p]['recall_at_slots']:.3f}")

preds.to_parquet(OUT / "preds_ml.parquet", index=False)
(OUT / "selection_ml.json").write_text(json.dumps(sel, indent=1))
