"""Rule baselines, logistic regression, gradient boosting.

Model selection (the same for every model):
  1. train on 2017–2021, pick hyperparameters on 2022 (by AUC)
  2. refit on 2017–2022
  3. score 2023 (test) and 2024 (supplementary); the test year plays no part in model selection

Outputs: out/preds_ml.parquet (every model's score for every book), out/selection_ml.json (chosen hyperparameters and validation scores)
Comparison and confidence intervals are in 03_compare.py.
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

# ---- Rule baselines: rankings an editor could use without a model ----
tag_cols = [i for i, c in enumerate(b.attrs["cols"]["S"]) if c.startswith("tag_")]
T = X["S"][:, tag_cols]
tag_rate = (T[trva] * y[trva, None]).sum(0) / np.maximum(T[trva].sum(0), 1)  # past share of each tag's books that reached the top 10%
rules = {
    "rule_fast_update": -b.first3_span_days.values,          # faster updates first
    "rule_hot_genre": (T @ tag_rate) / np.maximum(T.sum(1), 1),  # popular genres first
    "rule_known_author": col("A", "log_prior_n"),            # experienced authors first: more earlier books ranks higher
}
for name, s in rules.items():
    preds[name] = s[ev]
    sel[name] = {"val": metrics(y[va], s[va], q[va])}

# ---- Logistic regression: add one feature block at a time to see what each adds ----
# The main model is SAT. P (earlier books' success) and D (blurb) are leak-prone and serve only as an upper bound (see features.py).
CS = [1e-4, 1e-3, 1e-2, 1e-1]
lr = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
for names in ["S", "SA", "SAT", "SATP", "SATPD", "T", "D"]:
    Z = block(X, names)
    val = {C: metrics(y[va], lr(C).fit(Z[tr], y[tr]).decision_function(Z[va]), q[va]) for C in CS}
    C = max(val, key=lambda c: val[c]["auc"])
    preds[f"lr_{names}"] = lr(C).fit(Z[trva], y[trva]).decision_function(Z[ev])
    sel[f"lr_{names}"] = {"C": C, "val": val[C], "val_auc_by_C": {str(c): round(v["auc"], 4) for c, v in val.items()}}
    print(f"lr_{names:5s} C={C:<7g} val AUC {val[C]['auc']:.3f}  recall {val[C]['recall_at_slots']:.3f}", flush=True)

# ---- Gradient boosting: structured + author + first-three-chapter vector reduced to 32 dims with PCA, so 1,536 dims do not drown a few strong features ----
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
