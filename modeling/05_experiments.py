"""驗證年實驗：想讓主模型更好、但都只在 2022 驗證年比較，測試年 2023 完全不碰。

每一組都用 2017–2021 訓練、2022 打分數，報 AUC 與 Recall@名額（名額命中率）。
LR 的 C 在同一組格點裡用驗證年 AUC 挑。2022 年有 571 本熱門書，命中率的標準誤差約 0.02，
差距小於這個數字的結果都當成「沒差」。

1. dims      ：只取 book_vec 前 k 維（v3 embedding 截短後重新正規化 = 較小的版本），看維度的邊際效益，
               用來推估換成 text-embedding-3-large（3,072 維）值不值得。
2. chapters  ：三章怎麼組成文字特徵。token 加權平均（現在的 T）、等權平均、三章並列（3 × 1,536）、單一章。
3. knn       ：鄰居特徵。每本書找 cosine 最像的 50 本「至少早 365 天開始連載、不同作者」的書，
               算其中進前 10% 的比例、平均同季排名、依相似度加權的比例。
4. style     ：從前三章原文算 17 個文風統計（句長、對話佔比、[ ] 面板、小寫 i……）。
5. target_loss：改訓練目標或 loss。Ridge 猜 0/1、Ridge 猜同季排名（連續目標）、
               同季內配對的排序 loss（紅 vs 沒紅；依排名差 ≥ 0.2），後者是 RankNet 的線性版。
6. test_slot_quality：唯一用到 2023 的部分，只描述已定案的主模型與規則，不拿來選任何東西：
               版位裡的書有多少比例在同季前 20%、沒命中的書排在哪。

需要：embedding/out/books.parquet、embedding/out/chapters.parquet（逐章向量與原文，不在 HF，
要用 embedding/01–03 重建）、out/preds_ml.parquet（01_ml.py 的輸出）。
輸出：out/experiments_val.json。文風特徵快取在 out/style_trainval.parquet。約 8 分鐘（Mac CPU）。
"""
import json
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import EMB, OUT, block, load, recall_at_slots, splits

CS = [3e-4, 1e-3, 3e-3, 1e-2]
t0 = time.time()
b, X = load()
sp = splits(b)
y, q, pct = b.top10.values, b.quarter.values, b.pct_in_quarter.values
tr, va = sp["train"], sp["val"]
R = {}


def ev(s):
    return {"auc": round(float(roc_auc_score(y[va], s)), 4), "recall": round(float(recall_at_slots(y[va], s, q[va])), 4)}


def lr(Z, Cs=CS):
    """各 C 訓練一次，回傳驗證年 AUC 最高的那組。"""
    res = []
    for C in Cs:
        s = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Z[tr], y[tr]).decision_function(Z[va])
        res.append({**ev(s), "C": C})
    return max(res, key=lambda r: r["auc"])


def rec(group, name, r):
    R.setdefault(group, {})[name] = r
    print(f"[{time.time() - t0:4.0f}s] {group:11s} {name:28s} {r}", flush=True)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


SA, T = block(X, "SA"), X["T"]

# ---- 1. 維度 ----
for k in [128, 256, 512, 1024, 1536]:
    rec("dims", f"T first {k}", lr(unit(T[:, :k])))

# ---- 2. 三章怎麼組 ----
ch = pd.read_parquet(EMB / "chapters.parquet", columns=["fiction_id", "chapter_idx", "vector"])
row = pd.Series(np.arange(len(b)), index=b.fiction_id.values).reindex(ch.fiction_id.values).values
ok = ~np.isnan(row)
V = np.zeros((len(b), 3, 1536), np.float32)
have = np.zeros((len(b), 3), bool)
r_, k_ = row[ok].astype(int), ch.chapter_idx.values[ok] - 1
V[r_, k_] = np.stack(ch.vector.values[ok])
have[r_, k_] = True
for i in range(3):  # 少數書缺某一章的向量（截到 20,000 token 上限後沒剩），用書向量補
    V[~have[:, i], i] = T[~have[:, i]]
del ch
R["chapters_note"] = {"books_missing_a_chapter_vector": int((~have.all(1)).sum())}
eq = unit(V.mean(1))
cat = V.reshape(len(b), -1)
for name, Z in [("token-weighted mean (T)", T), ("equal mean", eq), ("concat 3x1536", cat),
                ("chapter 1 only", V[:, 0]), ("chapter 2 only", V[:, 1]), ("chapter 3 only", V[:, 2])]:
    rec("chapters", f"text only: {name}", lr(Z, [1e-4] + CS))
for name, Z in [("S+A (no text)", SA), ("S+A+T (main)", block(X, "SAT")), ("S+A+equal mean", np.hstack([SA, eq])),
                ("S+A+concat", np.hstack([SA, cat])), ("S+A+chapter 1", np.hstack([SA, V[:, 0]])),
                ("S+A+T+chapter 1", np.hstack([SA, T, V[:, 0]]))]:
    rec("chapters", name, lr(Z, [1e-4] + CS))
del cat

# ---- 3. 鄰居特徵 ----
K, LAG = 50, 365
day = b.first_release.values.astype("datetime64[D]").astype(np.int64)
auth = b.author_id.astype("category").cat.codes.values
rows = np.flatnonzero(tr | va)
F = np.zeros((len(b), 3))
for s in range(0, len(rows), 2000):
    r = rows[s:s + 2000]
    sim = T[r] @ T.T
    sim[(day[None, :] > day[r][:, None] - LAG) | (auth[None, :] == auth[r][:, None])] = -np.inf
    nn = np.argpartition(-sim, K, axis=1)[:, :K]
    w = np.take_along_axis(sim, nn, 1)
    okw = np.isfinite(w)
    n = np.maximum(okw.sum(1), 1)
    wp = np.where(okw, np.maximum(w, 0), 0)
    F[r] = np.c_[(y[nn] * okw).sum(1) / n, (pct[nn] * okw).sum(1) / n, (wp * y[nn]).sum(1) / np.maximum(wp.sum(1), 1e-9)]
    F[r[okw.sum(1) == 0]] = [0.1, 0.5, 0.1]  # 最早的書沒有夠早的鄰居，給全體平均
rec("knn", "neighbour top10 share alone", ev(F[va, 0]))
rec("knn", "S+A+knn", lr(np.hstack([SA, F])))
rec("knn", "S+A+T+knn", lr(np.hstack([block(X, "SAT"), F])))

# ---- 4. 文風統計 ----
STYLE = OUT / "style_trainval.parquet"
if not STYLE.exists():
    keep = b.fiction_id[tr | va].tolist()
    txt = pq.read_table(EMB / "chapters.parquet", columns=["fiction_id", "chapter_idx", "text"],
                        filters=[("fiction_id", "in", keep)]).to_pandas()
    P = {k: re.compile(v) for k, v in {
        "words": r"[A-Za-z']+", "sent": r"[.!?]+(?=\s|$|[\"”’])", "para": r"\n\s*\n|\n",
        "excl": r"!", "ques": r"\?", "ellip": r"\.\.\.|…", "bracket": r"\[[^\]\n]{1,200}\]",
        "an": r"(?i)\b(a/n|author'?s? note)\b", "first_i": r"\bI\b", "third": r"\b(?:he|she|they)\b",
        "lower_i": r"(?<=\s)i(?=\s)", "ly": r"\b\w+ly\b", "caps": r"\b[A-Z]{3,}\b",
        "dialog": r"[\"“][^\"”\n]{1,600}[\"”]", "dash": r"--|—|–", "semi": r";"}.items()}
    out = []
    for fid, idx, t in zip(txt.fiction_id.values, txt.chapter_idx.values, txt.text.values):
        t = t or ""
        d = {k: len(p.findall(t)) for k, p in P.items() if k != "dialog"}
        d.update(fid=fid, chars=len(t), dialog_chars=sum(map(len, P["dialog"].findall(t))),
                 letters=sum(map(len, P["words"].findall(t))))
        if idx == 1:  # 詞彙豐富度：第一章前 1,000 字的 type-token ratio
            w = [x.lower() for x in P["words"].findall(t)[:1000]]
            d["ttr"] = len(set(w)) / len(w) if len(w) >= 200 else np.nan
        out.append(d)
    c = pd.DataFrame(out)
    g = c.groupby("fid").sum(min_count=1)
    g["ttr"] = c.groupby("fid").ttr.max()
    W = g.words.clip(lower=1)
    pk = {f"{k}_pk": g[k] * 1000 / W for k in ["excl", "ques", "ellip", "bracket", "first_i", "third", "lower_i", "ly", "caps", "dash", "semi"]}
    pd.DataFrame({"sent_len": W / g.sent.clip(lower=1), "para_len": W / g.para.clip(lower=1), "word_len": g.letters / W,
                  **pk, "has_author_note": (g["an"] > 0).astype(float),
                  "dialog_share": g.dialog_chars / g.chars.clip(lower=1), "ttr": g.ttr}).to_parquet(STYLE)
f = pd.read_parquet(STYLE).reindex(b.fiction_id.values)
f["ttr_missing"] = f.ttr.isna().astype(float)
f["ttr"] = f.ttr.fillna(f.ttr.median())
for col in f.columns:
    if col.endswith("_pk") or col in ("para_len", "sent_len"):
        f[col] = np.log1p(f[col].clip(lower=0))
f = f.fillna(0)
R["style_univariate_auc"] = dict(sorted(((c, round(float(roc_auc_score(y[va], f[c].values[va])), 3)) for c in f.columns),
                                        key=lambda kv: -abs(kv[1] - 0.5)))
Fs = f.values
rec("style", "style alone", lr(Fs))
rec("style", "S+A+style", lr(np.hstack([SA, Fs])))
rec("style", "S+A+T+style", lr(np.hstack([block(X, "SAT"), Fs])))

# ---- 5. 訓練目標與 loss（S+A+T，標準化只用訓練年 fit） ----
SAT = block(X, "SAT")
Z = StandardScaler().fit(SAT[tr]).transform(SAT).astype(np.float32)
del SAT
rec("target_loss", "LR on 0/1 (main)", lr(Z))
for name, target in [("Ridge on 0/1", y.astype(float)), ("Ridge on quarter rank", pct)]:
    res = [{**ev(Ridge(alpha=a).fit(Z[tr], target[tr]).predict(Z[va])), "lambda": a} for a in [1e3, 1e4, 3e4, 1e5]]
    rec("target_loss", name, max(res, key=lambda r: r["auc"]))
itr = np.flatnonzero(tr)


def pairs(kind, seed, m=6000):
    """同一季內配對：binary = 一本紅的 vs 一本沒紅的；graded = 同季排名差 ≥ 0.2 的兩本。"""
    rng = np.random.default_rng(seed)
    A, B = [], []
    for qq in np.unique(q[itr]):
        idx = itr[q[itr] == qq]
        if kind == "binary":
            A.append(rng.choice(idx[y[idx]], m)); B.append(rng.choice(idx[~y[idx]], m))
        else:
            i, j = rng.choice(idx, 3 * m), rng.choice(idx, 3 * m)
            keep = np.abs(pct[i] - pct[j]) >= 0.2
            i, j = i[keep][:m], j[keep][:m]
            A.append(np.where(pct[i] > pct[j], i, j)); B.append(np.where(pct[i] > pct[j], j, i))
    A, B = np.concatenate(A), np.concatenate(B)
    flip = rng.random(len(A)) < 0.5
    return np.where(flip[:, None], Z[B] - Z[A], Z[A] - Z[B]), ~flip


for kind in ["binary", "graded"]:
    by_C = {}
    for seed in range(3):
        D, lab = pairs(kind, seed)
        for C in [3e-5, 1e-4, 3e-4]:
            w = LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(D, lab).coef_[0]
            by_C.setdefault(C, []).append(ev(Z[va] @ w))
    C = max(by_C, key=lambda c: np.mean([r["auc"] for r in by_C[c]]))
    rs = by_C[C]
    rec("target_loss", f"pairwise rank ({kind})", {"auc": round(float(np.mean([r["auc"] for r in rs])), 4),
                                                   "recall": round(float(np.mean([r["recall"] for r in rs])), 4),
                                                   "recall_by_seed": [r["recall"] for r in rs], "C": C})

# ---- 6. 測試年描述：版位裡的書離前 10% 有多近（主模型與規則已定案，這裡不選任何東西） ----
p = pd.read_parquet(OUT / "preds_ml.parquet", columns=["fiction_id", "rule_hot_genre", "lr_SAT"])
p = p.merge(b[["fiction_id", "year", "quarter", "top10", "pct_in_quarter"]], on="fiction_id")
p = p[p.year == 2023]
for m in ["rule_hot_genre", "lr_SAT"]:
    picks = pd.concat([g.nlargest(int(g.top10.sum()), m) for _, g in p.groupby("quarter")])
    miss = picks[~picks.top10]
    rec("test_slot_quality", m, {"slots": len(picks), "hit_top10": round(float(picks.top10.mean()), 3),
                                 "in_top20": round(float((picks.pct_in_quarter >= 0.8).mean()), 3),
                                 "misses_median_pct": round(float(miss.pct_in_quarter.median()), 3)})

R["val_positives"] = int(y[va].sum())
(OUT / "experiments_val.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
print(f"done in {time.time() - t0:.0f}s")
