"""口味漂移：固定測試年，換不同年代的訓練資料，看 AUC 會不會隨年代差距下降。

測試年固定，題目難度就固定。每個訓練窗都是連續 2 年、抽一樣多本（N），
所以唯一的差別是「訓練資料有多舊」。如果讓書紅的東西一直在變，訓練資料越舊，AUC 應該越低。
這回答的是：模型多久要重新訓練一次。

特徵分開看：只用 tags 等結構化特徵（S）、只用前三章向量（T）。
訓練窗最早從 2017 年起（2016 年 75% 的書含浮水印）。C 沿用 01_ml.py 在驗證年選到的值。

輸出：out/drift_auc.json、out/figs/drift_auc.png
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import OUT, load

N, REPS, FIRST_YEAR, TESTS = 5000, 5, 2017, [2020, 2021, 2022, 2023]
rng = np.random.default_rng(0)
b, X = load()
y, yr = b.top10.values, b.year.values
sel = json.loads((OUT / "selection_ml.json").read_text())

res = {}
for names in ["S", "T"]:
    Z, C = X[names], sel[f"lr_{names}"]["C"]
    res[names] = {}
    for ty in TESTS:
        te = yr == ty
        res[names][ty] = {}
        for gap in range(1, ty - FIRST_YEAR):  # 訓練窗 = (ty-gap-1, ty-gap)
            pool = np.flatnonzero((yr == ty - gap - 1) | (yr == ty - gap))
            aucs = []
            for _ in range(REPS):
                i = rng.choice(pool, N, replace=False)
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Z[i], y[i])
                aucs.append(roc_auc_score(y[te], m.decision_function(Z[te])))
            res[names][ty][gap] = {"train_years": [ty - gap - 1, ty - gap], "auc": round(float(np.mean(aucs)), 4), "sd": round(float(np.std(aucs)), 4)}
        print(names, ty, {g: v["auc"] for g, v in res[names][ty].items()}, flush=True)
    # 每多隔一年 AUC 變多少：同一測試年內比較（扣掉各測試年自己的平均）
    g = np.array([gap for ty in TESTS for gap in res[names][ty]], float)
    a = np.concatenate([np.array([v["auc"] for v in res[names][ty].values()]) - np.mean([v["auc"] for v in res[names][ty].values()]) for ty in TESTS])
    gc = np.concatenate([np.array(list(res[names][ty]), float) - np.mean(list(res[names][ty])) for ty in TESTS])
    res[names]["auc_change_per_year_of_gap"] = round(float((gc @ a) / (gc @ gc)), 4)
    print(names, "每多隔一年 AUC 變化：", res[names]["auc_change_per_year_of_gap"], flush=True)

(OUT / "drift_auc.json").write_text(json.dumps({"n_train": N, "reps": REPS, **res}, indent=1))

# ---- 圖：兩個特徵各一格，同一個 y 軸 ----
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # dataviz 預設色盤 1–4，測試年依固定順序
TITLE = {"S": "Tags and structure only", "T": "First-3-chapter text only"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, names in zip(axes, ["S", "T"]):
    for ty, c in zip(TESTS, COLORS):
        gaps = list(res[names][ty]); aucs = [res[names][ty][g]["auc"] for g in gaps]
        ax.plot(gaps, aucs, color=c, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=f"test year {ty}")
        ax.text(gaps[-1] + 0.12, aucs[-1], str(ty), color="#52514e", fontsize=9, va="center")
    ax.set_title(f"{TITLE[names]}  ({res[names]['auc_change_per_year_of_gap']:+.3f} AUC per extra year)", loc="left", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Years between training window and test year", fontsize=9.5, color="#52514e")
    ax.set_xticks(range(1, 6)); ax.set_xlim(0.7, 5.7)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#0b0b0b", alpha=0.08); ax.tick_params(colors="#52514e")
    ax.set_facecolor("#fcfcfb")
axes[0].set_ylabel("AUC on the fixed test year", fontsize=9.5, color="#52514e")
axes[1].legend(frameon=False, fontsize=9, loc="lower left")  # 右邊那格左下角是空的
fig.suptitle(f"Does an older training set predict worse? (each window: 2 years, {N:,} books)", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
(OUT / "figs").mkdir(exist_ok=True)
plt.savefig(OUT / "figs" / "drift_auc.png", dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
