"""Drift: fix the test year, train on data from different periods, and see whether AUC drops as the gap grows.

A fixed test year keeps the difficulty fixed. Every training window is 2 consecutive years with the same number of books (N),
so the only difference is how old the training data is. If what makes books take off keeps changing, older training data should give lower AUC.
This answers how often the model must be retrained.

Features are tested separately: structured features such as tags only (S), first-three-chapter vector only (T).
Training windows start in 2017 at the earliest (75% of 2016 books carry watermarks). C reuses the value 01_ml.py picked on the validation year.

Outputs: out/drift_auc.json, out/figs/drift_auc.png
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
        for gap in range(1, ty - FIRST_YEAR):  # training window = (ty-gap-1, ty-gap)
            pool = np.flatnonzero((yr == ty - gap - 1) | (yr == ty - gap))
            aucs = []
            for _ in range(REPS):
                i = rng.choice(pool, N, replace=False)
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Z[i], y[i])
                aucs.append(roc_auc_score(y[te], m.decision_function(Z[te])))
            res[names][ty][gap] = {"train_years": [ty - gap - 1, ty - gap], "auc": round(float(np.mean(aucs)), 4), "sd": round(float(np.std(aucs)), 4)}
        print(names, ty, {g: v["auc"] for g, v in res[names][ty].items()}, flush=True)
    # AUC change per extra year of gap: compared within each test year (each test year's own mean removed)
    g = np.array([gap for ty in TESTS for gap in res[names][ty]], float)
    a = np.concatenate([np.array([v["auc"] for v in res[names][ty].values()]) - np.mean([v["auc"] for v in res[names][ty].values()]) for ty in TESTS])
    gc = np.concatenate([np.array(list(res[names][ty]), float) - np.mean(list(res[names][ty])) for ty in TESTS])
    res[names]["auc_change_per_year_of_gap"] = round(float((gc @ a) / (gc @ gc)), 4)
    print(names, "AUC change per extra year of gap:", res[names]["auc_change_per_year_of_gap"], flush=True)

(OUT / "drift_auc.json").write_text(json.dumps({"n_train": N, "reps": REPS, **res}, indent=1))

# ---- figure: one panel per feature set, shared y axis ----
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # dataviz default palette 1–4, test years in a fixed order
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
axes[1].legend(frameon=False, fontsize=9, loc="lower left")  # the lower-left corner of the right panel is empty
fig.suptitle(f"Does an older training set predict worse? (each window: 2 years, {N:,} books)", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
(OUT / "figs").mkdir(exist_ok=True)
plt.savefig(OUT / "figs" / "drift_auc.png", dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
