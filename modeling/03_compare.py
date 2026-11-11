"""比較所有模型在測試年（2023）的表現。

  1. 每個模型的 AUC、AP、Recall@名額，Recall 附 bootstrap 95% 信賴區間
  2. 每加一個特徵區塊多了多少（成對 bootstrap）
  3. 分組：全部／第一次寫書的作者／有前作的作者
  4. leak 檢查：拿掉含浮水印的書、拿掉簡介有事後字眼的書，分數掉多少
  5. 影響換算：每季固定名額下，模型比規則多挑中幾本熱門書

主模型不是看測試分數選的：從沒有 leak 疑慮的候選（lr_SAT、gb_SAT、mlp_SAT）裡挑驗證年 AUC 最高的。

輸出：out/results.json、out/results.md、out/figs/model_comparison.png
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import OUT, RECENT_YEAR, TEST_YEAR, metrics, recall_at_slots

B = 500
rng = np.random.default_rng(0)
LABEL = {
    "rule_fast_update": "Rule: fastest updates first", "rule_hot_genre": "Rule: popular genres first",
    "rule_known_author": "Rule: experienced authors first",
    "lr_S": "LR: tags + structure", "lr_SA": "LR: + author experience", "lr_SAT": "LR: + first-3-chapter text",
    "lr_T": "LR: chapter text only", "gb_SAT": "Boosting: tags + author + text",
    "mlp_T": "MLP: chapter text only", "mlp_SAT": "MLP: tags + author + text",
    "lr_SATP": "LR: + prior-book success", "lr_SATPD": "LR: + blurb", "lr_D": "LR: blurb only",
}
KIND = {m: "Rule" if m.startswith("rule") else "Deep learning" if m.startswith("mlp") else "Machine learning" for m in LABEL}
LEAKY = {"lr_SATP", "lr_SATPD", "lr_D"}  # 用到 P 或 D，只當上限參考
COLOR = {"Machine learning": "#2a78d6", "Deep learning": "#eb6834", "Rule": "#1baf7a"}  # dataviz 預設色盤 1–3

p = pd.read_parquet(OUT / "preds_ml.parquet").merge(pd.read_parquet(OUT / "preds_dl.parquet"), on="fiction_id")
models = list(LABEL)
test = p[p.year == TEST_YEAR].reset_index(drop=True)
recent = p[p.year == RECENT_YEAR].reset_index(drop=True)
sel = json.loads((OUT / "selection_ml.json").read_text()) | json.loads((OUT / "selection_dl.json").read_text())
main = max(["lr_SAT", "gb_SAT", "mlp_SAT"], key=lambda m: sel[m]["val"]["auc"])
rule = max([m for m in models if m.startswith("rule")], key=lambda m: sel[m]["val"]["recall_at_slots"])

table = lambda d, ms=models: {m: metrics(d.top10.values, d[m].values, d.quarter.values) for m in ms}
r3 = lambda x: round(float(x), 3)

# ---- 1. bootstrap：在每一季裡重抽書，保留「每季名額」的結構 ----
by_q = [np.flatnonzero(test.quarter.values == q) for q in test.quarter.unique()]
boot = {m: np.empty(B) for m in models}
for i in range(B):
    s = test.iloc[np.concatenate([rng.choice(ix, len(ix)) for ix in by_q])]
    for m in models:
        boot[m][i] = recall_at_slots(s.top10.values, s[m].values, s.quarter.values)
ci = lambda v: [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))]

res = {"test_year": TEST_YEAR, "n_test": len(test), "n_positive": int(test.top10.sum()), "main_model": main, "best_rule": rule,
       "test": {m: {**{k: r3(v) for k, v in t.items()}, "recall_ci95": ci(boot[m]), "leak_prone": m in LEAKY} for m, t in table(test).items()},
       "recent_2024": {m: {k: r3(v) for k, v in t.items()} for m, t in table(recent).items()}}

# ---- 2. 每一步多了多少（百分點） ----
steps = [("model vs best rule", main, rule), ("author experience", "lr_SA", "lr_S"), ("chapter text", "lr_SAT", "lr_SA"),
         ("boosting vs LR", "gb_SAT", "lr_SAT"), ("MLP vs LR", "mlp_SAT", "lr_SAT"),
         ("prior-book success (leak-prone)", "lr_SATP", "lr_SAT"), ("blurb (leak-prone)", "lr_SATPD", "lr_SATP")]
res["increments_pp"] = {name: {"a": a, "b": c, "diff": round(100 * (res["test"][a]["recall_at_slots"] - res["test"][c]["recall_at_slots"]), 1),
                               "ci95": [round(100 * x, 1) for x in np.percentile(boot[a] - boot[c], [2.5, 97.5])]} for name, a, c in steps}

# ---- 3. 分組 ----
groups = {"all": np.ones(len(test), bool), "first_time_author": test.first_time_author.values, "returning_author": ~test.first_time_author.values}
res["by_group"] = {g: {"n": int(m.sum()), "n_positive": int(test.top10[m].sum()),
                       **{k: {a: r3(b) for a, b in v.items()} for k, v in table(test[m], [rule, "lr_S", "lr_SA", main]).items()}} for g, m in groups.items()}

# ---- 4. leak 檢查 ----
flag_auc = lambda f: r3(metrics(test.top10.values, test[f].values.astype(float), test.quarter.values)["auc"])
res["leak_checks"] = {
    "watermark": {"share_of_test": r3(test.wm.mean()), "top10_rate_if_flagged": r3(test.top10[test.wm].mean()), "auc_of_flag_alone": flag_auc("wm"),
                  "without_flagged_books": {m: {k: r3(v) for k, v in t.items()} for m, t in table(test[~test.wm], ["lr_SA", "lr_T", main]).items()}},
    "blurb_marker": {"share_of_test": r3(test.desc_marker.mean()), "top10_rate_if_flagged": r3(test.top10[test.desc_marker].mean()), "auc_of_flag_alone": flag_auc("desc_marker"),
                     "without_flagged_books": {m: {k: r3(v) for k, v in t.items()} for m, t in table(test[~test.desc_marker], ["lr_SAT", "lr_D", "lr_SATPD"]).items()}},
}

# ---- 5. 影響換算：每季名額 = 該季熱門書數 ----
def picks(m):
    return pd.concat([g.nlargest(int(g.top10.sum()), m) for _, g in test.groupby("quarter")])
nq = test.quarter.nunique()
pm, pr = picks(main), picks(rule)
res["impact"] = {
    "quarters": nq, "books_per_quarter": round(len(test) / nq), "slots_per_quarter": round(test.top10.sum() / nq),
    "hits_per_quarter": {"random": round(0.1 * test.top10.sum() / nq, 1), rule: round(pr.top10.sum() / nq, 1), main: round(pm.top10.sum() / nq, 1)},
    # 被推薦的書最後累積的 followers 總和（事後數字，只用來表達「推薦位有沒有給對書」）
    "eventual_followers_of_picks_per_quarter": {"random": round(test.followers.mean() * test.top10.sum() / nq), rule: round(pr.followers.sum() / nq),
                                                 main: round(pm.followers.sum() / nq), "oracle": round(test.followers[test.top10].sum() / nq)},
}
(OUT / "results.json").write_text(json.dumps(res, indent=1))

# ---- 表 ----
rows = ["| Model | AUC | AP | Recall@slots (95% CI) | 2024 recall |", "|---|---|---|---|---|"]
for m in models:
    t = res["test"][m]
    rows.append(f"| {LABEL[m]}{' †' if m in LEAKY else ''}{' **(main)**' if m == main else ''} | {t['auc']:.3f} | {t['ap']:.3f} | "
                f"{t['recall_at_slots']:.3f} ({t['recall_ci95'][0]:.3f}–{t['recall_ci95'][1]:.3f}) | {res['recent_2024'][m]['recall_at_slots']:.3f} |")
(OUT / "results.md").write_text("\n".join(rows) + "\n\n† uses information that may postdate chapter 3 (see README); upper bound only. Random = 0.10.\n")

# ---- 圖：一個量（recall）、三類模型；leak 疑慮的用斜線填 ----
order = sorted(models, key=lambda m: res["test"][m]["recall_at_slots"])
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for i, m in enumerate(order):
    v, (lo, hi) = res["test"][m]["recall_at_slots"], res["test"][m]["recall_ci95"]
    ax.barh(i, v, height=0.62, color=COLOR[KIND[m]], alpha=0.45 if m in LEAKY else 1, hatch="///" if m in LEAKY else None,
            edgecolor="white", linewidth=0)
    ax.plot([lo, hi], [i, i], color="#0b0b0b", lw=1.2)
    ax.text(hi + 0.008, i, f"{v:.2f}", va="center", fontsize=9, color="#0b0b0b")
ax.axvline(0.1, color="#52514e", lw=1, ls=(0, (3, 3)))
ax.text(0.1, len(order) - 0.35, "random", ha="center", fontsize=8.5, color="#52514e")
ax.set_yticks(range(len(order)), [LABEL[m] + (" †" if m in LEAKY else "") for m in order], fontsize=9.5)
ax.get_yticklabels()[order.index(main)].set_fontweight("bold")
ax.set_xlabel(f"Share of recommendation slots that go to a future top-10% book ({TEST_YEAR} test year, 95% CI)", fontsize=9.5, color="#52514e")
ax.set_xlim(0, max(res["test"][m]["recall_ci95"][1] for m in models) + 0.07)
ax.set_ylim(-0.6, len(order) - 0.1)
ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors="#52514e")
ax.grid(axis="x", color="#0b0b0b", alpha=0.08); ax.set_axisbelow(True)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR.values()] + [plt.Rectangle((0, 0), 1, 1, facecolor="#2a78d6", alpha=0.45, hatch="///", edgecolor="white", linewidth=0)]
ax.legend(handles, list(COLOR) + ["† leak-prone inputs (upper bound)"], loc="lower right", frameon=False, fontsize=9)
ax.set_title("Which new serials will be hits? Models vs. editorial rules", loc="left", fontsize=12, fontweight="bold")
(OUT / "figs").mkdir(exist_ok=True)
plt.savefig(OUT / "figs" / "model_comparison.png", dpi=160, bbox_inches="tight", facecolor="#fcfcfb")

print((OUT / "results.md").read_text())
print("main:", main, "| best rule:", rule)
print(json.dumps({k: res[k] for k in ["increments_pp", "by_group", "leak_checks", "impact"]}, indent=1))
