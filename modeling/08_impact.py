"""Business impact (for the deck and video): with a fixed number of recommendation slots per quarter,
how many future hits the model and the rule each pick, and what that is worth.

A "slot" here is a recommendation position for new serials each quarter (like the Rising Stars list,
which shows at most 50 serials per genre on the site); each quarter takes the SLOTS highest scores.
This differs from Recall@slots in 03_compare.py, where slots = that quarter's number of hits (about 200);
here the number of slots is a fixed business parameter. The fewer the slots, the larger the model's lead.

All assumptions are in the block below. After changing them, rerun this file and then
reports/slides/build_deck.js, and the deck's numbers update.
Money logic: a slot lifts the views of the serial on it by LIFT (assumed), and Royal Road earns CPM dollars
per 1,000 views (assumed). The model's and the rule's picks collect different views; the difference
x LIFT x CPM is the extra ad revenue from the model. Views are cumulative as of 2025, so this is the value
of "giving the same slots to the right serials"; it leaves out readers staying on to read other serials
and is conservative.
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from features import EMB, OUT, RECENT_YEAR, TEST_YEAR

# ==== Assumptions (edit here) ====
SLOTS = 50                                   # recommendation slots per quarter (base case)
SLOTS_GRID = [25, 50, 100, 200, 400]         # sensitivity over the number of slots
LIFT = {"low": 0.05, "base": 0.10, "high": 0.20}   # how much a slot lifts that serial's views
CPM = {"low": 2.0, "base": 3.0, "high": 5.0}       # ad revenue per 1,000 views (USD)
EDITOR_MIN_PER_BOOK = 10                     # minutes for an editor to screen one new serial
EDITOR_SHORTLIST = 200                       # with the model, editors only read its top N each quarter
EDITOR_HOURLY = 30.0                         # editor hourly cost (USD)
B = 500                                      # bootstrap replicates
# ======================

MODELS = ["rule_fast_update", "rule_hot_genre", "rule_known_author", "lr_S", "lr_SA", "lr_SAT", "lr_T", "gb_SAT", "mlp_T", "mlp_SAT"]
MAIN, RULE = "lr_SAT", "rule_hot_genre"
rng = np.random.default_rng(0)
r3 = lambda x: round(float(x), 3)

meta = pq.read_table(EMB / "books.parquet", columns=["fiction_id", "title", "author", "total_views", "pct_in_quarter"]).to_pandas()
p = (pd.read_parquet(OUT / "preds_ml.parquet").merge(pd.read_parquet(OUT / "preds_dl.parquet"), on="fiction_id")
     .merge(pd.read_parquet(OUT / "preds_day30.parquet")[["fiction_id", "day30"]], on="fiction_id").merge(meta, on="fiction_id"))
test, recent = p[p.year == TEST_YEAR].reset_index(drop=True), p[p.year == RECENT_YEAR].reset_index(drop=True)
nq = test.quarter.nunique()


def picks(d, m, k):
    """The k highest-scoring serials in each quarter (for a subgroup, k is scaled by its share)."""
    return pd.concat([g.nlargest(k, m) for _, g in d.groupby("quarter")])


def prec(d, m, k):
    return picks(d, m, k).top10.mean()


def boot(d, fns, k):
    """Resample books within each quarter (keeping the per-quarter slot structure); return B values per function."""
    by_q = [np.flatnonzero(d.quarter.values == q) for q in d.quarter.unique()]
    out = {name: np.empty(B) for name in fns}
    for i in range(B):
        s = d.iloc[np.concatenate([rng.choice(ix, len(ix)) for ix in by_q])]
        for name, f in fns.items():
            out[name][i] = f(s, k)
    return out


ci = lambda v: [r3(np.percentile(v, 2.5)), r3(np.percentile(v, 97.5))]
res = {"assumptions": {"slots_per_quarter": SLOTS, "lift": LIFT, "cpm_usd": CPM, "editor_minutes_per_book": EDITOR_MIN_PER_BOOK,
                       "editor_shortlist": EDITOR_SHORTLIST, "editor_hourly_usd": EDITOR_HOURLY, "test_year": TEST_YEAR}}

# ---- 1. Hit rate (precision) of SLOTS slots per quarter, all models ----
bt = boot(test, {m: (lambda s, k, m=m: prec(s, m, k)) for m in MODELS}, SLOTS)
res["precision_at_slots"] = {m: {"value": r3(prec(test, m, SLOTS)), "ci95": ci(bt[m]), "recent_2024": r3(prec(recent, m, SLOTS))} for m in MODELS}
res["precision_at_slots"]["random"] = {"value": r3(test.top10.mean())}
pairs = {"model_vs_rule": (MAIN, RULE), "text": ("lr_SAT", "lr_SA"), "author": ("lr_SA", "lr_S"), "boosting_vs_lr": ("gb_SAT", "lr_SAT"), "mlp_vs_lr": ("mlp_SAT", "lr_SAT")}
res["diff_pp"] = {n: {"diff": round(100 * (res["precision_at_slots"][a]["value"] - res["precision_at_slots"][c]["value"]), 1),
                      "ci95": [round(100 * x, 1) for x in np.percentile(bt[a] - bt[c], [2.5, 97.5])]} for n, (a, c) in pairs.items()}
res["hits_per_quarter"] = {m: round(res["precision_at_slots"][m]["value"] * SLOTS, 1) for m in [RULE, MAIN, "random"]}

# ---- 2. First-time / returning authors: slots split by each group's share ----
res["by_group"] = {}
for gname, gm in {"first_time_author": test.first_time_author.values, "returning_author": ~test.first_time_author.values}.items():
    d = test[gm]
    k = max(1, round(SLOTS * gm.mean()))
    res["by_group"][gname] = {"slots_per_quarter": k, **{m: r3(prec(d, m, k)) for m in [RULE, MAIN]}}

# ---- 3. Day-30 version (same SLOTS slots) ----
bd = boot(test, {"day30": lambda s, k: prec(s, "day30", k), MAIN: lambda s, k: prec(s, MAIN, k)}, SLOTS)
ft = test[test.first_time_author]
kft = max(1, round(SLOTS * test.first_time_author.mean()))
res["day30"] = {"test": {MAIN: r3(prec(test, MAIN, SLOTS)), "day30": r3(prec(test, "day30", SLOTS)),
                         "diff_pp": round(100 * (prec(test, "day30", SLOTS) - prec(test, MAIN, SLOTS)), 1),
                         "diff_ci95": [round(100 * x, 1) for x in np.percentile(bd["day30"] - bd[MAIN], [2.5, 97.5])]},
                "test_first_time": {MAIN: r3(prec(ft, MAIN, kft)), "day30": r3(prec(ft, "day30", kft))},
                "recent_2024": {MAIN: r3(prec(recent, MAIN, SLOTS)), "day30": r3(prec(recent, "day30", SLOTS))}}

# ---- 4. Money: views of the serials in the slots -> extra ad revenue ----
def views_per_q(m, k):
    return picks(test, m, k).total_views.sum() / nq


money = {}
for k in SLOTS_GRID:
    vm, vr = views_per_q(MAIN, k), views_per_q(RULE, k)
    money[k] = {"views_model_per_q": round(vm), "views_rule_per_q": round(vr), "views_random_per_q": round(test.total_views.mean() * k),
                "ratio_model_to_rule": round(vm / vr, 2),
                "extra_revenue_per_year": {f"{li}_{ci_}": round((vm - vr) * LIFT[li] / 1000 * CPM[ci_] * 4)
                                           for li in LIFT for ci_ in CPM}}
base = money[SLOTS]
res["money"] = {"by_slots": money, "base": {"slots": SLOTS, "ratio_model_to_rule": base["ratio_model_to_rule"],
                                            "extra_views_per_year": round((base["views_model_per_q"] - base["views_rule_per_q"]) * LIFT["base"] * 4),
                                            "extra_revenue_per_year": base["extra_revenue_per_year"]["base_base"],
                                            "low": base["extra_revenue_per_year"]["low_low"], "high": base["extra_revenue_per_year"]["high_high"]},
                "views_share_top10": r3(test.total_views[test.top10].sum() / test.total_views.sum())}

# ---- 5. Editor time (supplementary) ----
bpq = len(test) / nq
h_before, h_after = bpq * EDITOR_MIN_PER_BOOK / 60, EDITOR_SHORTLIST * EDITOR_MIN_PER_BOOK / 60
res["editor"] = {"books_per_quarter": round(bpq), "hours_per_quarter_before": round(h_before), "hours_per_quarter_after": round(h_after),
                 "hours_saved_per_year": round((h_before - h_after) * 4), "usd_saved_per_year": round((h_before - h_after) * 4 * EDITOR_HOURLY)}

# ---- 6. Example for the video opening: a 2023 first-time author, later in the top 10%, ranked in the quarter's top SLOTS at chapter 3 ----
test["rank_in_quarter"] = test.groupby("quarter")[MAIN].rank(ascending=False, method="first").astype(int)
test["books_in_quarter"] = test.groupby("quarter").fiction_id.transform("size")
ex = test[test.first_time_author & test.top10 & (test.rank_in_quarter <= SLOTS)].sort_values("followers", ascending=False)
res["example_candidates"] = ex.head(5)[["fiction_id", "title", "author", "quarter", "rank_in_quarter", "books_in_quarter", "followers"]].to_dict("records")

# ---- 7. Post-hoc check: with 2023 and 2024 pooled (8 quarters), do boosting / MLP reliably beat LR? 2024 labels are immature; supporting evidence only.
both = pd.concat([test, recent], ignore_index=True)
bt2 = boot(both, {m: (lambda s, k, m=m: prec(s, m, k)) for m in ["lr_SAT", "gb_SAT", "mlp_SAT"]}, SLOTS)
res["diff_pp_2023_2024"] = {n: {"diff": round(100 * (prec(both, a, SLOTS) - prec(both, c, SLOTS)), 1),
                                "ci95": [round(100 * x, 1) for x in np.percentile(bt2[a] - bt2[c], [2.5, 97.5])]}
                            for n, (a, c) in {"boosting_vs_lr": ("gb_SAT", "lr_SAT"), "mlp_vs_lr": ("mlp_SAT", "lr_SAT")}.items()}

(OUT / "impact.json").write_text(json.dumps(res, indent=1, default=int))
print(json.dumps({k: res[k] for k in ["precision_at_slots", "diff_pp", "diff_pp_2023_2024", "hits_per_quarter", "by_group", "day30", "editor"]}, indent=1))
print(json.dumps(res["money"]["base"], indent=1), "\n", {k: v["ratio_model_to_rule"] for k, v in money.items()})
print(pd.DataFrame(res["example_candidates"]).to_string())
