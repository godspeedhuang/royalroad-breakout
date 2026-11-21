"""Export the data behind the demo page (docs/demo/index.html).

Reads every model's score for every 2023 and 2024 book (out/preds_ml.parquet, out/preds_dl.parquet; run 01_ml.py and
02_dl.py first), titles, tags and cover URLs from embedding/out/books.parquet, and the cover status from
data/cover/cover_status.parquet when it is present. No author names are exported.

Also prints the headline numbers the page computes, so they can be checked against out/results.json and the deck
(out/impact.json from 08_impact.py). The page follows the deck: each quarter's top k serials are picked the way
08_impact.py does (pandas nlargest, so ties keep the row order of out/preds_ml.parquet, exported as "ord"), first-time
authors get slots in proportion to their share, and editor time is "screen the model's top 200 instead of every new
serial". With out/preds_day30.parquet (from 07_day30.py) the day-30 second pass is added as a model.

Output: docs/demo/data.js (sets window.DEMO_DATA; a script file rather than JSON so the page also works when opened
straight from disk, where browsers block fetch() of local files)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from features import OUT, recall_at_slots

HERE = Path(__file__).parent
ROOT = HERE.parent
DEMO = ROOT / "docs" / "demo" / "data.js"
COVER_STATUS = ROOT / "data" / "cover" / "cover_status.parquet"
MODELS = {   # order = order on the page
    "rule_hot_genre": "Editors' rule: popular genres first",
    "lr_S": "LR: tags and structure (S)",
    "lr_SA": "LR: + author (S+A)",
    "lr_SAT": "LR: + first three chapters (S+A+T)",
    "gb_SAT": "Gradient boosting (S+A+T)",
    "mlp_SAT": "MLP (S+A+T)",
    "lr_T": "LR: chapter text only (T)",
}
SHORT = {"rule_hot_genre": "Editors' rule", "lr_S": "S", "lr_SA": "S+A", "lr_SAT": "S+A+T", "gb_SAT": "Boosting S+A+T",
         "mlp_SAT": "MLP S+A+T", "lr_T": "Text only"}
STATUS = {"valid": 0, "after_ch3": 1, "no_timestamp": 2, "no_custom_cover": 3, "image_changed": 4}
COVER_STATS = ROOT / "data" / "cover" / "cover_stats.parquet"   # optional: average cover color per book (mean_r/g/b in 0–1)
# Square tint on the page when no cover color is available: the first matching group wins. Muted on purpose: the tint is
# texture, not a reading channel (picked / hit status is shown by brightness and the orange accent).
GENRE_TINT = [
    (("LitRPG", "GameLit", "Progression", "Dungeon"), "#2f8f88", "Game and progression"),
    (("Romance",), "#b45c79", "Romance"),
    (("Sci-fi", "Soft Sci-fi", "Hard Sci-fi", "Space Opera", "Cyberpunk", "Artificial Intelligence"), "#3f73b8", "Sci-fi"),
    (("Horror", "Grimdark", "Psychological", "Tragedy"), "#9a4a3c", "Dark and horror"),
    (("Slice of Life", "Comedy"), "#b89a5a", "Slice of life and comedy"),
    (("Mystery", "Supernatural", "Urban Fantasy", "Mythos"), "#7262b8", "Mystery and urban fantasy"),
]
DEFAULT_TINT = ("#6b8196", "Fantasy and other")


def tint(tag_list, fid):
    """Genre tint with a small, fixed per-book brightness jitter so the mosaic reads as texture."""
    hexcol = next((c for keys, c, _ in GENRE_TINT if any(t in keys for t in tag_list)), DEFAULT_TINT[0])
    f = 0.82 + 0.36 * ((int(fid) * 2654435761) % 1000) / 1000
    r, g, b = (min(255, int(int(hexcol[i:i + 2], 16) * f)) for i in (1, 3, 5))
    return f"#{r:02x}{g:02x}{b:02x}"

p = pd.read_parquet(OUT / "preds_ml.parquet").merge(pd.read_parquet(OUT / "preds_dl.parquet"), on="fiction_id")
p["ord"] = np.arange(len(p))   # row order of preds_ml.parquet: 08_impact.py breaks score ties in this order, and so does the page
PREDS_DAY30 = OUT / "preds_day30.parquet"   # optional: the day-30 second pass from 07_day30.py (not in git)
if PREDS_DAY30.exists():
    p = p.merge(pd.read_parquet(PREDS_DAY30)[["fiction_id", "day30"]], on="fiction_id", how="left")
PREDS_COVER = ROOT / "cover" / "out" / "preds_cover.parquet"   # optional: main model + cover adjustment, from cover/03_cover_merge.ipynb
if PREDS_COVER.exists():
    p = p.merge(pd.read_parquet(PREDS_COVER), on="fiction_id", how="left")
    MODELS["lr_SATC"] = "LR + cover adjustment (S+A+T+cover, tried)"
    SHORT["lr_SATC"] = "S+A+T+Cover"
if "day30" in p:
    MODELS["day30"] = "Day-30 second pass (S+A+T + first month's posting rhythm)"
    SHORT["day30"] = "S+A+T + day 30"
    p["day30"] = p.day30.fillna(p.day30.min() - 1)   # books without a day-30 score go last
meta = pq.read_table(ROOT / "embedding" / "out" / "books.parquet", columns=["fiction_id", "title", "tags", "cover_url", "total_views"]).to_pandas()
p = p.merge(meta, on="fiction_id", how="left").sort_values(["quarter", "fiction_id"]).reset_index(drop=True)
assert len(p) == 8075 + 12165, len(p)

if COVER_STATUS.exists():
    st = pd.read_parquet(COVER_STATUS, columns=["fiction_id", "status"]).set_index("fiction_id").status
    cover_status = p.fiction_id.map(st).map(STATUS).fillna(-1).astype(int)
else:   # the page then shows no cover-status note and no cover leak check
    cover_status = pd.Series(-1, index=p.index)

def cover_urls(s):
    """Full cover URL, or "" for serials without a custom cover."""
    s = s.fillna("")
    return np.where(s.str.contains("nocover") | (s == ""), "", np.where(s.str.startswith("/"), "https://www.royalroad.com" + s, s))
url = cover_urls(p.cover_url)
tags = p.tags.map(lambda s: json.loads(s) if isinstance(s, str) and s else [])
vocab = sorted({t for l in tags for t in l})
pos = {t: i for i, t in enumerate(vocab)}
quarters = sorted(p.quarter.unique())

if COVER_STATS.exists():   # real average cover color where we have one, genre tint otherwise
    cst = pd.read_parquet(COVER_STATS, columns=["fiction_id", "mean_r", "mean_g", "mean_b"]).set_index("fiction_id")
    rgb = p.fiction_id.map(lambda f: cst.loc[f] if f in cst.index else None)
    color = [("#%02x%02x%02x" % tuple(int(255 * v) for v in c)) if c is not None else tint(t, f) for c, t, f in zip(rgb, tags, p.fiction_id)]
    COLOR_SOURCE = "average cover color (genre tint where no cover)"
else:
    color = [tint(t, f) for t, f in zip(tags, p.fiction_id)]
    COLOR_SOURCE = "genre tint"

# Everything below reads embedding/out/books.parquet once (45,245 books, with the 1,536-d first-three-chapter vectors)
allb = pq.read_table(ROOT / "embedding" / "out" / "books.parquet",
                     columns=["fiction_id", "title", "year", "quarter", "followers", "total_views", "top10", "tags", "warnings", "cover_url", "first3_tokens",
                              "pca_x", "pca_y", "umap_x", "umap_y", "book_vec", "chap1_vec"]).to_pandas()
allb["tag_list"] = allb.tags.map(lambda s: json.loads(s) if isinstance(s, str) and s else [])
wm = set(pd.read_csv(ROOT / "embedding" / "out" / "watermark_books.csv").fiction_id)
allb["wm"] = allb.fiction_id.isin(wm)
allb["cover"] = cover_urls(allb.cover_url)

# "What counts as a hit" animation: the four 2016 quarters, 20 serials each (2 hits + 18 others, the real 10% share).
# Every hit has more followers than every non-hit of its quarter, so the top 2 of each sample are exactly its hits.
anim = {"quarters": [], "thresholds": {}}
for q, d in allb[allb.year == 2016].groupby("quarter"):
    s = pd.concat([d[d.top10].sample(2, random_state=0), d[~d.top10].sample(18, random_state=0)]).sort_values("fiction_id")
    anim["quarters"].append({"q": q, "n": len(d), "items": [{"t": r.title, "f": int(r.followers), "y": int(r.top10), "c": tint(r.tag_list, r.fiction_id), "u": r.cover}
                                                         for r in s.itertuples()]})
for q, d in allb[allb.top10].groupby("quarter"):
    anim["thresholds"][q] = int(d.followers.min())   # followers a serial needed to be a hit in its quarter

# What is inside each feature block (as built in features.py)
warn_list = allb.warnings.map(lambda s: json.loads(s) if isinstance(s, str) and s else [])
tag_counts = pd.Series([t for l in allb.tag_list for t in l]).value_counts()
features = {
    "S": {"tags": [[t, int(n), next((c for keys, c, _ in GENRE_TINT if t in keys), DEFAULT_TINT[0])] for t, n in tag_counts.items()],
          "warnings": sorted({w for l in warn_list for w in l})},
    "first_time_share_2023": round(float(p[p.year == 2023].first_time_author.mean()), 3),
}

# Hyperparameters each model searched on 2022, and the setting chosen (out/selection_*.json)
sel_ml = json.loads((OUT / "selection_ml.json").read_text())
sel_dl = json.loads((OUT / "selection_dl.json").read_text())
hyper = {
    "lr": {m: {"C": sel_ml[m]["C"], "val_auc_by_C": sel_ml[m]["val_auc_by_C"]} for m in ["lr_S", "lr_SA", "lr_SAT", "lr_T"]},
    "gb": {"chosen": [sel_ml["gb_SAT"]["learning_rate"], sel_ml["gb_SAT"]["max_leaf_nodes"]], "val_auc_by_param": sel_ml["gb_SAT"]["val_auc_by_param"]},
    "mlp": {"chosen": [sel_dl["mlp_SAT"]["dropout"], sel_dl["mlp_SAT"]["weight_decay"], sel_dl["mlp_SAT"]["epochs"]],
            "val_auc_by_param": sel_dl["mlp_SAT"]["val_auc_by_param"]},
}

# The text vector: two example 2023Q1 serials and the "hit direction" (average hit minus average other serial, 2017–2022,
# watermarked books left out), each 1,536 numbers
V = np.stack(allb.book_vec.values).astype(np.float32)
fit = allb.year.between(2017, 2022).values & ~allb.wm.values
direction = V[fit & allb.top10.values].mean(0) - V[fit & ~allb.top10.values].mean(0)
direction /= np.linalg.norm(direction)
q1 = p[p.quarter == "2023Q1"].sort_values(["lr_SAT", "fiction_id"], ascending=[False, True])
top_ex = q1[q1.top10].iloc[0]                               # highest-scored 2023Q1 serial that became a hit
low_ex = q1[~q1.top10].iloc[int(0.95 * (~q1.top10).sum())]  # a low-scored non-hit (95th percentile from the top)
pos_of = {f: i for i, f in enumerate(allb.fiction_id)}
def example(r, label):
    i = pos_of[r.fiction_id]; v = V[i]
    return {"label": label, "title": r.title, "rank": int((q1.lr_SAT > r.lr_SAT).sum()) + 1, "n": len(q1), "hit": int(r.top10),
            "cos": round(float(v @ direction / np.linalg.norm(v)), 3), "v": np.round(v, 4).tolist(),
            "u": allb.cover.iloc[i], "tokens": int(allb.first3_tokens.iloc[i]),
            "p": [round(float(allb.pca_x.iloc[i]), 4), round(float(allb.pca_y.iloc[i]), 4)]}
emb = {"examples": [example(top_ex, "A top pick"), example(low_ex, "A low pick")], "direction": np.round(direction, 4).tolist()}
# The first lines of the top pick's chapter 1 for the tokenizer animation (the public Atlas shows up to 300 characters)
ATLAS = ROOT / "embedding" / "out" / "atlas.parquet"
if ATLAS.exists():
    ex_txt = pq.read_table(ATLAS, columns=["fiction_id", "chapter1_excerpt"]).to_pandas().set_index("fiction_id").chapter1_excerpt.get(int(top_ex.fiction_id), "")
    emb["excerpt"] = ex_txt[:200].rsplit(" ", 1)[0] + " …" if len(ex_txt) > 200 else ex_txt
    import tiktoken   # the tokenizer of text-embedding-3-small, as in embedding/01_prepare.py
    enc = tiktoken.get_encoding("cl100k_base")
    emb["tokens"] = [enc.decode_single_token_bytes(t).decode("utf-8", "replace") for t in enc.encode(emb["excerpt"].rstrip(" …"))]
# Chapter 1's own vector for the top pick (chapters 2 and 3 are not kept separately), and how close it is to the book vector
c1 = np.asarray(allb.chap1_vec.iloc[pos_of[top_ex.fiction_id]], dtype=np.float32)
emb["examples"][0]["c1"] = np.round(c1, 4).tolist()
emb["examples"][0]["cos_c1_book"] = round(float(c1 @ V[pos_of[top_ex.fiction_id]] / np.linalg.norm(c1) / np.linalg.norm(V[pos_of[top_ex.fiction_id]])), 3)

# Screen 0 (why we built this): new serials per quarter, hits, first-time authors, and a wall of 2023 Q1 covers
y23 = p[p.year == 2023]
why = {
    "per_quarter_2023": round(len(y23) / 4), "hits_per_quarter_2023": round(int(y23.top10.sum()) / 4),
    "first_time_share_2023": round(float(y23.first_time_author.mean()), 3),
    "hit_rate_first_2023": round(float(y23[y23.first_time_author].top10.mean()), 3),
    "hit_rate_returning_2023": round(float(y23[~y23.first_time_author].top10.mean()), 3),
    "sample_books": len(allb), "sample_years": [int(allb.year.min()), int(allb.year.max())],
    # where the reading goes: the hits' share of all views and followers of 2023 serials (snapshot of June 2025)
    "views_share_hits_2023": round(float(allb.total_views[(allb.year == 2023) & allb.top10].sum() / allb.total_views[allb.year == 2023].sum()), 3),
    "followers_share_hits_2023": round(float(allb.followers[(allb.year == 2023) & allb.top10].sum() / allb.followers[allb.year == 2023].sum()), 3),
}
wall = p[(p.quarter == "2023Q1") & (url != "")].sample(96, random_state=1)
why["wall"] = [{"u": u, "y": int(y), "f": int(f), "t": t} for u, y, f, t in zip(cover_urls(wall.cover_url), wall.top10, wall.first_time_author, wall.title)]

# Story map for "taste over time": 4,000 sampled serials (2017–2024, no watermarks) in the PCA and UMAP planes, and the
# yearly average position of all serials and of hits (PCA is linear, so the average of 2D points is the projected centroid)
ok = allb[allb.year.between(2017, 2024) & ~allb.wm]
smp = ok.sample(4000, random_state=0)
ev = json.loads((ROOT / "embedding" / "out" / "projection_summary.json").read_text())["pca_explained"]
traj = lambda d: {str(y): [round(float(g.pca_x.mean()), 4), round(float(g.pca_y.mean()), 4)] for y, g in d.groupby("year")}
story_map = {
    "pca": np.round(smp[["pca_x", "pca_y"]].values, 4).tolist(), "umap": np.round(smp[["umap_x", "umap_y"]].values, 3).tolist(),
    "year": smp.year.astype(int).tolist(), "y": smp.top10.astype(int).tolist(),
    "color": [tint(t, f) for t, f in zip(smp.tag_list, smp.fiction_id)],
    "traj_all": traj(ok), "traj_hits": traj(ok[ok.top10]), "pca_explained_2d": round(ev[0] + ev[1], 3),
    "pca_explained": [round(ev[0], 3), round(ev[1], 3)],
}
# How the average serial and the average hit moved from 2017 to 2024 in the full 1,536 numbers (no watermarks), and whether
# the two moves point the same way (cosine); the 2D paths above are only a picture of this
mv = lambda d: V[d.index[d.year == 2024]].mean(0) - V[d.index[d.year == 2017]].mean(0)
m_all, m_hit = mv(ok), mv(ok[ok.top10])
# Two of the 1,536 raw numbers for the map's intro (the ones most correlated with PC1 and PC2), standardized
Vs = V[smp.index]
corr = lambda a: np.abs(np.corrcoef(np.c_[Vs, a].T)[-1, :-1])
d1 = int(np.argmax(corr(smp.pca_x.values))); c2 = corr(smp.pca_y.values); c2[d1] = 0; d2 = int(np.argmax(c2))
z = lambda x: (x - x.mean()) / x.std()
story_map["raw_dims"] = [d1, d2]
story_map["raw"] = np.round(np.c_[z(Vs[:, d1]), z(Vs[:, d2])].astype(float), 3).tolist()
story_map["shift_cos_1536"] = round(float(m_all @ m_hit / np.linalg.norm(m_all) / np.linalg.norm(m_hit)), 3)

# Small datasets for the "what else we checked" charts (read from the repo's saved results)
drift = json.loads((OUT / "drift_auc.json").read_text())
tc = pd.read_csv(ROOT / "embedding" / "out" / "tags_compare.csv")
tc = tc[tc.supply_e >= 3].sort_values("d_log2")   # as in 10_tags_compare.py: tags under 3% early supply (e.g. Progression) mostly reflect tags spreading later
cm = json.loads((ROOT / "cover" / "out" / "cover_merge.json").read_text())
cm_tab = {(r["test_set"], r["model"]): r["recall_at_slots"] for r in cm["plan2_table"]}
extras = {
    "drift_2023": {f: {"gap": [int(g) for g in drift[f]["2023"]], "auc": [v["auc"] for v in drift[f]["2023"].values()],
                       "per_year": drift[f]["auc_change_per_year_of_gap"]} for f in ["S", "T"]},
    "tag_movers": [{"tag": r.tag, "early": round(r.lift_e, 2), "late": round(r.lift_l, 2)}
                   for r in pd.concat([tc.tail(4).iloc[::-1], tc.head(4)]).itertuples()],
    "cover_experiment": [
        {"label": "Valid-cover serials, 2023", "without": cm_tab[("2023 valid cover", "Main model S+A+T (all serials)")],
         "with": cm_tab[("2023 valid cover", "Two-stage: main + cover adjustment")]},
        {"label": "Valid-cover serials, 2024", "without": cm_tab[("2024 valid cover", "Main model S+A+T (all serials)")],
         "with": cm_tab[("2024 valid cover", "Two-stage: main + cover adjustment")]},
        {"label": "All serials, 2023", "without": cm_tab[("2023 all", "Main model S+A+T (all serials)")],
         "with": cm_tab[("2023 all", "Two-stage: main + cover adjustment")]},
    ],
    "genre_tints": [{"label": lab, "color": c} for _, c, lab in GENRE_TINT] + [{"label": DEFAULT_TINT[1], "color": DEFAULT_TINT[0]}],
    "color_source": COLOR_SOURCE,
}

IMPACT = OUT / "impact.json"   # the deck's numbers (08_impact.py); the page quotes its confidence intervals and day-30 result
deck = json.loads(IMPACT.read_text()) if IMPACT.exists() else None
if deck:
    deck = {k: deck[k] for k in ["assumptions", "precision_at_slots", "diff_pp", "diff_pp_2023_2024", "by_group", "day30", "editor"] if k in deck} | \
           {"money_base": deck["money"]["base"]}
sby = pd.read_csv(ROOT / "cover" / "out" / "status" / "status_by_year.csv")   # cover status of every serial, 2016–2024
extras["valid_cover_share_all"] = round(float(sby[sby.status == "valid"].n.sum() / sby.n.sum()), 3)   # the deck's "only 42% had one by chapter 3"

data = {
    "note": "Generated by modeling/05_demo_data.py. Scores are each model's decision_function (higher = more likely a hit); "
            "y = 1 if the book ended in the top 10% of followers among books first released in the same quarter.",
    "models": MODELS,
    "short": SHORT,
    "quarters": quarters,
    "anim": anim,
    "features": features,
    "hyper": hyper,
    "emb": emb,
    "why": why,
    "story_map": story_map,
    "extras": extras,
    "deck": deck,
    "tags": vocab,
    "cover_status": {v: k for k, v in STATUS.items()},
    "books": {
        "id": p.fiction_id.astype(int).tolist(),
        "q": p.quarter.map({q: i for i, q in enumerate(quarters)}).tolist(),
        "y": p.top10.astype(int).tolist(),
        "first": p.first_time_author.astype(int).tolist(),
        "title": p.title.fillna("").tolist(),
        "tags": [[pos[t] for t in l] for l in tags],
        "cover": list(url),
        "cover_status": cover_status.tolist(),
        "blurb_flag": p.desc_marker.astype(int).tolist(),
        "views": p.total_views.fillna(0).astype(int).tolist(),   # total views in the June 2025 snapshot (business screen)
        "color": color,
        "ord": p.ord.astype(int).tolist(),   # tie-break order (see above)
        # enough decimals that rounding creates no false ties (the rule's scores differ only in the 5th–8th decimal)
        "score": {m: np.round(p[m].values.astype(float), 8 if m.startswith("rule") else 6).tolist() for m in MODELS},
    },
}


def page_picks(d, m, k):
    """Each quarter's top k books by model m, the way the page picks them: its rounded scores, ties by ord."""
    dec = 8 if m.startswith("rule") else 6
    return d.assign(_s=np.round(d[m].values.astype(float), dec)).sort_values(["_s", "ord"], ascending=[False, True]).groupby("quarter").head(k)


def deck_picks(d, m, k):
    """The same, the way 08_impact.py does it: nlargest on the raw scores, rows in preds_ml order."""
    d = d.sort_values("ord")
    return pd.concat([g.nlargest(k, m) for _, g in d.groupby("quarter")])


def share_of_slots(d, m, k):
    picks = page_picks(d, m, k)
    return picks.top10.mean(), int(picks.top10.sum()), len(picks)


t23 = p[p.year == 2023]
K_STEPS = [k for k in range(10, 100, 5)] + list(range(100, 500, 10)) + list(range(500, 1000, 50)) + list(range(1000, 4001, 100))   # the page's slider
for yr in (2023, 2024):   # the page must pick exactly the deck's serials at every slider position
    d = p[p.year == yr]
    for m in MODELS:
        for k in K_STEPS:
            assert set(page_picks(d, m, k).fiction_id) == set(deck_picks(d, m, k).fiction_id), (yr, m, k)
top_views = lambda d, m, k: int(page_picks(d, m, k).total_views.sum())
ad = {k: round((top_views(t23, "lr_SAT", k) - top_views(t23, "rule_hot_genre", k)) * 0.10 * 3 / 1000) for k in [25, 50, 100]}
print("Ad revenue vs rule, 2023 (10% lift, $3 per 1,000 views):", ad)
if deck:   # the page's numbers against the deck's (out/impact.json)
    ft = t23[t23.first_time_author]; kft = max(1, round(50 * t23.first_time_author.mean()))
    page = {"rule_hot_genre": share_of_slots(t23, "rule_hot_genre", 50)[0], "lr_SAT": share_of_slots(t23, "lr_SAT", 50)[0],
            "gb_SAT": share_of_slots(t23, "gb_SAT", 50)[0], "mlp_SAT": share_of_slots(t23, "mlp_SAT", 50)[0],
            "first_time lr_SAT": share_of_slots(ft, "lr_SAT", kft)[0], "first_time rule": share_of_slots(ft, "rule_hot_genre", kft)[0],
            "ad revenue at 50": ad[50]}
    want = {m: deck["precision_at_slots"][m]["value"] for m in ["rule_hot_genre", "lr_SAT", "gb_SAT", "mlp_SAT"]} | \
           {"first_time lr_SAT": deck["by_group"]["first_time_author"]["lr_SAT"], "first_time rule": deck["by_group"]["first_time_author"]["rule_hot_genre"],
            "ad revenue at 50": deck["money_base"]["extra_revenue_per_year"]}
    for key, w in want.items():
        g = round(page[key], 3) if key != "ad revenue at 50" else page[key]
        print(f"{'ok  ' if g == w else 'DIFF'} {key}: page {g}, deck {w}")
    if "day30" in MODELS:
        print("day-30 share of 50 slots, 2023: page", round(share_of_slots(t23, "day30", 50)[0], 3), "deck", deck["day30"]["test"]["day30"])
checks = {"recall_at_slots_2023": {m: round(float(recall_at_slots(t23.top10.values, t23[m].values, t23.quarter.values)), 3) for m in MODELS}}
for k in [50]:
    checks[f"share_of_{k}_slots_2023"] = {m: dict(zip(["share", "hits", "slots"], share_of_slots(t23, m, k))) for m in MODELS}
data["checks"] = checks

if "lr_SATC" in MODELS:   # must match cover/out/cover_merge.json (two-stage, 2023 all serials)
    want = round(cm_tab[("2023 all", "Two-stage: main + cover adjustment")], 3)
    assert checks["recall_at_slots_2023"]["lr_SATC"] == want, (checks["recall_at_slots_2023"]["lr_SATC"], want)
ref = json.loads((OUT / "results.json").read_text())["test"]
checks["differs_from_results_json"] = {}
for m in [m for m in MODELS if m in ref]:   # the page must reproduce the repo's headline numbers
    got, want = checks["recall_at_slots_2023"][m], round(ref[m]["recall_at_slots"], 3)
    if m == "gb_SAT" and got != want:
        # Gradient boosting is not exactly reproducible across machines: tiny floating-point differences in the PCA step
        # change which of four near-tied settings wins on 2022. Reported, not fatal; the other models must match exactly.
        checks["differs_from_results_json"][m] = {"this_export": got, "results_json": want}
        print(f"note: {m} recall@slots {got} here vs {want} in out/results.json (not exactly reproducible across machines)")
        continue
    assert got == want, (m, got, want)

DEMO.parent.mkdir(parents=True, exist_ok=True)
DEMO.write_text("window.DEMO_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
print(f"wrote {DEMO.relative_to(ROOT)} ({DEMO.stat().st_size / 1e6:.1f} MB), {len(p):,} books, cover status {'yes' if COVER_STATUS.exists() else 'no'}")
print("Recall@slots 2023 (slots = true hits per quarter):", checks["recall_at_slots_2023"])
print("Share of 50 slots per quarter, 2023:", {m: f"{v['share']:.3f} ({v['hits']}/{v['slots']})" for m, v in checks["share_of_50_slots_2023"].items()})
