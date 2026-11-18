"""Put the movement of the vectors into words.

Compares early (2016–2018) with recent (2021–2023). 2024 has been observed too briefly and is only a sensitivity check.
Three directions (all in 1,536 dimensions):
  supply: all books, recent minus early
  hot   : top 10%, recent minus early
  pref  : (top 10% minus others) recent minus (top 10% minus others) early, i.e. the change in preference net of supply

Three ways to explain them:
  1. books and tags at the two ends of each direction
  2. word comparison: share of books containing a word, compared with log-odds (per book, so one long book does not dominate)
  3. concept probes: embed sentences that describe concepts and see how close each year gets to them

Only books without watermark sentences are used (see 11_watermark_check.py).
Outputs: out/direction_summary_clean.json, out/figs/concept_change_clean.png
To redraw only the figure from the saved JSON (no data or API needed): python 12_explain_direction.py --plot-only
(out/direction_summary.json is the older result from before watermarks were removed; this script no longer writes it.)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import dotenv_values
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer

HERE = Path(__file__).parent
OUT = HERE / "out"
FIG = OUT / "figs"
plt.rcParams.update({"font.sans-serif": ["PingFang TC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"], "axes.unicode_minus": False})
EARLY, LATE = (2016, 2018), (2021, 2023)
YEARS = list(range(2016, 2025))
rng = np.random.default_rng(0)

# Concepts were fixed before looking at the results; they cover both subject matter and writing style
CONCEPTS = {
    "game system": "A story with a game-like system: status screens, levels, skills, stats and experience points.",
    "dungeons and monsters": "Adventurers explore a dungeon, fight monsters and collect loot.",
    "cultivation sect": "A young disciple of a cultivation sect refines qi and breaks through to higher realms.",
    "isekai reincarnation": "The protagonist dies and is reincarnated in another world with memories of their past life.",
    "kingdom building": "The protagonist builds up a settlement, managing resources, economy, trade and people.",
    "war and military": "Soldiers and commanders fight a war, with battles, tactics and military campaigns.",
    "space sci-fi": "A spaceship crew travels between planets using advanced technology.",
    "urban fantasy": "Magic and supernatural creatures hidden in a modern city, with phones, cars and police.",
    "magic academy": "Students attend a magic academy, with classes, teachers and rivalries.",
    "romance": "Two people fall in love, and the story focuses on their romance and feelings.",
    "cozy slice of life": "A cozy, gentle story about daily life, food, friends and small pleasures.",
    "grimdark": "A grim, brutal world full of violence, despair and moral corruption.",
    "comedy": "A funny story full of jokes, sarcasm and absurd situations.",
    "gods and myths": "Gods, divine powers, prophecy and ancient myths shape the world.",
    "first-person inner monologue": "A first-person narrator shares a constant stream of personal thoughts and sarcastic inner commentary.",
    "literary prose": "Slow, lyrical literary prose with rich description of landscapes and atmosphere.",
    "opens with a fight": "The story opens immediately in the middle of a violent fight.",
    "female lead emotions": "A female protagonist, her relationships, her emotions and the people around her.",
}


def unit(v):
    return v / np.linalg.norm(v)


def logit(y, n):
    return np.log((y + .5) / (n - y + .5))


def var(y, n):
    return 1 / (y + .5) + 1 / (n - y + .5)


def plot_concepts(tab, path):
    """Concept-probe figure from a table with supply_change, hot_change, gap_change, gap_change_lo, gap_change_hi (one row per concept)."""
    o = tab.sort_values("hot_change")
    fig, ax = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    ypos = np.arange(len(o))
    ax[0].barh(ypos - .2, o.supply_change, height=.4, color="#2f6f73", label="all books (supply)")
    ax[0].barh(ypos + .2, o.hot_change, height=.4, color="#c8553d", label="top 10% (hits)")
    ax[0].set_yticks(ypos, o.index); ax[0].axvline(0, color="gray", lw=.8); ax[0].legend(); ax[0].set_title("Recent minus early: closer to the concept (right) or further (left)", fontsize=10)
    ax[0].set_xlabel("change (unit: one standard deviation across all books)")
    err = np.vstack([o.gap_change - o.gap_change_lo, o.gap_change_hi - o.gap_change])
    ax[1].barh(ypos, o.gap_change, xerr=err, color="#8a6fb0", ecolor="gray"); ax[1].axvline(0, color="gray", lw=.8)
    ax[1].set_title("Preference change: hits minus others, recent minus early (95% CI)", fontsize=10); ax[1].set_xlabel("change in the gap")
    plt.tight_layout(); plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    t = pq.read_table(OUT / "books.parquet")
    b = t.drop(["book_vec", "chap1_vec", "desc_vec"]).to_pandas()
    V = t.column("book_vec").combine_chunks().flatten().to_numpy().reshape(len(b), -1).astype(np.float32)
    # Drop books whose first three chapters contain Royal Road's anti-piracy watermark (see 11_watermark_check.py)
    cm = pd.read_parquet(OUT / "chapters_meta.parquet", columns=["fiction_id", "text"])
    low = cm.text.str.lower()
    cm["wm"] = np.logical_or.reduce([low.str.contains(w, regex=False).values for w in ("amazon", "unlawfully", "unauthorized", "royal road", "royalroad")])
    clean = ~b.fiction_id.map(cm.groupby("fiction_id").wm.any()).fillna(False).values
    del cm, low
    b, V = b[clean].reset_index(drop=True), V[clean]
    yr, top = b.year.values, b.top10.values
    early, late = (yr >= EARLY[0]) & (yr <= EARLY[1]), (yr >= LATE[0]) & (yr <= LATE[1])
    tags = b.tags.map(lambda s: json.loads(s) if isinstance(s, str) and s else [])
    S = {"periods": {"early": EARLY, "late": LATE}, "n": {
        "early_all": int(early.sum()), "late_all": int(late.sum()), "early_top": int((early & top).sum()), "late_top": int((late & top).sum())}}

    # ---- The three directions ----
    m = lambda mask: V[mask].mean(0)
    d = {"supply": m(late) - m(early), "hot": m(late & top) - m(early & top),
         "pref": (m(late & top) - m(late & ~top)) - (m(early & top) - m(early & ~top))}
    S["direction_norm"] = {k: float(np.linalg.norm(v)) for k, v in d.items()}
    S["direction_cos"] = {f"{a}~{c}": float(unit(d[a]) @ unit(d[c])) for a, c in (("supply", "hot"), ("supply", "pref"), ("hot", "pref"))}

    # ---- 1. Books and tags at the two ends of each direction ----
    first = pd.read_parquet(OUT / "chapters_meta.parquet", columns=["fiction_id", "chapter_idx", "text"])
    text1 = first[first.chapter_idx == 1].set_index("fiction_id").text
    ends = {}
    vocab = sorted({x for l in tags for x in l})
    T = pd.DataFrame({v: tags.map(lambda l, v=v: v in l).values for v in vocab})
    T = T.loc[:, T.mean() > .01]
    for k, vec in d.items():
        score = V @ unit(vec)
        pool = np.flatnonzero((early | late) & top)
        order = pool[np.argsort(score[pool])]
        pick = lambda idx: [{"title": b.title[i], "year": int(yr[i]), "followers": int(b.followers[i]), "tags": tags[i][:6],
                             "opening": " ".join(str(text1.get(b.fiction_id[i], "")).split())[:320]} for i in idx]
        sel = (early | late)
        corr = T[sel].astype(float).corrwith(pd.Series(score[sel], index=T[sel].index)).sort_values()
        ends[k] = {"negative_end": pick(order[:12]), "positive_end": pick(order[-12:][::-1]),
                   "tags_positive": corr.tail(8).iloc[::-1].round(3).to_dict(), "tags_negative": corr.head(8).round(3).to_dict()}
    S["ends"] = ends

    # ---- 2. Word comparison (per book) ----
    keep = set(b.fiction_id[early | late])
    txt = pd.read_parquet(OUT / "chapters_meta.parquet", columns=["fiction_id", "text"])
    txt = txt[txt.fiction_id.isin(keep)].groupby("fiction_id").text.agg(" ".join)
    idx = b.set_index("fiction_id").loc[txt.index]
    cv = CountVectorizer(lowercase=True, token_pattern=r"[a-z][a-z']{2,}", stop_words="english", binary=True, min_df=100, max_df=.7)
    X = cv.fit_transform(txt.values)
    words = np.array(cv.get_feature_names_out())
    print("books", X.shape[0], "vocabulary", X.shape[1])
    y_e = (idx.year <= EARLY[1]).values; y_l = ~y_e; tp = idx.top10.values
    def cnt(mask): return np.asarray(X[mask].sum(0)).ravel().astype(float), float(mask.sum())
    g = {"et": cnt(y_e & tp), "er": cnt(y_e & ~tp), "lt": cnt(y_l & tp), "lr": cnt(y_l & ~tp), "e": cnt(y_e), "l": cnt(y_l)}
    def contrast(pos, neg):
        delta = sum(logit(*g[k]) for k in pos) - sum(logit(*g[k]) for k in neg)
        return delta, delta / np.sqrt(sum(var(*g[k]) for k in pos + neg))
    share = lambda k: g[k][0] / g[k][1]
    W = {}
    for name, pos, neg, a, c in (("supply", ["l"], ["e"], "e", "l"), ("hot", ["lt"], ["et"], "et", "lt"), ("pref", ["lt", "er"], ["lr", "et"], "et", "lt")):
        delta, z = contrast(pos, neg)
        df = pd.DataFrame({"word": words, "z": z.round(2), "log_odds": delta.round(2), "share_early": share(a).round(3), "share_late": share(c).round(3)})
        W[name] = {"up": df.sort_values("z", ascending=False).head(40).to_dict("records"), "down": df.sort_values("z").head(40).to_dict("records")}
    S["words"] = W

    # ---- 3. Concept probes ----
    env = dotenv_values(HERE / ".env")
    client = OpenAI(api_key=env["PARLEY_API_KEY"], base_url=env["PARLEY_BASE_URL"], timeout=60)
    resp = client.embeddings.create(model=env.get("EMBEDDING_MODEL") or "text-embedding-3-small", input=list(CONCEPTS.values()))
    P = np.asarray([x.embedding for x in sorted(resp.data, key=lambda r: r.index)], dtype=np.float32)
    C = (V @ P.T); C = (C - C.mean(0)) / C.std(0)         # standardize each concept; the unit is one standard deviation across all books
    names = list(CONCEPTS)
    rows = []
    for y in YEARS:
        a = yr == y
        for j, nme in enumerate(names):
            tp_, rs_ = C[a & top, j], C[a & ~top, j]
            boots = [rng.choice(tp_, len(tp_)).mean() - rng.choice(rs_, len(rs_)).mean() for _ in range(300)]
            rows.append(dict(concept=nme, year=y, supply=float(C[a, j].mean()), hot=float(tp_.mean()), gap=float(tp_.mean() - rs_.mean()),
                             gap_lo=float(np.percentile(boots, 2.5)), gap_hi=float(np.percentile(boots, 97.5))))
    R = pd.DataFrame(rows)
    per = lambda col, lo, hi: R[R.year.between(lo, hi)].groupby("concept")[col].mean()
    tab = pd.DataFrame({"supply_change": per("supply", *LATE) - per("supply", *EARLY), "hot_change": per("hot", *LATE) - per("hot", *EARLY),
                        "gap_early": per("gap", *EARLY), "gap_late": per("gap", *LATE)})
    tab["gap_change"] = tab.gap_late - tab.gap_early
    # bootstrap interval of the difference in period means
    ci = {}
    for j, nme in enumerate(names):
        def gap(mask_e, mask_l):
            return C[mask_l & top, j].mean() - C[mask_l & ~top, j].mean() - (C[mask_e & top, j].mean() - C[mask_e & ~top, j].mean())
        ie, il = np.flatnonzero(early), np.flatnonzero(late); vals = []
        for _ in range(300):
            se, sl = rng.choice(ie, len(ie)), rng.choice(il, len(il))
            vals.append(C[sl[top[sl]], j].mean() - C[sl[~top[sl]], j].mean() - (C[se[top[se]], j].mean() - C[se[~top[se]], j].mean()))
        ci[nme] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    tab["gap_change_lo"] = pd.Series({k: v[0] for k, v in ci.items()}); tab["gap_change_hi"] = pd.Series({k: v[1] for k, v in ci.items()})
    S["concepts"] = tab.round(3).sort_values("gap_change").to_dict("index")
    S["concept_by_year"] = R.round(3).to_dict("records")
    late24 = (per("gap", 2024, 2024) - per("gap", *EARLY)).round(3).to_dict(); S["gap_change_with_2024"] = late24

    plot_concepts(tab, FIG / "concept_change_clean.png")

    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.generic): return x.item()
        return x
    (OUT / "direction_summary_clean.json").write_text(json.dumps(clean(S), ensure_ascii=False, indent=1))
    print("saved")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help="redraw the concept figure from saved JSON (needs neither the data nor the API)")
    ap.add_argument("--json", default="direction_summary_clean.json", help="JSON in out/ to read with --plot-only")
    ap.add_argument("--fig", default="concept_change_clean.png", help="figure in out/figs/ to write with --plot-only")
    args = ap.parse_args()
    if args.plot_only:
        plot_concepts(pd.DataFrame(json.load(open(OUT / args.json))["concepts"]).T, FIG / args.fig)
        print("redrew", FIG / args.fig)
    else:
        main()
