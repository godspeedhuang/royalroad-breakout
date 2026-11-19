"""All tags in one chart: x = lift, y = tag, hollow dot = 2017–2019, filled dot = 2022–2024, right column = supply share.

2016 is excluded (its tag data is inconsistent with later years). 2020–2021 are left out to separate the two periods.
Outputs: out/figs/tags_compare.png, out/tags_compare.csv
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "out"
EARLY, LATE = (2017, 2019), (2022, 2024)
MIN_HOT = 30   # both periods need at least this many top-10% books, so small-sample lifts do not jump around
UP, DOWN, INK, MUTED, GRID = "#b84a32", "#0077b6", "#1b2829", "#566667", "#d6dedf"
plt.rcParams.update({"font.sans-serif": ["PingFang TC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"], "axes.unicode_minus": False})

b = pd.read_parquet(OUT / "books_meta.parquet", columns=["year", "top10", "tags"])
b["tag"] = b.tags.map(lambda s: json.loads(s) if isinstance(s, str) and s else [])


def stats(lo, hi):
    d = b[b.year.between(lo, hi)]
    x = d[["top10", "tag"]].explode("tag").dropna(subset=["tag"])
    n, n_top = len(d), int(d.top10.sum())
    s = x.groupby("tag").top10.agg(books="size", hot="sum")
    s["supply"] = s.books / n * 100
    s["hot_share"] = s.hot / n_top * 100
    s["lift"] = s.hot_share / s.supply
    return s


e, l = stats(*EARLY), stats(*LATE)
t = e.join(l, lsuffix="_e", rsuffix="_l", how="inner")
t = t[(t.hot_e >= MIN_HOT) & (t.hot_l >= MIN_HOT)].copy()
t["d_log2"] = np.log2(t.lift_l / t.lift_e)
t.sort_values("d_log2").round(3).to_csv(OUT / "tags_compare.csv")

t = t.sort_values("lift_l")
y = np.arange(len(t))
fig, ax = plt.subplots(figsize=(12, 13.5))
ax.set_xscale("log", base=2)
ax.axvline(1, color=MUTED, lw=1)
for yi, (tag, r) in zip(y, t.iterrows()):
    c = UP if r.lift_l >= r.lift_e else DOWN
    ax.plot([r.lift_e, r.lift_l], [yi, yi], color=c, lw=2, solid_capstyle="round", zorder=2)
    ax.plot(r.lift_e, yi, "o", ms=8, mfc="white", mec=c, mew=1.8, zorder=3)
    ax.plot(r.lift_l, yi, "o", ms=9, color=c, mec="white", mew=1, zorder=4)
    ax.text(5.4, yi, f"{r.supply_e:.0f}% → {r.supply_l:.0f}%", va="center", ha="left", fontsize=10, color=MUTED)
ax.text(5.4, len(t) + .1, "supply share", ha="left", va="bottom", fontsize=10, color=MUTED)
new_tag = t.index[t.supply_e < 3].tolist()
ax.set_yticks(y); ax.set_yticklabels([x + (" †" if x in new_tag else "") for x in t.index], fontsize=11, color=INK)
ax.set_ylim(-1, len(t) + .8); ax.set_xlim(.35, 5)
ticks = [.35, .5, .7, 1, 1.4, 2, 2.8, 4]
ax.set_xticks(ticks); ax.set_xticklabels([str(v) for v in ticks]); ax.minorticks_off()
ax.grid(True, axis="x", color=GRID, lw=.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.set_xlabel("x: lift = share among the top 10% ÷ share among all books (log scale; above 1 means more likely to take off)", fontsize=12, color=INK)
ax.set_ylabel("y: tag, sorted by 2022–2024 lift", fontsize=12, color=INK)
ax.set_title(f"Change in tag preference: {EARLY[0]}–{EARLY[1]} (hollow) → {LATE[0]}–{LATE[1]} (filled)", fontsize=14, color=INK, loc="left")
ax.plot([], [], color=UP, lw=2, marker="o", label="lift up"); ax.plot([], [], color=DOWN, lw=2, marker="o", label="lift down")
ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(.86, .01), fontsize=11)
fig.text(.01, .004, "† Early supply under 3%: the lift is unstable and the tag may simply have spread later. Only tags with at least 30 top-10% books in both periods.", fontsize=10, color=MUTED)
plt.tight_layout(rect=(0, .015, 1, 1))
plt.savefig(OUT / "figs" / "tags_compare.png", dpi=140, bbox_inches="tight")

print(len(t), "tags; up", int((t.d_log2 > 0).sum()), "down", int((t.d_log2 < 0).sum()))
print(t.sort_values("d_log2")[["supply_e", "supply_l", "lift_e", "lift_l"]].round(2).iloc[list(range(6)) + list(range(-6, 0))].to_string())
