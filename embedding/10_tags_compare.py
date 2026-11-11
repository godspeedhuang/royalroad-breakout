"""把各 tag 放在同一張圖比較：x = lift，y = tag，空心點 = 2017–2019，實心點 = 2022–2024，右欄是供給占比。

不含 2016 年（標籤資料和之後不一致）。2020–2021 也不算，讓兩個時期隔開。
輸出：out/figs/tags_compare.png、out/tags_compare.csv
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "out"
EARLY, LATE = (2017, 2019), (2022, 2024)
MIN_HOT = 30   # 兩個時期的前 10% 都至少要有這麼多本，避免小樣本的 lift 亂跳
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
ax.text(5.4, len(t) + .1, "供給占比", ha="left", va="bottom", fontsize=10, color=MUTED)
new_tag = t.index[t.supply_e < 3].tolist()
ax.set_yticks(y); ax.set_yticklabels([x + (" †" if x in new_tag else "") for x in t.index], fontsize=11, color=INK)
ax.set_ylim(-1, len(t) + .8); ax.set_xlim(.35, 5)
ticks = [.35, .5, .7, 1, 1.4, 2, 2.8, 4]
ax.set_xticks(ticks); ax.set_xticklabels([str(v) for v in ticks]); ax.minorticks_off()
ax.grid(True, axis="x", color=GRID, lw=.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.set_xlabel("x：lift = 前 10% 裡的占比 ÷ 全部書裡的占比（log 尺度；大於 1 表示比較容易紅）", fontsize=12, color=INK)
ax.set_ylabel("y：tag，依 2022–2024 的 lift 排序", fontsize=12, color=INK)
ax.set_title(f"各 tag 的偏好變化：{EARLY[0]}–{EARLY[1]}（空心）→ {LATE[0]}–{LATE[1]}（實心）", fontsize=14, color=INK, loc="left")
ax.plot([], [], color=UP, lw=2, marker="o", label="lift 上升"); ax.plot([], [], color=DOWN, lw=2, marker="o", label="lift 下降")
ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(.86, .01), fontsize=11)
fig.text(.01, .004, "† 早期供給不到 3%，lift 很不穩，而且可能是標籤後來才普及。只列兩個時期前 10% 都至少 30 本的 tag。", fontsize=10, color=MUTED)
plt.tight_layout(rect=(0, .015, 1, 1))
plt.savefig(OUT / "figs" / "tags_compare.png", dpi=140, bbox_inches="tight")

print(len(t), "tags；上升", int((t.d_log2 > 0).sum()), "下降", int((t.d_log2 < 0).sum()))
print(t.sort_values("d_log2")[["supply_e", "supply_l", "lift_e", "lift_l"]].round(2).iloc[list(range(6)) + list(range(-6, 0))].to_string())
