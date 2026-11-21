// Royal Road Breakout demo. Data comes from data.js (window.DEMO_DATA), made by modeling/05_demo_data.py.
// Tweak things in CONFIG. Each screen has one section below: its drawing functions and a step player
// (Play button, space bar). Arrow keys move between screens.

const CONFIG = {
  quarter: "Y2023",        // default quarter: "Y2023" = all of 2023, or a quarter index
  slots: 50,               // default slots per quarter (the slider runs from 10 to every serial of a quarter, finer steps at the low end)
  model: "lr_SAT",         // default model
  coversPerQuarter: 12,    // covers per quarter on the picks shelves before "Show all picks"
  coversMax: 100,          // at most this many covers per quarter after "Show all picks"
  coverW: 110,             // largest cover width on the shelves (height = 1.5 ×); smaller when space is short
  mosaic: { maxSize: 9, minSize: 3, gap: 1, dimAlpha: 0.2, missedAlpha: 0.38, fadeMs: 500 },
  waits: {                 // how long each step stays on screen while playing (ms); the last step stays
    why: [1250, 1250, 1250],
    hit: [1800, 3000],
    picks: [1500, 1500, 1500],
  },
  tPhaseMs: 1100,          // first-three-chapters animation: time per phase
  blocks: { S: "#45a89c", A: "#9a8ad6", T: "#5d9be0", C: "#d585a8" },   // one color per input block (kept quiet next to the orange)
  colors: { accent: "#d27a50", picked: "#5d9be0", rule: "#95a0a6", other: "#46545c", text: "#f2f4f5", muted: "#7a878e", light: "#e8e9ea", panel: "#000000" },
  sans: '"Helvetica Neue", Helvetica, Arial, sans-serif',   // labels inside canvases
  team: ["Yinqi (Erich) Yu", "Hongyu Yue", "Syuan-Bo Huang"],   // names on the cover page, as on the deck
  stories: {               // the four story buttons in the filter bar
    1: { model: "rule_hot_genre", authors: "all" },
    2: { model: "lr_S", authors: "all" },
    3: { model: "lr_SAT", authors: "all" },
    4: { model: "lr_SAT", authors: "first" },
  },
};

const D = window.DEMO_DATA, B = D && D.books, N = B ? B.id.length : 0;
const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const K_STEPS = [];   // slider positions for slots per quarter: steps of 5 up to 100, then 10, 50 and 100 (roughly a log scale), up to 4,000 (more than any quarter)
for (let k = 10; k <= 4000; k += k < 100 ? 5 : k < 500 ? 10 : k < 1000 ? 50 : 100) K_STEPS.push(k);
const kAt = (pos) => K_STEPS[Math.max(0, Math.min(K_STEPS.length - 1, Math.round(pos)))];
const posOfK = (k) => K_STEPS.reduce((b, x, i) => (Math.abs(x - k) < Math.abs(K_STEPS[b] - k) ? i : b), 0);
const slots = () => kAt(Number($("f-k").value)) || CONFIG.slots;
function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
function coverImg(src, alt = "") {   // a Royal Road cover; removes itself if the image fails
  const im = el("img", { src, alt, loading: "lazy", referrerpolicy: "no-referrer" });
  im.onerror = () => im.remove();
  return im;
}
const VEC_BASE = [24, 26, 29], VEC_POS = [236, 138, 92], VEC_NEG = [70, 160, 255];   // vector grids: zero, above zero, below zero (colors on a square-root scale)
const FX = {   // formulas shown on the page (KaTeX): [formula, legend, optional second formula]
  target: [String.raw`y_i = \mathbf{1}\!\left[\, f_i \ge Q_{0.9}\{\, f_j : q_j = q_i \,\}\right]`, "f: followers in June 2025 · q: start quarter · Q₀.₉: 90th percentile"],
  share: [String.raw`\textstyle \text{share of slots} = \frac{1}{4K}\sum_{q=1}^{4}\ \sum_{i\,\in\,\mathrm{Top}_K(q)} y_i`, "Topₖ(q): the K highest scores in quarter q · random ≈ 10%"],
  t: [String.raw`t_i = \frac{n^{\top} E_i}{\lVert n^{\top} E_i \rVert}, \quad E_i = \begin{bmatrix} e_1 \\ e_2 \\ e_3 \end{bmatrix} \in \mathbb{R}^{3 \times 1536}, \quad n = (n_1, n_2, n_3)`,
      "row c of Eᵢ: chapter c's 1,536 numbers (chunks over 8,000 tokens averaged by length) · n: tokens per chapter · match uses d from 2017–2022",
      String.raw`\text{match}_i = \cos(t_i,\, d), \quad d = \bar t_{\text{hit}} - \bar t_{\text{other}}`],
  cover: [String.raw`\operatorname{logit} P(y_i = 1) = s_i + w^{\top} z_i`, "sᵢ: the main model's score, held fixed · zᵢ: standardized 768 CLIP numbers · no intercept · w fitted on valid covers with an L2 penalty · no valid cover: + 0"],
  lift: [String.raw`\text{lift}(g) = \frac{P(\text{hit} \mid \text{tag } g)}{P(\text{hit})}`, "1 = average"],
};
function fx(key, target = el("div", { class: "formula" })) {   // render a formula (and its legend) into an element
  if (!key || !FX[key]) return target;
  const [src, legend, src2] = FX[key]; target.replaceChildren();
  for (const t of [src, src2].filter(Boolean)) {
    const m = el("div", { class: "fx-math" });
    if (window.katex) katex.render(t, m, { throwOnError: false, displayMode: true }); else m.textContent = t;   // display mode never breaks a formula over lines; no internet: show the source
    target.append(m);
  }
  if (legend) target.append(el("div", { class: "fx-legend" }, legend));
  return target;
}
function fitFormulas(root = document) {   // a formula that is wider than its box is scaled down instead of wrapped
  requestAnimationFrame(() => root.querySelectorAll(".fx-math").forEach((m) => {
    m.style.fontSize = "";
    if (!m.clientWidth) return;
    let fs = parseFloat(getComputedStyle(m).fontSize);
    while (m.scrollWidth > m.clientWidth + 1 && fs > 9) { fs -= 0.5; m.style.fontSize = fs + "px"; }
  }));
}
const restart = (node, cls) => { node.classList.remove(cls); node.offsetHeight; node.classList.add(cls); };   // replay a CSS animation

// ---------- selection: which serials, which picks ----------
function pool(authors = $("f-authors").value) {
  const qsel = $("f-quarter").value;
  const qs = qsel.startsWith("Y") ? D.quarters.map((q, i) => (q.startsWith(qsel.slice(1)) ? i : -1)).filter((i) => i >= 0) : [Number(qsel)];
  const byQ = new Map(qs.map((i) => [i, []]));
  for (let i = 0; i < N; i++) if (byQ.has(B.q[i]) && (authors === "all" || B.first[i])) byQ.get(B.q[i]).push(i);
  return byQ;
}
const tie = (a, c) => (B.ord ? B.ord[a] - B.ord[c] : B.id[a] - B.id[c]);   // equal scores: the deck's order (modeling/08_impact.py)
function pick(byQ, model, k) {           // each quarter: the k highest scores, picked exactly as the deck does
  const s = B.score[model], out = new Map();
  for (const [q, idx] of byQ) out.set(q, [...idx].sort((a, c) => s[c] - s[a] || tie(a, c)).slice(0, Math.min(k, idx.length)));
  return out;
}
function hits(picks) { let h = 0, n = 0; for (const ix of picks.values()) for (const i of ix) { h += B.y[i]; n++; } return { h, n, share: h / n }; }
function baseRate(byQ) { let h = 0, n = 0; for (const ix of byQ.values()) for (const i of ix) { h += B.y[i]; n++; } return h / n; }
function quarterLabel(byQ) { const q = [...byQ.keys()].map((i) => D.quarters[i]); return q.length > 1 ? `${q[0].slice(0, 4)}, ${q.length} quarters` : q[0]; }
const whoLabel = () => ($("f-authors").value === "first" ? "first-time authors only" : "all authors");

// ---------- step player: Play / Pause button, step chips, space bar ----------
function makePlayer(name, steps, waits) {
  const P = { i: -1, playing: false, timer: null };
  const ui = () => {
    const b = document.querySelector(`[data-play="${name}"]`);
    if (b) b.textContent = P.playing ? "❚❚ Pause" : P.i >= steps.length - 1 ? "↻ Replay" : "▶ Play";
    document.querySelectorAll(`[data-step-of="${name}"]`).forEach((c) => c.classList.toggle("on", Number(c.dataset.step) === P.i));
  };
  P.show = (i) => { P.i = i; steps[i](); ui(); };
  const tick = () => {
    P.show(P.i + 1);
    if (P.i >= steps.length - 1) { P.playing = false; ui(); return; }   // the last step stays
    P.timer = setTimeout(tick, waits[P.i] ?? 3000);
  };
  P.play = (fromStart) => { clearTimeout(P.timer); P.playing = true; if (fromStart || P.i >= steps.length - 1) P.i = -1; tick(); };
  P.pause = () => { clearTimeout(P.timer); P.playing = false; ui(); };
  P.toggle = () => (P.playing ? P.pause() : P.play());
  P.jump = (i) => { P.pause(); P.show(i); };
  P.finish = () => { if (P.i >= 0 && (P.playing || P.i < steps.length - 1)) { P.pause(); P.show(steps.length - 1); } };   // leaving the screen: show the end state
  return P;
}
const players = {};

// ---------- small SVG chart helpers (every chart fills its box; marks carry classes for entrance animations) ----------
const NS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.append(e);
  return e;
}
function tip(node, text) { svg("title", {}, node, text); return node; }
function newChart(box, maxH = Infinity, minH = 120) {   // empty the box; return an SVG as wide as the box, at most maxH tall
  box.replaceChildren();
  const W = box.clientWidth || 600, H = Math.max(minH, Math.min(maxH, box.clientHeight || minH));
  box.dataset.w = W;   // the width it was drawn at (see watchWidth)
  return { s: svg("svg", { class: "chart in", viewBox: `0 0 ${W} ${H}`, height: H }, box), W, H };
}
function labelWidth(s, labels) {         // width of the longest label, so labels never get cut off
  return Math.max(...labels.map((l) => { const t = svg("text", { x: -9999, y: -9999 }, s, l); const w = t.getComputedTextLength(); t.remove(); return w; })) + 14;
}
function watchWidth(box, redraw) {   // redraw a chart once its box has a new width (a drawer opened or closed next to it)
  let timer = 0;
  new ResizeObserver(() => {
    clearTimeout(timer);
    timer = setTimeout(() => { if (box.clientWidth && Math.abs(box.clientWidth - Number(box.dataset.w || 0)) >= 4) redraw(); }, 80);   // after the size stops changing
  }).observe(box);
}
function hideChart(box) {               // put a chart in its "before" state without animating
  const s = box.querySelector("svg.chart"); if (!s) return;
  s.classList.add("noanim"); s.classList.remove("in"); s.getBoundingClientRect(); s.classList.remove("noanim");
}
function animateIn(box, stagger = true) {   // replay a chart's entrance: bars grow, lines draw, dots slide, labels fade in (one after another, or all at once)
  const s = box.querySelector("svg.chart"); if (!s) return;
  hideChart(box); s.classList.toggle("stagger", stagger);
  requestAnimationFrame(() => requestAnimationFrame(() => s.classList.add("in")));
}

// Horizontal bars. rows: [{id, label, parts?, value, color, bold, title}]; parts = [[text, color]] colors parts of the label.
function barChart(box, rows, { max = 1, fmt = (v) => pct(v), ref, refLabel } = {}) {
  const foot = ref !== undefined ? 28 : 0, { s, W, H } = newChart(box, rows.length * 68 + foot);
  const labelW = labelWidth(s, rows.map((r) => r.label)), plotW = W - labelW - 70;
  const slot = (H - foot) / rows.length, barH = Math.min(32, slot * 0.6);
  rows.forEach((r, j) => {
    const y = j * slot + (slot - barH) / 2, w = Math.max(2, (r.value / max) * plotW);
    const g = svg("g", { "data-id": r.id || "", style: `--i:${j}` }, s);
    const t = svg("text", { x: labelW - 12, y: y + barH / 2 + 5, "text-anchor": "end", style: r.bold ? "fill:var(--text);font-weight:700" : "" }, g);
    if (r.parts) r.parts.forEach(([txt, c]) => svg("tspan", c ? { style: `fill:${c}` } : {}, t, txt)); else t.textContent = r.label;
    tip(svg("rect", { class: "grow" + (r.bold ? " glow" : ""), x: labelW, y, width: w, height: barH, rx: 3, fill: r.color }, g), r.title || `${r.label}: ${fmt(r.value)}`);
    svg("text", { class: "val fade", x: labelW + w + 8, y: y + barH / 2 + 5 }, g, fmt(r.value));
  });
  if (ref !== undefined) {
    const x = labelW + (ref / max) * plotW;
    svg("line", { x1: x, x2: x, y1: 0, y2: H - foot, stroke: CONFIG.colors.muted, "stroke-dasharray": "3 4" }, s);
    svg("text", { class: "muted", x, y: H - 6, "text-anchor": "middle" }, s, refLabel);
  }
  return s;
}

// Lines over a small x axis, one y axis. series: [{label, values, color, glow, dim, mark}]; mark = index of a point to ring ("chosen")
function lineChart(box, xs, series, { yMin, yMax, fmt = (v) => v.toFixed(2), xLabel = "" } = {}) {
  const { s, W, H } = newChart(box, 460);
  const L = 48, R = labelWidth(s, series.map((sr) => sr.label)) + 6, T = 14, Bm = 50;
  const X = (i) => L + (i / (xs.length - 1)) * (W - L - R), Y = (v) => T + (1 - (v - yMin) / (yMax - yMin)) * (H - T - Bm);
  const g = svg("g", { class: "grid" }, s);
  for (let k = 0; k <= 3; k++) {
    const v = yMin + (k / 3) * (yMax - yMin);
    svg("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v) }, g);
    svg("text", { class: "muted", x: L - 8, y: Y(v) + 4, "text-anchor": "end" }, s, fmt(v));
  }
  xs.forEach((x, i) => svg("text", { class: "muted", x: X(i), y: H - Bm + 22, "text-anchor": "middle" }, s, x));
  svg("text", { class: "muted", x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle" }, s, xLabel);
  const last = xs.length - 1, ends = series.map((sr, k) => ({ k, y: Y(sr.values[last]) + 5 })).sort((a, b) => a.y - b.y);
  ends.forEach((e, i) => { if (i && e.y < ends[i - 1].y + 16) e.y = ends[i - 1].y + 16; });   // end labels never overlap
  series.forEach((sr, k) => {
    svg("path", { class: "draw" + (sr.glow ? " glow" : ""), pathLength: 1, style: `--i:${k * 5}`, d: sr.values.map((v, i) => `${i ? "L" : "M"}${X(i)},${Y(v)}`).join(""), fill: "none", stroke: sr.color, "stroke-width": sr.dim ? 1.6 : 2.5 }, s);
    sr.values.forEach((v, i) => tip(svg("circle", { class: "fade", style: `--i:${k * 5 + i}`, cx: X(i), cy: Y(v), r: sr.dim ? 3.5 : 5, fill: sr.color, stroke: "var(--panel)", "stroke-width": 2 }, s), `${sr.label}, ${xs[i]}: ${fmt(v)}`));
    if (sr.mark !== undefined && sr.mark >= 0) {
      const v = sr.values[sr.mark];
      svg("circle", { class: "fade", cx: X(sr.mark), cy: Y(v), r: 10, fill: "none", stroke: sr.color, "stroke-width": 1.5 }, s);
      svg("text", { class: "val fade", x: X(sr.mark), y: Y(v) - 16, "text-anchor": "middle" }, s, `chosen: ${fmt(v)}`);
    }
    svg("text", { class: (sr.dim ? "muted" : "val") + " fade", style: `--i:${k * 5 + last}`, x: X(last) + 12, y: ends.find((e) => e.k === k).y }, s, sr.label);
  });
}

// Before → after per row. rows: [{label, a, b}]
function dumbbell(box, rows, { min, max, ref = 1, aLabel, bLabel } = {}) {
  const { s, W, H } = newChart(box, rows.length * 52 + 34);
  const labelW = labelWidth(s, rows.map((r) => r.label)), plotW = W - labelW - 50, rowH = (H - 34) / rows.length;
  const X = (v) => labelW + ((v - min) / (max - min)) * plotW;
  svg("line", { x1: X(ref), x2: X(ref), y1: 0, y2: H - 30, stroke: CONFIG.colors.muted, "stroke-dasharray": "3 4" }, s);
  svg("text", { class: "muted", x: X(ref), y: H - 8, "text-anchor": "middle" }, s, "1.0 = average");
  rows.forEach((r, j) => {
    const y = j * rowH + rowH / 2, up = r.b > r.a, c = up ? CONFIG.colors.accent : CONFIG.colors.rule;
    const g = svg("g", { style: `--i:${j}` }, s);
    svg("text", { x: labelW - 12, y: y + 5, "text-anchor": "end" }, g, r.label);
    svg("path", { class: "draw", pathLength: 1, d: `M${X(r.a)},${y}H${X(r.b)}`, stroke: c, "stroke-width": 2.5 }, g);
    tip(svg("circle", { cx: X(r.a), cy: y, r: 5, fill: CONFIG.colors.other, stroke: CONFIG.colors.rule }, g), `${r.label}, ${aLabel}: ${r.a}`);
    tip(svg("circle", { class: "slide" + (up ? " glow" : ""), style: `--dx:${X(r.a) - X(r.b)}px`, cx: X(r.b), cy: y, r: 7, fill: c }, g), `${r.label}, ${bLabel}: ${r.b}`);
    svg("text", { class: "val fade", x: X(r.b) + (up ? 12 : -12), y: y + 5, "text-anchor": up ? "start" : "end" }, g, r.b.toFixed(2));
  });
}

// Two bars per group: a = without, b = with. groups: [{label, a, b}]
function pairBars(box, groups, { aLabel, bLabel, max, fmt = (v) => pct(v), bColor = CONFIG.colors.accent } = {}) {
  const { s, W, H } = newChart(box, groups.length * 140);
  const labelW = labelWidth(s, groups.map((g) => g.label)), plotW = W - labelW - 70, rowH = H / groups.length, barH = Math.min(30, rowH * 0.3);
  groups.forEach((g, j) => {
    const y = j * rowH + (rowH - 2 * barH - 4) / 2, grp = svg("g", { style: `--i:${j * 2}` }, s);
    svg("text", { x: labelW - 12, y: y + barH + 6, "text-anchor": "end" }, grp, g.label);
    [[g.a, CONFIG.colors.other, aLabel], [g.b, bColor, bLabel]].forEach(([v, c, lab], k) => {
      const yy = y + k * (barH + 4), w = Math.max(2, (v / max) * plotW);
      tip(svg("rect", { class: "grow" + (k ? " glow" : ""), x: labelW, y: yy, width: w, height: barH, rx: 3, fill: c }, grp), `${g.label}, ${lab}: ${fmt(v)}`);
      svg("text", { class: "val fade", x: labelW + w + 8, y: yy + barH / 2 + 5 }, grp, fmt(v));
    });
  });
}

// ---------- cover page: a tilted wall of real covers behind the title ----------
function heroInit() {
  const wall = $("hero-wall"), s = B.score.lr_SAT;
  const ids = [...Array(N).keys()].filter((i) => B.cover[i]).sort((a, c) => s[c] - s[a]).slice(0, 96);   // the model's favourite serials with covers
  ids.forEach((i, j) => { const t = el("div", { class: "ht", style: `--j:${j % 12}` }); t.append(coverImg(B.cover[i])); wall.append(t); });
  $("hero-team").textContent = CONFIG.team.join("   ·   ");
}

// ---------- screen 0: why we built this ----------
const FLOW = [   // "what we built", top to bottom
  { t: "Royal Road", d: "An English web-fiction platform: serials published chapter by chapter, free to read", covers: true },
  { t: "Public dataset", d: "RoyalRoad-1.61M on Hugging Face: 1.61 million chapters, followers as of June 2025" },
  { t: "Our sample", d: () => `${D.why.sample_books.toLocaleString()} serials that started ${D.why.sample_years[0]}–${D.why.sample_years[1]} with at least 3 chapters` },
  { t: "At chapter 3", d: "Only what an editor could see then", chips: [["S", "tags and structure"], ["A", "author"], ["T", "first three chapters"]] },
  { t: "Model", d: "Logistic regression scores every new serial" },
  { t: "Recommendation slots", d: "Each quarter the highest scores fill the slots; tested on 2023, a year the model never saw" },
];
function whyInit() {
  const W0 = D.why, wall = $("wall"); wall.replaceChildren();
  W0.wall.forEach((it, j) => {
    const t = el("div", { class: "wt" + (it.y ? " hit" : "") + (it.f ? " first" : ""), title: it.t, style: `--j:${j}` });
    t.append(coverImg(it.u));
    if (it.f) t.append(el("span", { class: "badge" }, "1st"));
    wall.append(t);
  });
  const f = $("funnel"); f.replaceChildren();
  const per = W0.per_quarter_2023;   // each bar is a share of its whole: serials, or all views
  [{ n: per.toLocaleString(), f: 1, lab: "new serials per quarter (2023)", c: "#6c7a82", s: 0 },
   { n: String(W0.hits_per_quarter_2023), f: W0.hits_per_quarter_2023 / per, lab: "become hits: the top 10% of their quarter", c: CONFIG.colors.accent, s: 1 },
   { n: pct(W0.views_share_hits_2023, 0), f: W0.views_share_hits_2023, lab: "of all views go to those hits (views as of June 2025)", c: CONFIG.colors.accent, s: 1 },
   { n: String(CONFIG.slots), f: CONFIG.slots / per, lab: "recommendation slots per quarter (our scenario)", c: CONFIG.colors.picked, s: 3 }].forEach((row) => {
    const r = el("div", { class: "fr", "data-s": row.s }), bar = el("i");
    bar.style.background = row.c; bar.style.width = `max(6px, ${100 * row.f}%)`; bar.style.boxShadow = row.s ? `0 0 8px ${row.c}, 0 0 22px ${row.c}99` : "none";   // glow (99 = 60% alpha)
    const track = el("span", { class: "track" }); track.append(bar);
    r.append(el("b", {}, row.n), track, el("span", { class: "lab" }, row.lab));
    f.append(r);
  });
  const flow = $("flow"); flow.replaceChildren();
  const hitCovers = W0.wall.filter((it) => it.y).slice(0, 5);
  FLOW.forEach((n, k) => {
    const node = el("div", { class: "fnode", style: `--k:${k}` }), body = el("div");
    body.append(el("b", {}, n.t), el("span", {}, typeof n.d === "function" ? n.d() : n.d));
    if (n.chips) { const c = el("div", { class: "fchips" }); n.chips.forEach(([key, lab]) => { const s = el("span", {}); s.style.setProperty("--c", CONFIG.blocks[key]); s.append(el("em", {}, key), lab); c.append(s); }); body.append(c); }
    if (n.covers) { const c = el("div", { class: "fcovers" }); hitCovers.forEach((it) => { const d = el("i"); d.append(coverImg(it.u)); c.append(d); }); node.append(c); }
    node.prepend(el("i", { class: "fdot" }), body);
    flow.append(node);
  });
}
function layoutWall() {   // largest tile size that fits all covers in the wall's box
  const wall = $("wall"), n = D.why.wall.length, g = 6, W = wall.clientWidth, H = wall.clientHeight;
  let best = { w: 0, c: 16 };
  for (let c = 8; c <= 24; c++) { const r = Math.ceil(n / c), w = Math.min((W - (c - 1) * g) / c, ((H - (r - 1) * g) / r) / 1.5); if (w > best.w) best = { w, c }; }
  wall.style.gridTemplateColumns = `repeat(${best.c}, ${Math.floor(best.w)}px)`;
  wall.style.gridAutoRows = `${Math.floor(best.w * 1.5)}px`;
}
function whyShow(step) {
  const sec = $("why"), W0 = D.why;
  if (step === 0) { restart($("wall"), "in"); restart($("flow"), "run"); }
  [1, 2, 3].forEach((k) => sec.classList.toggle(`s${k}`, step >= k));
  $("why-cap").textContent = [
    `Every quarter, about ${W0.per_quarter_2023.toLocaleString()} new serials start on Royal Road (2023).`,
    `About ${W0.hits_per_quarter_2023} of them become hits, the top 10% of their quarter, and those hits collect ${pct(W0.views_share_hits_2023, 0)} of all views.`,
    `Two in three are first books, with no track record: they become hits ${pct(W0.hit_rate_first_2023)} of the time, returning authors ${pct(W0.hit_rate_returning_2023)}.`,
    `With ${CONFIG.slots} slots a quarter, which serials should get them? We try to spot the future hits at chapter 3.`,
  ][step];
}

// ---------- screen 1: what counts as a hit (the four 2016 quarters, counted in June 2025) ----------
const A = { step: 0 };
function animInit() {
  const box = $("anim"); box.replaceChildren();
  A.line = box.appendChild(el("div", { class: "tl" }));
  A.gap = box.appendChild(el("div", { class: "gaplab" }, "2017 … 2024 skipped"));
  A.pulse = box.appendChild(el("div", { class: "pulse" }));
  A.node = box.appendChild(el("div", { class: "node" }));
  A.node.append(el("i"), el("b", {}, "June 2025"), el("span", {}, "followers counted once, for every serial"));
  A.maxLog = Math.log10(1 + Math.max(...D.anim.quarters.flatMap((q) => q.items.map((it) => it.f))));
  A.cols = D.anim.quarters.map((qd) => {
    const head = box.appendChild(el("div", { class: "colhead" }));
    head.append(el("i"), el("b", {}, qd.q.replace("Q", " Q")), el("span", {}, `${qd.n} serials started, 20 shown`));
    const cut = box.appendChild(el("div", { class: "cut" }, `top 10% = ${D.anim.thresholds[qd.q]}+ followers`));
    const rank = [...qd.items.keys()].sort((a, b) => qd.items[b].f - qd.items[a].f);   // position of each serial after sorting
    const rows = qd.items.map((it, j) => {
      const r = box.appendChild(el("div", { class: "row", title: `${it.t}: ${it.f.toLocaleString()} followers in June 2025` }));
      const sq = r.appendChild(el("i", { class: "sq" }));
      sq.style.background = it.c;   // genre color shows through when there is no cover
      if (it.u) sq.append(coverImg(it.u));
      r.appendChild(el("i", { class: "bar" }));
      r.appendChild(el("span", { class: "num" }, it.f.toLocaleString()));
      return { r, it, j, sorted: rank.indexOf(j) };
    });
    return { head, cut, rows };
  });
  animShow(0, true);
}
function animShow(step, instant = false) {
  A.step = step;
  const box = $("anim"), W = box.clientWidth, H = box.clientHeight;
  const nodeW = 230, colW = (W - nodeW - 90) / 4, axisY = 14, top = 104, gapH = 26, rowH = Math.min(31, (H - top - 10 - gapH) / 20);
  const nodeX = W - nodeW / 2, barMax = colW - 40 - 22 - 64;
  A.line.style.left = colW / 2 + "px"; A.line.style.top = axisY + "px"; A.line.style.width = nodeX - colW / 2 + "px";
  A.gap.style.left = 4 * colW + "px"; A.gap.style.width = W - nodeW - 4 * colW + "px"; A.gap.style.top = axisY - 22 + "px";
  A.node.style.left = W - nodeW + "px"; A.node.style.width = nodeW + "px"; A.node.style.top = axisY - 8 + "px";
  A.node.classList.toggle("on", step >= 1);
  A.pulse.style.top = axisY - 6 + "px";
  A.pulse.style.left = (step >= 1 ? nodeX : colW / 2) - 6 + "px";
  A.pulse.style.opacity = step === 1 ? 1 : 0;
  A.cols.forEach((c, k) => {
    const x0 = k * colW + 20;
    c.head.style.left = x0 + "px"; c.head.style.top = axisY - 6 + "px"; c.head.style.width = colW - 40 + "px";
    c.cut.style.left = x0 + "px"; c.cut.style.width = colW - 40 + "px"; c.cut.style.top = top + 2 * rowH + 2 + "px";
    c.cut.style.opacity = step === 2 ? 1 : 0;
    c.rows.forEach((o) => {
      const slot = step === 2 ? o.sorted : o.j, bar = o.r.children[1], extra = step === 2 && slot >= 2 ? gapH : 0;   // room for the cut line
      o.r.style.width = colW - 40 + "px";
      o.r.style.transform = `translate(${x0}px, ${top + slot * rowH + extra}px)`;
      o.r.classList.toggle("hit", step === 2 && o.it.y === 1);
      o.r.style.opacity = step === 2 && !o.it.y ? 0.4 : 1;
      bar.style.width = step >= 1 ? (Math.log10(1 + o.it.f) / A.maxLog) * barMax + "px" : "0px";
      bar.style.transitionDelay = step === 1 && !instant ? "1.1s" : "0s";
      o.r.children[2].style.opacity = step >= 1 ? 1 : 0;
      o.r.children[2].style.transitionDelay = bar.style.transitionDelay;
    });
  });
  if (step === 0 && !instant) {   // serials drop in, one quarter after another
    const rows = A.cols.flatMap((c) => c.rows.map((o) => o.r));
    rows.forEach((r) => { r.style.transition = "none"; r.style.opacity = 0; });
    box.offsetHeight;
    rows.forEach((r, j) => { r.style.transition = ""; r.style.transitionDelay = (j % 20) * 25 + Math.floor(j / 20) * 120 + "ms"; r.style.opacity = 1; });
  } else A.cols.forEach((c) => c.rows.forEach((o) => (o.r.style.transitionDelay = "0s")));
  const th = D.anim.thresholds;
  $("anim-cap").textContent = [
    "Take the serials that started in 2016, one column per quarter.",
    "Their followers are counted only once: in the June 2025 snapshot, years later. A newer serial has had far less time to collect them.",
    `So each quarter gets its own top 10%: those are the hits. In 2016 Q1 a hit needed ${th["2016Q1"]} followers; in 2024 Q4, only ${th["2024Q4"]}.`,
  ][step];
}

// ---------- screen 2: how we chose the model (inputs × algorithm, details, results) ----------
const S3 = { inputs: new Set(["S", "A", "T"]), method: "lr", focus: "lr", say: null, dkey: null };
const nS = () => D.features.S.tags.length + D.features.S.warnings.length + 5;
const nInputs = () => nS() + 2 + 1536;
const BLOCKS = [
  { key: "S", name: "Tags and structure", summary: "tags, warnings, length, pace", dims: () => `${nS()} numbers` },
  { key: "A", name: "Author", summary: "first book? earlier books", dims: () => "2 numbers" },
  { key: "T", name: "First three chapters", summary: "the story as a text embedding", dims: () => "1,536 numbers" },
  { key: "C", name: "Cover", summary: "image embedding, on top", dims: () => "tried", tried: true },
];
const METHODS = [
  { key: "rule", name: "Editors' rule", summary: "baseline: popular genres" },
  { key: "lr", name: "Logistic regression", summary: "our model: one weight per number" },
  { key: "gb", name: "Gradient boosting", summary: "up to 300 small trees" },
  { key: "mlp", name: "Neural network", summary: "MLP, two hidden layers" },
];
const sh = (m) => pct(hits(pick(pool(), m, slots())).share);
const deckDefault = () => slots() === 50 && $("f-quarter").value === "Y2023" && $("f-authors").value === "all";   // the deck's setting: its confidence intervals apply
function modelOf() {   // the trained model behind the current choice, or null if we never trained that combination
  if (S3.method === "rule") return "rule_hot_genre";
  const id = `${S3.method}_${["S", "A", "T", "C"].filter((k) => S3.inputs.has(k)).join("")}`;
  return D.models[id] ? id : null;
}
function syncBuilder() {   // model picked in the filter bar → builder
  const m = $("f-model").value;
  if (m === "rule_hot_genre") { S3.method = "rule"; return; }
  if (m === "day30") { S3.method = "lr"; S3.inputs = new Set("SAT"); return; }   // the day-30 second pass builds on LR S+A+T
  const [meth, key] = m.split("_"); S3.method = meth; S3.inputs = new Set(key);
}
function manualMode() { players.model.pause(); S3.say = null; }
function setDetails(open) {   // the details drawer on the right; the chart takes the room when it is closed
  const cols = $("model-cols"); if (cols.classList.contains("d-open") === open) return false;
  cols.classList.toggle("d-open", open);   // the chart redraws itself at its new width (watchWidth)
  return true;
}
function builderClick(kind, key, toggle) {   // toggle = the switch on an input was clicked (otherwise just show its details)
  manualMode(); const moved = setDetails(true);
  if (kind === "block" && toggle) { S3.inputs.has(key) ? S3.inputs.delete(key) : S3.inputs.add(key); if (S3.method === "rule") S3.method = "lr"; }
  if (kind === "method") S3.method = key;
  S3.focus = key; S3.dkey = null;   // redraw (and replay) the details even if nothing else changed
  const m = modelOf();
  if (m) { $("f-model").value = m; render(); } else renderBuilder(pool(), slots());
  if (kind === "block") setTimeout(() => flashBars(key), moved ? 560 : 0);   // after the chart has settled at its new width
}
function modelIntro() {   // Play: every bar grows at once, with the chosen model (logistic regression on S+A+T) lit
  S3.method = "lr"; S3.inputs = new Set(["S", "A", "T"]); S3.focus = "lr"; S3.dkey = null;
  S3.say = () => {
    const dk = D.deck, def = deckDefault(), ci = def && dk ? ` (95% CI ${pct(dk.precision_at_slots.lr_SAT.ci95[0], 0)} to ${pct(dk.precision_at_slots.lr_SAT.ci95[1], 0)})` : "";
    const lead = dk && dk.diff_pp_2023_2024 ? `; over 2023 and 2024 together the MLP leads by ${dk.diff_pp_2023_2024.mlp_vs_lr.diff.toFixed(1)} points (95% CI ${dk.diff_pp_2023_2024.mlp_vs_lr.ci95[0].toFixed(1)} to ${dk.diff_pp_2023_2024.mlp_vs_lr.ci95[1].toFixed(1)})` : "";
    return `LR on S+A+T gives ${sh("lr_SAT")} of the slots to future hits${ci}; the editors' rule, ${sh("rule_hot_genre")}. Boosting (${sh("gb_SAT")}) and the MLP (${sh("mlp_SAT")}) are not significantly better in 2023${lead}. We keep the simple, explainable model and recheck at each yearly retrain.`;
  };
  $("f-model").value = "lr_SAT"; render();
  animateIn($("models-chart"), false);
}
function flashBars(key) {   // an input was clicked: every bar whose model uses it flashes once in the input's color
  document.querySelectorAll("#models-chart g[data-id]").forEach((g) => {
    const id = g.dataset.id;
    const used = id === "day30" ? "SAT" : id.split("_")[1] || "";
    if (id === "rule_hot_genre" || !used.includes(key)) return;
    g.style.setProperty("--fc", CONFIG.blocks[key]); g.classList.remove("flash"); g.getBBox(); g.classList.add("flash");
  });
}
const soft = (hex, a = "33") => hex + a;   // #rrggbb + alpha
function labelParts(m) {   // "S+A+T" with each block letter in its color
  const C = CONFIG.blocks;
  if (m === "rule_hot_genre") return [["Editors' rule", null]];
  if (m === "lr_T") return [["Text only", C.T]];
  if (m === "day30") return [["S", C.S], ["+", null], ["A", C.A], ["+", null], ["T", C.T], [" + day 30", null]];
  const [meth, key] = m.split("_");
  const parts = key.split("").flatMap((k, i) => [...(i ? [["+", null]] : []), [k === "C" ? "Cover" : k, C[k]]]);
  return [...(meth === "gb" ? [["Boosting ", null]] : meth === "mlp" ? [["MLP ", null]] : []), ...parts];
}
function barColor(m) {   // quiet grays; only the selected model is orange (the label letters carry the block colors)
  return m === "rule_hot_genre" ? "#76797d" : "#3a3d41";
}
const sigmoidD = (cx, cy, hw, hh) => Array.from({ length: 21 }, (_, i) => { const x = i / 10 - 1; return `${i ? "L" : "M"}${(cx + x * hw).toFixed(1)},${(cy + hh - (2 * hh) / (1 + Math.exp(-5 * x))).toFixed(1)}`; }).join("");
function methodIcon(key) {   // small line icons for the four algorithms
  const s = svg("svg", { class: "micon", viewBox: "0 0 28 28" });
  if (key === "rule") [7, 14, 21].forEach((y, i) => svg("line", { x1: 4, x2: 24 - i * 7, y1: y, y2: y }, s));   // a ranked list
  else if (key === "lr") svg("path", { d: sigmoidD(14, 14, 11, 9) }, s);   // a sigmoid
  else if (key === "gb") { svg("path", { d: "M14 6V13M14 13L7 21M14 13L21 21" }, s); [[14, 6], [14, 13], [7, 21], [21, 21]].forEach(([cx, cy]) => svg("circle", { cx, cy, r: 2.6 }, s)); }   // a tree
  else if (key === "mlp") {   // a small network
    const L = [[5, [9, 19]], [14, [5, 14, 23]], [23, [14]]];
    L.forEach(([x, ys], i) => { if (i) for (const y1 of L[i - 1][1]) for (const y2 of ys) svg("line", { x1: L[i - 1][0], y1, x2: x, y2, class: "thin" }, s); });
    L.forEach(([x, ys]) => ys.forEach((cy) => svg("circle", { cx: x, cy, r: 2.4 }, s)));
  }
  return s;
}
function option(cls, color, key, name, summary, right) {   // key: a letter or an icon; right: "switch", a short tag, or nothing
  const b = el("button", { class: cls }), k = el("span", { class: "key" });
  b.style.setProperty("--c", color); b.style.setProperty("--c-soft", soft(color, "2e")); b.style.setProperty("--c-glow", soft(color, "80"));
  k.append(key); b.append(k, el("b", {}, name), el("span", { class: "sum" }, summary));
  if (right === "switch") b.append(el("span", { class: "sw", title: "add to / remove from the model" }));
  else if (right) b.append(el("span", { class: "dims" }, right));
  return b;
}
function drawShares(byQ, k) {   // the share-of-slots bars; the selected model is orange
  const sel = modelOf(), base = baseRate(byQ), shares = {};
  const rows = Object.keys(D.models).map((m) => {
    const r = hits(pick(byQ, m, k)); shares[m] = r.share;
    return { id: m, label: D.short[m], parts: labelParts(m), value: r.share, bold: m === sel, title: `${D.models[m]}: ${r.h} hits in ${r.n} slots (${pct(r.share)})`,
             color: m === sel ? CONFIG.colors.accent : barColor(m) };
  });
  barChart($("models-chart"), rows, { max: Math.max(...rows.map((r) => r.value), base) * 1.1, ref: base, refLabel: `random ${pct(base, 0)}` });
  return shares;
}
function renderBuilder(byQ, k) {
  const sel = modelOf();
  const bl = $("blocks"); bl.replaceChildren();
  for (const b of BLOCKS) {
    const on = S3.inputs.has(b.key) && S3.method !== "rule";
    const btn = option(`opt${on ? " on" : ""}${b.tried ? " tried" : ""}${S3.focus === b.key ? " focus" : ""}`, CONFIG.blocks[b.key], b.key, b.name, `${b.dims()} · ${b.summary}`, "switch");
    btn.addEventListener("click", (ev) => builderClick("block", b.key, !!ev.target.closest(".sw")));
    bl.append(btn);
  }
  const me = $("methods"); me.replaceChildren();
  for (const m of METHODS) {
    const btn = option(`opt meth${S3.method === m.key ? " on" : ""}${S3.focus === m.key ? " focus" : ""}`, CONFIG.colors.accent, methodIcon(m.key), m.name, m.summary);
    btn.addEventListener("click", () => builderClick("method", m.key));
    me.append(btn);
  }
  renderDetail();
  const shares = drawShares(byQ, k);
  $("models-cap").textContent = `${k} slots per quarter · ${quarterLabel(byQ)} · ${whoLabel()}`;
  const manual = S3.method === "rule" ? `The rule only looks at tags: ${pct(shares.rule_hot_genre)} of slots.` :
    sel ? `${D.models[sel]}: ${pct(shares[sel])} of the ${k} slots per quarter went to hits.` :
    S3.inputs.has("C") ? "Not trained: the cover was only added on top of S+A+T with logistic regression." :
    S3.method === "lr" ? "Not trained: logistic regression ran on S, S+A, S+A+T and T alone." : "Not trained: boosting and the MLP ran on S+A+T only.";
  $("model-cap").textContent = S3.say ? S3.say() : manual;
  const all = [...pool("all").values()].flat(), kf = Math.max(1, Math.round((k * all.filter((i) => B.first[i]).length) / all.length));
  const ft = pool("first"), mft = hits(pick(ft, "lr_SAT", kf)), rft = hits(pick(ft, "rule_hot_genre", kf));
  $("ft-mult").textContent = (mft.share / rft.share).toFixed(1) + "×";
  $("ft-cap").textContent = `the editors' rule. With their share of the slots (${kf} of ${k} a quarter), ${pct(mft.share, 0)} of first-time authors' slots go to future hits, against ${pct(rft.share, 0)} (${mft.h} vs ${rft.h})`;
  const d30 = D.deck && D.deck.day30;
  if (D.models.day30) {
    $("d30-big").textContent = sh("day30");
    $("d30-cap").textContent = `of the slots once the first month's posting rhythm is added (S+A+T alone ${sh("lr_SAT")}). A second pass, to A/B test first.`;
  } else if (d30) {
    $("d30-big").textContent = pct(d30.test.day30, 0);
    $("d30-cap").textContent = `of 50 slots in 2023 once the first month's posting rhythm is added, up from ${pct(d30.test.lr_SAT, 0)} (+${d30.test.diff_pp.toFixed(1)} points, 95% CI ${d30.test.diff_ci95[0].toFixed(1)} to ${d30.test.diff_ci95[1].toFixed(1)}); first-time authors ${pct(d30.test_first_time.lr_SAT, 0)} → ${pct(d30.test_first_time.day30, 0)}. A second pass, to A/B test first.`;
  } else $("d30-card").style.display = "none";
}

// Detail panel: what is inside the focused input or algorithm (always fits the panel, no scrolling)
function section(parent, title, ...children) { const s = el("div", { class: "sec" }); s.append(el("h4", {}, title), ...children); parent.append(s); return s; }
function para(text, cls = "note") { return el("p", { class: cls }, text); }
function chips(list) {   // list: [label, count?, color?]
  const box = el("div", { class: "chips" }), max = Math.max(...list.map((x) => x[1] || 1));
  for (const [t, n, c] of list) {
    const ch = el("span", { class: "chip", title: n ? `${t}: ${n.toLocaleString()} serials` : t }, t);
    if (c) { ch.style.background = soft(c, "40"); ch.style.borderColor = "transparent"; }
    if (n) ch.style.opacity = 0.5 + 0.5 * Math.sqrt(n / max);
    box.append(ch);
  }
  return box;
}
let heatId = 0;
function heatmap(rows, cols, get, { rowName = "", colName = "", fmt = (v) => v.toFixed(3), scale = "AUC on 2022" } = {}) {   // a standard heatmap: rows on the y axis, columns on the x axis, lighter = higher, color scale on the right
  const cw = 132, ch = 52, L = 104, T = 8, gw = cols.length * cw, gh = rows.length * ch, W = L + gw + 118, H = T + gh + 46, id = `heat${heatId++}`;
  const s = svg("svg", { class: "heat", viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  const vals = rows.flatMap((r) => cols.map((c) => get(r, c).v)), lo = Math.min(...vals), hi = Math.max(...vals);
  const dark = [30, 32, 35], light = [112, 116, 121], shade = (t) => `rgb(${dark.map((x, i) => Math.round(x + (light[i] - x) * t))})`;
  rows.forEach((r, i) => {
    svg("text", { x: L - 10, y: T + i * ch + ch / 2 + 5, "text-anchor": "end", class: "tick" }, s, String(r));
    cols.forEach((c, j) => {
      const v = get(r, c), x = L + j * cw, y = T + i * ch, sub = [v.chosen ? "chosen" : "", v.sub || ""].filter(Boolean).join(" · ");
      tip(svg("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, fill: shade((v.v - lo) / (hi - lo || 1)) }, s), `${rowName} ${r}, ${colName} ${c}: ${fmt(v.v)}${v.sub ? " · " + v.sub : ""}`);
      if (v.chosen) svg("rect", { x: x + 1, y: y + ch - 5, width: cw - 2, height: 4, class: "pick" }, s);
      svg("text", { x: x + cw / 2, y: y + (sub ? 24 : 32), "text-anchor": "middle", class: "v" }, s, fmt(v.v));
      if (sub) svg("text", { x: x + cw / 2, y: y + 40, "text-anchor": "middle", class: "s" + (v.chosen ? " on" : "") }, s, sub);
    });
  });
  cols.forEach((c, j) => svg("text", { x: L + j * cw + cw / 2, y: T + gh + 18, "text-anchor": "middle", class: "tick" }, s, String(c)));
  svg("text", { x: L + gw / 2, y: T + gh + 40, "text-anchor": "middle", class: "axis" }, s, colName);
  svg("text", { x: -(T + gh / 2), y: 14, transform: "rotate(-90)", "text-anchor": "middle", class: "axis" }, s, rowName);
  const gr = svg("linearGradient", { id, x1: 0, y1: 1, x2: 0, y2: 0 }, svg("defs", {}, s));   // the color scale: low at the bottom, high at the top
  svg("stop", { offset: 0, "stop-color": shade(0) }, gr); svg("stop", { offset: 1, "stop-color": shade(1) }, gr);
  const bx = L + gw + 22;
  svg("rect", { x: bx, y: T + 1, width: 12, height: gh - 2, fill: `url(#${id})` }, s);
  svg("text", { x: bx + 18, y: T + 12, class: "tick" }, s, fmt(hi));
  svg("text", { x: bx + 18, y: T + gh - 3, class: "tick" }, s, fmt(lo));
  svg("text", { x: bx + 6, y: T + gh + 40, "text-anchor": "middle", class: "axis" }, s, scale);
  return s;
}
function vecGrid(v, scale, s = 5) {   // 1,536 numbers as a 48 × 32 grid: orange above zero, blue below
  const cv = el("canvas", { class: "vec" }), cols = 48, rows = v.length / cols, dpr = window.devicePixelRatio || 1;
  cv.width = cols * s * dpr; cv.height = rows * s * dpr; cv.style.width = cols * s + "px"; cv.style.height = rows * s + "px";
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
  drawVec(cv, v, scale, s);
  cv.addEventListener("mousemove", (ev) => {
    const r = cv.getBoundingClientRect(), z = r.width / cv.offsetWidth, i = Math.floor((ev.clientY - r.top) / z / s) * cols + Math.floor((ev.clientX - r.left) / z / s);
    if (v[i] !== undefined && $("vec-readout")) $("vec-readout").textContent = `number ${i + 1} of 1,536: ${v[i].toFixed(4)}`;
  });
  return cv;
}

// First three chapters → tokens → one vector per chapter → their weighted average (plays when T is opened by hand)
const TOKEN_COLORS = ["#45a89c", "#9a8ad6", "#5d9be0", "#d585a8", "#c9a35a"];
let tTimers = [], tFrame = 0;
function drawVec(cv, v, scale, s) {   // paint 1,536 numbers into an existing grid canvas
  const ctx = cv.getContext("2d"), cols = 48, dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const mix = (a, b, t) => a.map((x, i) => Math.round(x + (b[i] - x) * t));
  v.forEach((x, i) => {
    const t = Math.min(1, Math.sqrt(Math.abs(x) / scale)), c = mix(VEC_BASE, x > 0 ? VEC_POS : VEC_NEG, t);
    ctx.fillStyle = `rgb(${c})`; ctx.fillRect((i % cols) * s, Math.floor(i / cols) * s, s - 0.5, s - 0.5);
  });
}
function drawStrip(cv, v, scale) {   // 1,536 numbers as one thin row: orange above zero, blue below; one sharp color per screen pixel (no blending into brown)
  const ctx = cv.getContext("2d"), Wd = cv.width, Hd = cv.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, Wd, Hd);
  for (let px = 0; px < Wd; px++) {
    const x = v[Math.floor((px * v.length) / Wd)], t = Math.min(1, Math.sqrt(Math.abs(x) / scale)), c = (x > 0 ? VEC_POS : VEC_NEG).map((p, j) => Math.round(VEC_BASE[j] + (p - VEC_BASE[j]) * t));
    ctx.fillStyle = `rgb(${c})`; ctx.fillRect(px, 0, 1, Hd);
  }
}
function tStage(animate) {   // first lines → tokens → a 3 × 1,536 matrix (one row per chapter) → its length-weighted sum, scaled to length 1
  const ex = D.emb.examples[0], st = el("div", { class: "tstage" }), SW = 440, SH = 20;
  st.style.setProperty("--sw", SW + "px");
  const head = el("div", { class: "t-head" }), cap = el("span", { class: "t-label" }), replay = el("button", { class: "mini" }, "↻ Replay");
  if (ex.u) head.append(coverImg(ex.u));
  const hb = el("div"); hb.append(el("b", {}, ex.title), cap); head.append(hb, replay);
  const toks = el("div", { class: "t-tokens" });
  (D.emb.tokens || []).forEach((w, i) => { const sp = el("span", {}, w.trim() || "·"); sp.style.setProperty("--h", soft(TOKEN_COLORS[i % 5], "55")); toks.append(sp); });
  const mat = el("div", { class: "t-mat" });
  const strip = (cls) => { const cv = el("canvas", { class: "t-strip " + cls }), dpr = window.devicePixelRatio || 1; cv.width = SW * dpr; cv.height = SH * dpr; cv.style.width = SW + "px"; cv.style.height = SH + "px"; return cv; };
  const row = (left, cv, right, cls = "") => { const r = el("div", { class: "t-row " + cls }); r.append(el("span", { class: "t-w" }, left), cv, el("span", { class: "t-lab" }, right)); mat.append(r); return r; };
  const c1 = strip("");
  row("n₁ ×", c1, "e₁: chapter 1");
  const ghost = (txt) => { const cv = strip("ghost"), g = cv.getContext("2d"), dpr = window.devicePixelRatio || 1;
    g.scale(dpr, dpr); g.font = `600 12px ${CONFIG.sans}`; g.fillStyle = "#c9ccd0"; g.textAlign = "center"; g.fillText(txt, SW / 2, SH / 2 + 4); return cv; };
  row("n₂ ×", ghost("its 1,536 numbers were not saved"), "e₂: chapter 2", "ghost");
  row("n₃ ×", ghost("only chapter 1 and the average were kept"), "e₃: chapter 3", "ghost");
  mat.append(el("div", { class: "t-sum" }, "Σ  weighted by length, then scaled to length 1"));
  const res = strip("res");
  row("t =", res, `the serial (ch. 1 alone: cos ${ex.cos_c1_book.toFixed(2)})`, "res");
  const col = el("div", { class: "t-col" }); mat.append(col);   // one column = one of the 1,536 numbers, averaged over the 3 chapters
  st.append(head, el("div", { class: "t-text" }, D.emb.excerpt || ex.title), toks, mat);
  drawStrip(c1, ex.c1, 0.08);
  const caps = ["Chapter 1, first lines",
    `Tokens, as the model reads them (${ex.tokens.toLocaleString()} in chapters 1–3)`,
    "Each chapter embedded on its own: one row of 1,536 numbers",
    "Columns averaged over the chapters, weighted by length n"];
  const morph = (anim) => {   // the result row starts as chapter 1 and turns into the real average as chapters 2 and 3 come in
    cancelAnimationFrame(tFrame);
    const t0 = performance.now(), dur = anim ? 1200 : 0;
    const step = (now) => {
      const t = dur ? Math.min(1, (now - t0) / dur) : 1;
      drawStrip(res, ex.c1.map((a, i) => a + (ex.v[i] - a) * t), 0.08);
      if (t < 1) tFrame = requestAnimationFrame(step);
    };
    tFrame = requestAnimationFrame(step);
  };
  const go = (k) => { st.dataset.phase = "abcd"[k]; cap.textContent = caps[k]; if (k === 3) morph(animate); };
  const run = (anim) => {
    animate = anim; tTimers.forEach(clearTimeout); tTimers = [];
    if (!anim) return go(3);
    go(0); [1, 2, 3].forEach((k) => tTimers.push(setTimeout(() => go(k), k * CONFIG.tPhaseMs)));
  };
  replay.addEventListener("click", () => run(true));
  run(animate);
  return st;
}

// Diagrams drawn as SVG sketches (thin lines, icons, few labels); their groups fade in one after another
function sketch(W, H, x0 = 0, animate = true) {
  const wrap = el("div", { class: "cflow2-wrap" }), s = svg("svg", { class: "cflow2" + (animate ? "" : " done"), viewBox: `${x0} 0 ${W - x0} ${H}` }, wrap);
  let k = 0;
  const K = { wrap, s };
  K.grp = () => svg("g", { class: "cfx", style: `--k:${k++}` }, s);
  K.label = (g, x, y, txt, cls = "lab", anchor = "middle") => svg("text", { x, y, "text-anchor": anchor, class: cls }, g, txt);
  K.arrow = (g, x1, y1, x2, y2) => {
    svg("line", { x1, y1, x2, y2, class: "arw" }, g);
    const a = Math.atan2(y2 - y1, x2 - x1), h = 6;
    svg("path", { d: `M${x2},${y2}L${x2 - h * Math.cos(a - 0.5)},${y2 - h * Math.sin(a - 0.5)}L${x2 - h * Math.cos(a + 0.5)},${y2 - h * Math.sin(a + 0.5)}Z`, class: "arh" }, g);
  };
  K.sigmoid = (g, cx, cy) => { svg("circle", { cx, cy, r: 17, class: "ico" }, g); svg("path", { d: sigmoidD(cx, cy, 10, 7), class: "icoline" }, g); };   // logistic: a sigmoid in a circle
  K.pill = (g, cx, cy, txt, fill, w = 52) => { svg("rect", { x: cx - w / 2, y: cy - 13, width: w, height: 26, rx: 2, fill }, g); svg("text", { x: cx, y: cy + 5, "text-anchor": "middle", class: "pilltxt" }, g, txt); };
  K.chip = (g, x, y, key, w = 19) => { svg("rect", { x, y: y - 10, width: w, height: 20, rx: 2, fill: CONFIG.blocks[key] }, g); svg("text", { x: x + w / 2, y: y + 5, "text-anchor": "middle", class: "chiptxt" }, g, key); };
  K.net = (g, layers, x, y, dx, dy, cls = "") => {   // layers: node counts per column; every node linked to the next column
    const pos = layers.map((n, i) => Array.from({ length: n }, (_, j) => [x + i * dx, y + (j - (n - 1) / 2) * dy]));
    pos.forEach((col, i) => { if (i) for (const [x1, y1] of pos[i - 1]) for (const [x2, y2] of col) svg("line", { x1, y1, x2, y2, class: "nnl " + cls }, g); });
    pos.flat().forEach(([cx, cy]) => svg("circle", { cx, cy, r: 3, class: "nnn " + cls }, g));
    return pos;
  };
  K.start = () => {
    if (animate) requestAnimationFrame(() => requestAnimationFrame(() => { s.classList.add("run"); wrap.classList.add("run"); }));
    else wrap.classList.add("run");
    return wrap;
  };
  return K;
}

// Logistic regression: every number times its weight, added up, squeezed to 0–1, ranked within the quarter
function lrFlow(animate) {
  const K = sketch(680, 170, 0, animate), { label, arrow } = K, ys = [36, 82, 128], n = [nS(), 2, 1536];
  let g = K.grp();
  ["S", "A", "T"].forEach((key, j) => { K.chip(g, 6, ys[j], key, 22); label(g, 36, ys[j] + 4, `${n[j].toLocaleString()} numbers`, "lab", "start"); });
  label(g, 6, 162, "each standardized: mean 0, sd 1", "lab small", "start");
  g = K.grp(); ys.forEach((y) => arrow(g, 126, y, 236, 82 + (y - 82) * 0.3));
  label(g, 181, 16, `× its own weight β (${nInputs().toLocaleString()} weights)`);
  g = K.grp(); svg("circle", { cx: 256, cy: 82, r: 17, class: "ico" }, g); svg("text", { x: 256, y: 89, "text-anchor": "middle", class: "sigma" }, g, "Σ"); label(g, 256, 118, "add up");
  g = K.grp(); arrow(g, 274, 82, 330, 82); K.sigmoid(g, 350, 82); label(g, 350, 118, "squeeze to 0–1");
  g = K.grp(); arrow(g, 368, 82, 424, 82); K.pill(g, 452, 82, "p̂ᵢ", "#2a2c2f"); label(g, 452, 112, "chance of a hit");
  g = K.grp(); arrow(g, 480, 82, 534, 82);
  [0, 1, 2, 3, 4].forEach((j) => svg("rect", { x: 546, y: 58 + j * 10, width: 76 - j * 12, height: 6, rx: 1, fill: j < 2 ? CONFIG.colors.accent : "#3a3d41" }, g));
  label(g, 584, 126, "rank in the quarter:"); label(g, 584, 140, "the top K get the slots");
  return K.start();
}

// Gradient boosting: tree 1 + tree 2 + tree 3 + … (each fitted to what is still wrong), summed with a learning rate
function gbFlow(animate, eta, leaves) {
  const K = sketch(680, 176, 0, animate), { label, arrow } = K, yc = 84;
  let g = K.grp();
  ["S", "A", "T"].forEach((key, j) => K.chip(g, 4 + j * 24, yc, key, 21));
  label(g, 4, yc + 30, "S + A + T", "lab", "start"); label(g, 4, yc + 44, "T: 1,536 → 32 (PCA)", "lab small", "start");
  label(g, 330, 16, `each tree has at most ${leaves} leaves and is fitted to what the trees before it still get wrong`);
  const tree = (g, cx) => {   // three levels of splits → 8 leaves
    const lv = [[cx], [cx - 22, cx + 22], [cx - 33, cx - 11, cx + 11, cx + 33]], ty = [yc - 42, yc - 18, yc + 6];
    const leafX = Array.from({ length: 8 }, (_, i) => cx - 38.5 + i * 11), leafY = yc + 30;
    lv.forEach((xs, d) => xs.forEach((x, i) => {
      (d < 2 ? lv[d + 1].slice(2 * i, 2 * i + 2) : leafX.slice(2 * i, 2 * i + 2)).forEach((kx) => svg("line", { x1: x, y1: ty[d], x2: kx, y2: d < 2 ? ty[d + 1] : leafY, class: "tline" }, g));
      svg("circle", { cx: x, cy: ty[d], r: 3.2, class: "tnode" }, g);
    }));
    leafX.forEach((x) => svg("rect", { x: x - 4, y: leafY - 4, width: 8, height: 8, rx: 1, class: "tleaf" }, g));
  };
  g = K.grp(); arrow(g, 80, yc, 100, yc);
  [[145, "tree 1", "a first guess"], [275, "tree 2", "fixes tree 1's errors"], [405, "tree 3", "fixes what is left"]].forEach(([cx, a, b]) => {
    g = K.grp(); tree(g, cx); label(g, cx, yc + 56, a, "lab strong"); label(g, cx, yc + 70, b);
    svg("text", { x: cx + 65, y: yc + 7, "text-anchor": "middle", class: "op" }, g, "+");
  });
  g = K.grp(); svg("text", { x: 528, y: yc + 6, "text-anchor": "middle", class: "op" }, g, "···"); label(g, 528, yc + 56, "up to 300 trees", "lab strong"); label(g, 528, yc + 70, "until no gain");
  g = K.grp(); arrow(g, 560, yc, 592, yc); K.pill(g, 630, yc, "score", CONFIG.colors.accent, 66); label(g, 630, yc + 56, "votes added up,"); label(g, 630, yc + 70, `each × η = ${eta}`);
  return K.start();
}

// Neural network: frozen text encoder → T; with S and A into two hidden layers (ReLU); an inset shows ReLU itself
function mlpFlow(animate, drop, epochs) {
  const K = sketch(680, 266, 0, animate), { label, arrow } = K;
  let g = K.grp();   // frozen: chapter text → text-embedding-3-small → 1,536 numbers
  [0, 1, 2, 3].forEach((j) => svg("line", { x1: 6, x2: j === 3 ? 28 : 40, y1: 40 + j * 7, y2: 40 + j * 7, class: "txtline" }, g));
  label(g, 23, 80, "ch. 1–3");
  g = K.grp(); arrow(g, 46, 50, 74, 50); K.net(g, [3, 4, 4, 3], 86, 50, 18, 9, "frozen");
  label(g, 113, 12, "text-embedding-3-small", "lab strong"); label(g, 113, 25, "pretrained, frozen", "lab small");
  g = K.grp(); arrow(g, 146, 50, 176, 50);
  for (let j = 0; j < 20; j++) svg("rect", { x: 184 + j * 3, y: 40, width: 2, height: 20, fill: CONFIG.blocks.T, opacity: 0.3 + 0.7 * ((j * 7919) % 13) / 13 }, g);
  label(g, 214, 80, "T: 1,536 numbers");
  g = K.grp(); K.chip(g, 184, 120, "S", 21); label(g, 212, 124, `S: ${nS()}`, "lab", "start"); K.chip(g, 184, 156, "A", 21); label(g, 212, 160, "A: 2", "lab", "start");
  g = K.grp();   // trained here: inputs → 256 → 64 → 1
  [50, 120, 156].forEach((y) => arrow(g, 252, y, 298, 106 + (y - 106) * 0.35));
  K.net(g, [7, 6, 4, 1], 312, 106, 80, 18);
  svg("path", { d: "M304,40V34H560V40", class: "brk" }, g);
  label(g, 432, 26, `trained from scratch here: AdamW, ${epochs} epochs, 3 seeds averaged`, "lab strong");
  [[312, `${nInputs().toLocaleString()} inputs`, "standardized"], [392, "256 units", "ReLU", `dropout ${drop}`], [472, "64 units", "ReLU", `dropout ${drop}`], [552, "1 output"]].forEach(([x, ...t]) => {
    t.forEach((txt, i) => label(g, x, 188 + i * 14, txt, i ? "lab" : "lab strong"));
  });
  g = K.grp(); arrow(g, 558, 106, 584, 106); K.sigmoid(g, 603, 106); arrow(g, 621, 106, 634, 106); K.pill(g, 656, 106, "p̂ᵢ", "#2a2c2f", 42);
  g = K.grp();   // inset: what ReLU does
  const ox = 80, oy = 254;
  svg("line", { x1: 24, x2: 140, y1: oy, y2: oy, class: "axisl" }, g); svg("line", { x1: ox, x2: ox, y1: oy - 36, y2: oy + 5, class: "axisl" }, g);
  svg("path", { d: `M24,${oy}H${ox}L136,${oy - 32}`, class: "relu" }, g);
  label(g, 82, 192, "ReLU(x) = max(0, x)", "lab strong"); label(g, 82, 206, "below zero → 0, above passes", "lab small");
  g = K.grp();   // legend: frozen vs trained
  svg("line", { x1: 330, x2: 356, y1: 240, y2: 240, class: "nnl frozen" }, g); label(g, 362, 244, "frozen: not changed (no fine-tuning)", "lab", "start");
  svg("line", { x1: 330, x2: 356, y1: 256, y2: 256, class: "nnl" }, g); label(g, 362, 260, "trained here, from a random start", "lab", "start");
  return K.start();
}

// The two-stage cover model: main score + a cover adjustment (plays when the cover is opened by hand)
function coverFlow(animate) {   // two lanes (main model, cover) meeting at a "+"
  const byQ = pool(), cand = [...byQ.values()].flat().filter((i) => B.cover_status[i] === 0 && B.cover[i]).sort((a, c) => B.score.lr_SAT[c] - B.score.lr_SAT[a]);
  const C = CONFIG.blocks, W = 760, H = 196, y1 = 50, y2 = 142, ym = 96, X0 = -24, VW = W - X0;   // X0: a little room on the left for the labels
  const K = sketch(W, H, X0, animate), { label, arrow, sigmoid, pill } = K;
  // main lane: S A T → logistic regression → main score
  let g = K.grp(); ["S", "A", "T"].forEach((key, j) => K.chip(g, 14 + j * 22, y1, key));
  label(g, 46, y1 + 30, "tags, author, chapters");
  g = K.grp(); arrow(g, 86, y1, 150, y1); sigmoid(g, 172, y1); label(g, 172, y1 + 34, "logistic regression");
  g = K.grp(); arrow(g, 192, y1, 300, y1); pill(g, 328, y1, "sᵢ", "#2a2c2f"); label(g, 328, y1 + 30, "main score, held fixed");
  // cover lane: cover → CLIP (a neural network) → 768 numbers → cover head → adjustment
  g = K.grp();
  svg("rect", { x: 26, y: y2 - 24, width: 32, height: 48, fill: "#2a2c2f" }, g);   // the real cover is an <img> laid over this spot
  if (cand.length) {
    const im = coverImg(B.cover[cand[0]]); im.className = "cfx-cover";
    Object.assign(im.style, { left: `${((26 - X0) / VW) * 100}%`, top: `${((y2 - 24) / H) * 100}%`, width: `${(32 / VW) * 100}%`, height: `${(48 / H) * 100}%` });
    K.wrap.append(im);
  }
  label(g, 42, y2 + 40, "cover (before ch. 3)");
  g = K.grp(); arrow(g, 64, y2, 116, y2); K.net(g, [3, 5, 5, 2], 134, y2, 16, 9, "frozen"); label(g, 158, y2 + 40, "CLIP ViT-L/14 (frozen)");
  g = K.grp(); arrow(g, 192, y2, 232, y2);
  for (let j = 0; j < 24; j++) svg("rect", { x: 240 + j * 3, y: y2 - 10, width: 2, height: 20, fill: C.C, opacity: 0.25 + 0.75 * ((j * 7919) % 13) / 13 }, g);
  label(g, 276, y2 + 40, "zᵢ: 768 numbers");
  g = K.grp(); arrow(g, 318, y2, 372, y2); sigmoid(g, 394, y2); label(g, 394, y2 + 40, "cover head w (valid covers)");
  g = K.grp(); arrow(g, 414, y2, 476, y2); pill(g, 504, y2, "w·zᵢ", soft(C.C, "55")); label(g, 504, y2 + 40, "adjustment");
  // merge: + → final score
  g = K.grp(); arrow(g, 356, y1, 600, ym - 12); arrow(g, 530, y2, 600, ym + 12);
  svg("circle", { cx: 614, cy: ym, r: 14, class: "plus" }, g); svg("text", { x: 614, y: ym + 6, "text-anchor": "middle", class: "plustxt" }, g, "+");
  g = K.grp(); arrow(g, 628, ym, 680, ym); pill(g, 710, ym, "s′ᵢ", CONFIG.colors.accent); label(g, 710, ym + 30, "final score"); label(g, 652, ym + 54, "no valid cover: + 0", "lab small");
  return K.start();
}

// Tags and structure: everything a serial's page shows → drop what is only known later or edited later → S (plays when S is opened by hand)
const S_FIELDS = [   // [label, where it goes]: S / T / A = used in that block; later = only known after chapter 3; edited = 2025 version; unused
  ["Title", "unused"], ["Cover", "edited"], ["Tags", "S"], ["Blurb", "edited"], ["Followers", "later"], ["Content warnings", "S"],
  ["Views", "later"], ["Original or fan fiction", "S"], ["Ratings", "later"], ["Text of chapters 1–3", "T"], ["Favorites", "later"],
  ["Number of tags", "S"], ["Status (ongoing, completed)", "later"], ["Length of chapters 1–3", "S"], ["Number of chapters", "later"],
  ["Days from chapter 1 to 3", "S"], ["Author's earlier books", "A"],
];
const S_NOTES = { S: "→ S", T: "→ T", A: "→ A", later: "known only later", edited: "edited later", unused: "not used" };
let sTimers = [];
function sStage(animate) {
  const st = el("div", { class: "sstage" }), cap = el("span", { class: "t-label" }), replay = el("button", { class: "mini" }, "↻ Replay");
  const head = el("div", { class: "t-head" }), hb = el("div");
  hb.append(el("b", {}, "What a serial's page shows"), cap); head.append(hb, replay);
  const page = el("div", { class: "sfields" });
  for (const [label, k] of S_FIELDS) {
    const f = el("span", { class: "sf", "data-k": k }); f.append(el("b", {}, label), el("em", {}, S_NOTES[k]));
    if (CONFIG.blocks[k]) f.style.setProperty("--c", CONFIG.blocks[k]);
    page.append(f);
  }
  const sum = el("div", { class: "ssum" });
  sum.innerHTML = `<em>S</em> = ${D.features.S.tags.length} tags + ${D.features.S.warnings.length} warnings + 5 numbers = <b>${nS()} numbers</b>`;
  st.append(head, page, sum);
  const caps = ["Everything on a serial's Royal Road page", "Out: what is only known after chapter 3, the outcome itself",
                "Out: what authors edit once a serial takes off (blurb, cover), and the title", `In: what existed at chapter 3 becomes S, ${nS()} numbers`];
  const go = (k) => { st.className = "sstage p" + k; cap.textContent = caps[k]; };
  const run = (anim) => { sTimers.forEach(clearTimeout); sTimers = []; if (!anim) return go(3); go(0); [1, 2, 3].forEach((k) => sTimers.push(setTimeout(() => go(k), k * CONFIG.tPhaseMs))); };
  replay.addEventListener("click", () => run(true));
  run(animate);
  return st;
}

function renderDetail() {
  const box = $("detail"), f = S3.focus, H = D.hyper;
  const key = [f, f === "lr" ? modelOf() : "", f === "C" ? $("f-quarter").value + $("f-authors").value : ""].join("|");
  if (key === S3.dkey) return;   // redraw (and replay the animations) only when what the panel shows changes
  S3.dkey = key;
  const set = (title, sub) => { $("detail-title").textContent = title; $("detail-sub").textContent = sub; };
  box.replaceChildren();
  restart(box, "enter");   // content slides in when it changes
  setTimeout(() => fitFormulas(box), 0);
  if (f === "S") {
    set("S · Tags and structure", `${nS()} numbers per serial, all known when chapter 3 goes up`);
    box.append(sStage(true));
    section(box, `The ${D.features.S.tags.length} tags, colored by genre group (brighter = more common)`, chips(D.features.S.tags));
    section(box, `The ${D.features.S.warnings.length} content warnings`, chips(D.features.S.warnings.map((w) => [w, 0, "#9a4a3c"])));
  } else if (f === "A") {
    set("A · Author", "2 numbers, counted from the author's earlier serials (released before this one)");
    section(box, "The two numbers", chips([["First book: yes / no", 0, CONFIG.blocks.A], ["Number of earlier books", 0, CONFIG.blocks.A]]));
    const k = el("div", { class: "kpi-inline" });
    k.append(el("div", { class: "big accent" }, pct(D.features.first_time_share_2023, 0)), para("of 2023's new serials come from first-time authors. For them, A says nothing: the model has to read the story.", "note big-note"));
    section(box, "Why the story matters", k);
    section(box, "Left out on purpose", para("Whether an earlier book became a hit. Those followers are a 2025 snapshot, and a new hit pulls readers back to the author's older books, so it would leak the answer."));
  } else if (f === "T") {
    set("T · First three chapters", "Each serial's opening becomes 1,536 numbers (text-embedding-3-small)");
    box.append(tStage(true));
    section(box, "In formulas", fx("t"));
    const grids = el("div", { class: "vecs" }), E = D.emb, low = E.examples[1];
    const card = (cv, u, title, sub) => { const g = el("div"); const top = el("div", { class: "vtop" }); if (u) top.append(coverImg(u)); top.append(cv); g.append(top, el("b", {}, title), el("span", {}, sub)); return g; };
    grids.append(card(vecGrid(low.v, 0.08, 3), low.u, `${low.label}: ${low.title.length > 26 ? low.title.slice(0, 25) + "…" : low.title}`,
                      `rank ${low.rank.toLocaleString()} of ${low.n.toLocaleString()} · ${low.hit ? "became a hit" : "not a hit"} · match ${low.cos.toFixed(2)} (top pick ${E.examples[0].cos.toFixed(2)})`),
                 card(vecGrid(E.direction, 0.072, 3), "", "The hit direction", "average hit minus average serial, 2017–2022"));
    section(box, "Compare: orange above zero, blue below", grids, el("div", { id: "vec-readout", class: "note small" }, "\"Match\" = how closely a serial points the same way as past hits. Hover a grid to read a number."));
  } else if (f === "C") {
    const valid = B.cover_status.filter((s) => s === 0).length / B.cover_status.filter((s) => s >= 0).length;
    set("Cover · tried, not used", "Added on top of the main model, only for covers set before chapter 3");
    section(box, "How it works", coverFlow(true), fx("cover"));
    const lg = el("div", { class: "legend" });
    lg.innerHTML = '<span><i style="background:var(--other)"></i>main model</span><span><i style="background:var(--c-cover)"></i>+ cover</span>';
    const cb = el("div", { class: "chart-box", style: "height:150px;flex:none" });
    section(box, "Share of slots that went to hits (slots = true hits)", lg, cb);
    pairBars(cb, D.extras.cover_experiment.map((g) => ({ label: g.label, a: g.without, b: g.with })), { aLabel: "main model", bLabel: "+ cover", max: 0.6, bColor: CONFIG.blocks.C });
    box.append(para(`Covers changed after chapter 3 come with three times the hit rate (21% vs 7%, 2017–2022), so only covers set before chapter 3 are fair: ${D.extras.valid_cover_share_all ? pct(D.extras.valid_cover_share_all, 0) + " of all serials (2016–2024) had one, " : ""}${pct(valid, 0)} in 2023–24. The adjustment helps on those in 2023, not in 2024, and drifts upward on newer serials, so across all serials it changes nothing.`, "note small"));
  } else if (f === "rule") {
    set("Editors' rule · baseline", "What an editor could do without a model");
    section(box, "How it scores a serial", para("The average past hit rate of its tags (2017–2022). A LitRPG serial inherits LitRPG's hit rate, and so on. No text, no author.", "note big-note"));
    section(box, "Ties", para("Serials with the same tags get the same score; ties are broken by serial id."));
  } else if (f === "lr") {
    const m = ["lr_S", "lr_SA", "lr_SAT", "lr_T"].includes(modelOf()) ? modelOf() : "lr_SAT", Cs = Object.keys(H.lr.lr_SAT.val_auc_by_C);
    set("Logistic regression · our model", "Every number gets one weight; the weighted sum becomes a chance of a hit");
    section(box, "How it scores a serial", lrFlow(true));
    const cb = el("div", { class: "chart-box", style: "height:200px;flex:none" });
    section(box, "Tuned on 2022: C, how freely the weights may grow (AUC on 2022, higher is better)", cb);
    lineChart(cb, Cs, ["lr_S", "lr_SA", "lr_T", "lr_SAT"].map((id) => ({ label: D.short[id], values: Cs.map((c) => H.lr[id].val_auc_by_C[c]),
      color: id === m ? CONFIG.colors.accent : "#5c5f63", dim: id !== m, glow: id === m, mark: id === m ? Cs.findIndex((c) => Number(c) === H.lr[id].C) : undefined })),
      { yMin: 0.78, yMax: 0.9, fmt: (v) => v.toFixed(2), xLabel: "C, log scale (smaller C keeps the weights smaller)" });
    section(box, "Why that matters", para(`S+A+T has ${nInputs().toLocaleString()} weights, 1,536 of them for the text, so a small C (strong penalty) works best. Refit on 2017–2022 with the chosen C, scored once on 2023.`));
  } else if (f === "gb") {
    const h = H.gb, [eta, leaves] = h.chosen;
    set("Gradient boosting", "Small decision trees added one after another, each correcting the ones before it (HistGradientBoosting)");
    section(box, "How it scores a serial", gbFlow(true, eta, leaves));
    section(box, "Tuned on 2022: learning rate × leaves per tree (AUC on 2022)",
      heatmap([0.03, 0.1], [8, 31], (r, c) => ({ v: h.val_auc_by_param[`(${r}, ${c})`], chosen: r === eta && c === leaves }), { rowName: "learning rate η", colName: "max leaves per tree" }));
    section(box, "Near ties", para("The four settings are within 0.007 of each other, so tiny numerical differences between machines can change the winner (2023 result: 0.465 in this export, 0.476 in the repo's results file)."));
  } else if (f === "mlp") {
    const h = H.mlp, [drop, wd, ep] = h.chosen;
    set("Neural network (MLP)", "Not fine-tuned: a small network trained from scratch on top of frozen text vectors");
    section(box, "How it scores a serial", mlpFlow(true, drop, ep));
    section(box, "Tuned on 2022: dropout × weight decay (AUC on 2022; epochs by early stopping)",
      heatmap([0.2, 0.5], [0.0001, 0.01], (r, c) => { const v = h.val_auc_by_param[`(${r}, ${c})`]; return { v: v[0], sub: `${v[1]} epochs`, chosen: r === drop && c === wd }; }, { rowName: "dropout", colName: "weight decay" }));
    section(box, "Training", para(`AdamW, learning rate 0.001, batches of 256, up to 40 epochs with early stopping on 2022 (patience 5); then refit on 2017–2022 for ${ep} epochs, average of 3 seeds.`));
  }
}

// ---------- screen 3: the model's picks (mosaic of every test serial + a shelf of picked covers) ----------
const M = { layer: "both", prev: null, key: "" };
const STATE = { dim: 0, picked: 1, hit: 2, missed: 3 };  // per-square state for the current layer
function squareState(i, isPicked) {
  const isHit = B.y[i] === 1;
  if (M.layer === "picks") return isPicked ? STATE.picked : STATE.dim;
  if (M.layer === "hits") return isHit ? STATE.hit : STATE.dim;
  return isPicked ? (isHit ? STATE.hit : STATE.picked) : isHit ? STATE.missed : STATE.dim;
}
function renderPicks(byQ, k, model) {
  const picks = pick(byQ, model, k), mine = hits(picks), rule = hits(pick(byQ, "rule_hot_genre", k)), nq = byQ.size;
  const who = $("f-authors").value === "first" ? ", first-time authors only" : "";
  $("kpi-rule").textContent = (rule.h / nq).toFixed(0);
  $("kpi-model").textContent = (mine.h / nq).toFixed(0);
  $("kpi-mult").textContent = (mine.h / rule.h).toFixed(1) + "×";
  $("kpi-rule-cap").textContent = `Editors' rule: hits per quarter${who}`;
  $("kpi-model-cap").textContent = `${D.short[model]}: hits per quarter in ${k} slots${who}`;
  $("picks-cap").textContent = `Test year, never seen in training · ${quarterLabel(byQ)} · ${whoLabel()} · ${mine.h} of ${mine.n} picks became hits (${pct(mine.share, 0)}; random ${pct(baseRate(byQ), 0)})`;
  renderMosaic(byQ, picks);
  renderCovers(picks);
}
function renderMosaic(byQ, picks) {
  const box = $("mosaic"), W = box.clientWidth, Hbox = box.clientHeight;
  const order = [];                          // quarters one after another, serials by id inside a quarter
  for (const [q, idx] of byQ) order.push({ q, ids: [...idx].sort((a, c) => B.id[a] - B.id[c]) });
  const labelH = 16, { maxSize, minSize, gap } = CONFIG.mosaic;
  let size = maxSize;                        // largest square size that fits the panel
  while (size > minSize) {
    const cols = Math.floor(W / (size + gap));
    if (order.reduce((a, o) => a + Math.ceil(o.ids.length / cols) * (size + gap) + labelH, 0) <= Hbox) break;
    size--;
  }
  const cols = Math.floor(W / (size + gap)), pickedSet = new Map();
  for (const ix of picks.values()) ix.forEach((i, r) => pickedSet.set(i, r + 1));
  const cells = []; let y = 0;
  for (const o of order) {
    cells.push({ label: D.quarters[o.q], y });
    y += labelH;
    o.ids.forEach((i, j) => cells.push({ i, x: (j % cols) * (size + gap), y: y + Math.floor(j / cols) * (size + gap), st: squareState(i, pickedSet.has(i)) }));
    y += Math.ceil(o.ids.length / cols) * (size + gap);
  }
  const key = [...byQ.keys()].join(",") + $("f-authors").value + W;
  const prev = M.key === key ? M.prev : null;  // fade only when the same serials are on screen
  M.key = key; M.prev = new Map(cells.filter((c) => c.i !== undefined).map((c) => [c.i, c.st]));
  M.cells = cells; M.size = size; M.picked = pickedSet;
  let cv = box.querySelector("canvas");
  if (!cv) {
    cv = el("canvas"); box.append(cv);
    cv.addEventListener("mousemove", mosaicHover); cv.addEventListener("click", mosaicClick); cv.addEventListener("mouseleave", () => $("peek").classList.remove("on"));
  }
  const dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = y * dpr; cv.style.height = y + "px";
  const ctx = cv.getContext("2d"), t0 = performance.now();
  const draw = (now) => {
    const p = prev ? Math.min(1, (now - t0) / CONFIG.mosaic.fadeMs) : 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, y);
    ctx.font = `12px ${CONFIG.sans}`;
    for (const c of cells) {
      if (c.label) { ctx.globalAlpha = 1; ctx.fillStyle = CONFIG.colors.muted; ctx.fillText(c.label, 0, c.y + 11); continue; }
      const old = prev ? prev.get(c.i) : c.st;
      if (old !== undefined && old !== c.st && p < 1) { paint(ctx, c, old, 1 - p, size); paint(ctx, c, c.st, p, size); }
      else paint(ctx, c, c.st, 1, size);
    }
    ctx.globalAlpha = 1;
    if (p < 1) requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
  const lg = $("mosaic-legend"); lg.replaceChildren();
  const acc = CONFIG.colors.accent, pk = CONFIG.colors.picked, dim = "#2a2c2f";   // background squares keep a faint genre tint
  const items = { picks: [[pk, 1, "picked by the model"], [dim, 1, "not picked"]],
                  hits: [[acc, 1, "became a hit"], [dim, 1, "did not"]],
                  both: [[acc, 1, "picked and became a hit"], [pk, 1, "picked, not a hit"], [acc, CONFIG.mosaic.missedAlpha, "hit the model missed"], [dim, 1, "neither"]] }[M.layer];
  for (const [bg, op, t] of items) { const i = el("i"); i.style.background = bg; i.style.opacity = op; const sp = el("span"); sp.append(i, t); lg.append(sp); }
}
function paint(ctx, c, st, alpha, size) {
  const col = B.color[c.i];
  const [fill, a] = { [STATE.dim]: [col, CONFIG.mosaic.dimAlpha], [STATE.picked]: [CONFIG.colors.picked, 1],
                      [STATE.hit]: [CONFIG.colors.accent, 1], [STATE.missed]: [CONFIG.colors.accent, CONFIG.mosaic.missedAlpha] }[st];
  ctx.globalAlpha = alpha * a; ctx.fillStyle = fill;
  ctx.shadowColor = fill; ctx.shadowBlur = st === STATE.hit || st === STATE.picked ? 14 : 0;   // lit squares glow
  ctx.fillRect(c.x, c.y, size, size);
  ctx.shadowBlur = 0;
}
function cellAt(ev) {   // the screens are scaled, so divide mouse offsets by the scale
  const r = ev.target.getBoundingClientRect(), z = r.width / ev.target.offsetWidth, x = (ev.clientX - r.left) / z, y = (ev.clientY - r.top) / z, s = M.size;
  return M.cells.find((c) => c.i !== undefined && x >= c.x && x < c.x + s && y >= c.y && y < c.y + s);
}
function mosaicHover(ev) {   // a floating cover card follows the mouse
  const c = cellAt(ev), pk = $("peek");
  if (!c) { pk.classList.remove("on"); return; }
  const i = c.i, r = M.picked.get(i), status = `${r ? "picked #" + r : "not picked"} · ${B.y[i] ? "became a hit" : "not a hit"}${B.first[i] ? " · first book" : ""}`;
  $("hover").textContent = `${B.title[i]} · ${status}`;
  if (pk.dataset.i !== String(i)) {
    pk.dataset.i = i; pk.replaceChildren();
    const im = el("div", { class: "pimg" }); if (B.cover[i]) im.append(coverImg(B.cover[i])); else im.textContent = B.title[i];
    pk.append(im, el("b", {}, B.title[i]), el("span", {}, status));
    pk.classList.toggle("hit", B.y[i] === 1);
  }
  pk.style.left = Math.min(ev.clientX + 18, innerWidth - 200) + "px"; pk.style.top = Math.min(ev.clientY + 18, innerHeight - 260) + "px";
  pk.classList.add("on");
}
function mosaicClick(ev) { const c = cellAt(ev); if (c) window.open("https://www.royalroad.com/fiction/" + B.id[c.i], "_blank", "noopener"); }
function setLayer(l) {
  M.layer = l;
  document.querySelectorAll("[data-layer]").forEach((x) => x.classList.toggle("on", x.dataset.layer === l));
  const byQ = pool(); renderMosaic(byQ, pick(byQ, $("f-model").value, slots()));
}
function setDim(on) { $("covers").classList.toggle("dim", on); $("dim").classList.toggle("on", on); }

let showAllCovers = false;
function coverTile(i, rank) {
  const a = el("a", { class: "cv " + (B.y[i] ? "hit" : "miss"), href: "https://www.royalroad.com/fiction/" + B.id[i], target: "_blank", rel: "noopener", title: `#${rank} ${B.title[i]}` });
  a.append(el("div", { class: "nt" }, B.title[i]));   // title underneath; the cover covers it when it loads
  if (B.cover[i]) a.append(coverImg(B.cover[i], B.title[i]));
  a.append(el("span", { class: "r" }, "#" + rank));
  if (B.y[i]) a.append(el("span", { class: "h" }));
  return a;
}
function renderCovers(picks) {   // shelves of overlapping covers, highest rank in front, sized to fit the card
  const box = $("covers"); box.replaceChildren();
  const perShelf = Math.max(1, Math.floor(4 / picks.size)), nShelves = Math.min(4, picks.size * perShelf);
  const shelfH = Math.min(196, (box.clientHeight - 16) / nShelves), ch = shelfH - 30, cw = Math.min(CONFIG.coverW, ch / 1.5), W = box.clientWidth;
  box.style.setProperty("--sh", shelfH + "px"); box.style.setProperty("--cw", cw + "px"); box.style.setProperty("--ch", cw * 1.5 + "px");
  for (const [q, ix] of picks) {
    const list = ix.slice(0, showAllCovers ? CONFIG.coversMax : CONFIG.coversPerQuarter), size = Math.ceil(list.length / perShelf);
    for (let s = 0; s < perShelf; s++) {
      const part = list.slice(s * size, (s + 1) * size); if (!part.length) continue;
      const shelf = el("div", { class: "shelf" });
      if (s === 0) shelf.append(el("div", { class: "qlab" }, `${D.quarters[q]} · ${ix.filter((i) => B.y[i]).length} of ${ix.length} picks became hits`));
      const step = part.length > 1 ? Math.min(cw + 10, (W - cw - 30) / (part.length - 1)) : 0;
      part.forEach((i, j) => { const a = coverTile(i, s * size + j + 1); a.style.left = j * step + "px"; a.style.zIndex = part.length - j; shelf.append(a); });
      box.append(shelf);
    }
  }
  const all = [...picks.values()].flat(), nHit = all.filter((i) => B.y[i]).length;
  const many = [...picks.values()].some((ix) => ix.length > CONFIG.coversMax);
  $("covers-all").textContent = showAllCovers ? "Show fewer" : many ? `Show the top ${CONFIG.coversMax} per quarter` : `Show all ${all.length} picks`;
  $("hit-share").replaceChildren(el("b", {}, `${nHit} hits`), el("i", { style: `--p:${nHit / all.length}` }), el("span", {}, `${all.length - nHit} misses`));
}

// ---------- screen 4: what else we checked ----------
function playChecks() {   // arriving: the three charts play at the same time
  animateIn($("leak-chart")); animateIn($("drift-chart")); renderTaste(true);
}
function renderChecks(byQ) {
  const all = [...byQ.values()].flat();
  const rate = (f) => { const g = all.filter(f); return g.length ? g.reduce((a, i) => a + B.y[i], 0) / g.length : NaN; };
  const groups = [   // a = the rest, b = signs of an edit after chapter 3
    { label: "Blurb", a: rate((i) => B.blurb_flag[i] === 0), b: rate((i) => B.blurb_flag[i] === 1) },
    { label: "Cover", a: rate((i) => B.cover_status[i] === 0), b: rate((i) => B.cover_status[i] === 1) },
  ].filter((g) => !isNaN(g.a) && !isNaN(g.b));   // no cover status in the export: the cover pair is left out
  $("leak-sub").textContent = `Blurbs and covers are 2025 versions. Hit rate · ${quarterLabel(byQ)} · ${whoLabel()}.`;
  pairBars($("leak-chart"), groups, { aLabel: "the rest", bLabel: "edited after chapter 3", max: Math.max(...groups.map((g) => g.b)) * 1.15 });
  const dr = D.extras.drift_2023;
  lineChart($("drift-chart"), dr.S.gap.map((g) => `${g}y`), [
    { label: `Tags ${dr.S.per_year.toFixed(3)}/yr`, values: dr.S.auc, color: CONFIG.colors.rule },
    { label: `Text ${dr.T.per_year.toFixed(3)}/yr`, values: dr.T.auc, color: CONFIG.colors.accent, glow: true },
  ], { yMin: 0.7, yMax: 0.85, xLabel: "how much older the training data is (test year 2023)" });
  renderTaste();
}

// Taste over time: tag lift, or the story map (2D views of the 1,536-number vectors)
let tasteView = "pca";
const lighten = (hex, f = 0.35) => "#" + [1, 3, 5].map((i) => Math.round(parseInt(hex.slice(i, i + 2), 16) * (1 - f) + 255 * f).toString(16).padStart(2, "0")).join("");
function renderTaste(animate = false) {
  const box = $("taste-chart"), M2 = D.story_map, lg = $("taste-legend");
  document.querySelectorAll("[data-taste]").forEach((b) => b.classList.toggle("on", b.dataset.taste === tasteView));
  const swatch = (bg, text, line) => { const i = el("i"); i.style.background = bg; if (line) i.style.height = "3px"; const sp = el("span"); sp.append(i, text); return sp; };
  lg.replaceChildren();
  $("fx-lift").style.display = tasteView === "tags" ? "" : "none";
  $("taste-take").textContent = `Taste mostly moves with the market: from 2017 to 2024 the average hit shifted the same way as the average new serial (cosine ${M2.shift_cos_1536.toFixed(2)}).`;
  if (tasteView === "tags") {
    $("taste-sub").textContent = "How much more often a tag becomes a hit than average (lift). Gray dot 2017–19, colored dot 2022–24.";
    $("taste-cap").textContent = "LitRPG and Xianxia rose, Romance and Harem fell.";
    dumbbell(box, D.extras.tag_movers.map((r) => ({ label: r.tag, a: r.early, b: r.late })), { min: 0.3, max: 2.7, aLabel: "2017–19", bLabel: "2022–24" });
    if (animate) animateIn(box);
    return;
  }
  const pca = tasteView === "pca";
  $("taste-sub").textContent = pca ? "Each serial's 1,536 numbers squeezed to 2 (PCA). 4,000 serials, 2017–2024, colored by genre."
                                   : "UMAP: similar stories sit together; colors are genres. Where the hits sat, early vs late.";
  $("taste-cap").textContent = pca ? `PCA keeps ${pct(M2.pca_explained_2d, 0)} of the information, so the map is only a sketch; the similarity above is measured in all 1,536 numbers.`
                                   : "UMAP is good at clusters, not at distances; trends are measured in the full 1,536 numbers.";
  if (pca) lg.append(swatch(CONFIG.colors.light, "all serials, yearly average", true), swatch(CONFIG.colors.accent, "hits, yearly average", true));
  else lg.append(swatch(CONFIG.colors.light, "hits 2017–19"), swatch(CONFIG.colors.accent, "hits 2022–24"));
  storyMap(box, pca, animate);
}
let mapFrame = 0;
function storyMap(box, pca, animate) {   // canvas; with animate (PCA only), a 2.4-second intro: two raw numbers → the two PCA directions
  box.replaceChildren(); cancelAnimationFrame(mapFrame);
  const M2 = D.story_map, pts = pca ? M2.pca : M2.umap, ex = D.emb.examples[0], [d1, d2] = M2.raw_dims;
  const W = box.clientWidth, Hh = box.clientHeight, dpr = window.devicePixelRatio || 1, cv = el("canvas");
  cv.width = W * dpr; cv.height = Hh * dpr; cv.style.width = W + "px"; cv.style.height = Hh + "px"; box.append(cv);
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
  const q = (arr, p) => [...arr].sort((a, b) => a - b)[Math.floor(p * (arr.length - 1))];
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]), lo = 0.01, top = pca ? 44 : 0;
  const [x0, x1, y0, y1] = [q(xs, lo), q(xs, 1 - lo), q(ys, lo), q(ys, 1 - lo)], pad = 16;
  const k = Math.min((W - 2 * pad) / (x1 - x0), (Hh - top - 2 * pad) / (y1 - y0));   // same scale on both axes keeps the cloud's shape
  const cx = W / 2, cy = top + (Hh - top) / 2, mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  const X = (x) => cx + (x - mx) * k, Y = (y) => cy - (y - my) * k;
  const kr = Math.min(W - 2 * pad, Hh - top - 2 * pad) / 6;   // raw numbers are standardized: ±3 sd fills the box
  const RX = (z) => cx + z * kr, RY = (z) => cy - z * kr;
  const colors = M2.color.map((c) => lighten(c, 0.3));
  const ease = (t) => { t = Math.min(1, Math.max(0, t)); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
  const arrow = (ax, ay, bx, by, col) => {   // segment with a small arrowhead at its end
    ctx.strokeStyle = col; ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    const a = Math.atan2(by - ay, bx - ax), h = 7;
    ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx - h * Math.cos(a - 0.45), by - h * Math.sin(a - 0.45)); ctx.lineTo(bx - h * Math.cos(a + 0.45), by - h * Math.sin(a + 0.45)); ctx.fill();
  };
  const label = (txt, x, y, col = CONFIG.colors.text, align = "center", size = 14) => {   // text with a dark outline, readable over the dots
    ctx.font = `700 ${size}px ${CONFIG.sans}`; ctx.textAlign = align; ctx.lineJoin = "round";
    ctx.lineWidth = 5; ctx.strokeStyle = "rgba(0, 0, 0, .9)"; ctx.strokeText(txt, x, y);
    ctx.fillStyle = col; ctx.fillText(txt, x, y); ctx.textAlign = "left";
  };
  const wrap = (txt, y) => {   // centered caption, wrapped to the canvas width
    ctx.font = `600 14px ${CONFIG.sans}`; const lines = [""];
    txt.split(" ").forEach((w) => { const t = lines[lines.length - 1] ? lines[lines.length - 1] + " " + w : w; if (ctx.measureText(t).width > W - 24) lines.push(w); else lines[lines.length - 1] = t; });
    lines.forEach((l, i) => label(l, cx, y + i * 18));
  };
  const dur = animate && pca ? 2400 : 0, t0 = performance.now();
  const frame = (now) => {
    const t = dur ? Math.min(1, (now - t0) / dur) : 1, ms = t * 2400;
    ctx.clearRect(0, 0, W, Hh);
    if (pca && ms < 900) {   // 0–0.9 s: one serial's 1,536 numbers; two of them are picked out
      const a = Math.min(1, ms / 250), bw = W - 2 * pad, bx = pad, by = top + 6;
      ex.v.forEach((v, i) => { const s = Math.min(1, Math.abs(v) / 0.08); ctx.globalAlpha = a * (i === d1 || i === d2 ? 1 : 0.55); ctx.fillStyle = `rgb(${(v > 0 ? VEC_POS : VEC_NEG).map((c, j) => Math.round(VEC_BASE[j] + (c - VEC_BASE[j]) * s))})`; ctx.fillRect(bx + (i / 1536) * bw, by, Math.max(0.7, bw / 1536), 30); });
      ctx.globalAlpha = 1;
      if (ms > 350) [d1, d2].forEach((d, j) => { const x = bx + (d / 1536) * bw; ctx.fillStyle = CONFIG.colors.text; ctx.shadowColor = CONFIG.colors.text; ctx.shadowBlur = 8; ctx.fillRect(x - 1.5, by - 4, 3, 38); ctx.shadowBlur = 0; label(`number ${d + 1}`, x, by + 52, CONFIG.colors.text, "center", 13); });
    }
    let txt;
    if (pca) txt = ms < 900 ? `One serial = 1,536 numbers. Pick two of them…` : ms < 1300 ? `…and each serial is a dot: number ${d1 + 1} across, ${d2 + 1} up` :
      ms < 2000 ? `PCA mixes all 1,536 into the two directions that spread serials most` : "";
    if (txt) wrap(txt, 14);
    const appear = pca ? Math.min(1, Math.max(0, (ms - 700) / 400)) : 1, mo = pca ? ease((ms - 1300) / 700) : 1;   // dots appear at the two raw numbers, then glide to PCA
    if (appear > 0) {
      pts.forEach((p, i) => {
        const r = M2.raw[i], x = pca ? RX(r[0]) + (X(p[0]) - RX(r[0])) * mo : X(p[0]), y = pca ? RY(r[1]) + (Y(p[1]) - RY(r[1])) * mo : Y(p[1]);
        ctx.globalAlpha = (pca && ms >= 2000 ? 0.4 : 0.65) * appear; ctx.fillStyle = colors[i]; ctx.fillRect(x - 1.6, y - 1.6, 3.2, 3.2);
      });
      ctx.globalAlpha = 1;
    }
    if (pca && ms >= 2000) {   // last 0.4 s: the yearly averages, joined by arrows
      const show = Math.min(1, (ms - 2000) / 400);
      for (const [tr, col] of [[M2.traj_all, CONFIG.colors.light], [M2.traj_hits, CONFIG.colors.accent]]) {
        const yrs = Object.keys(tr), P = yrs.map((y) => [X(tr[y][0]), Y(tr[y][1])]), n = Math.max(1, Math.round(show * (P.length - 1)));
        ctx.shadowColor = col; ctx.shadowBlur = 16; ctx.lineWidth = 2.5;
        for (let s = 0; s < n; s++) arrow(P[s][0], P[s][1], P[s + 1][0], P[s + 1][1], col);
        ctx.shadowBlur = 0;
        P.slice(0, n + 1).forEach(([x, y]) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 7); ctx.fill(); });
        label(yrs[0], P[0][0] + 10, P[0][1] - 10, col, "left", 18);
        if (show >= 1) label(yrs[yrs.length - 1], P[P.length - 1][0] - 10, P[P.length - 1][1] - 10, col, "right", 18);
      }
      if (show >= 1) { ctx.strokeStyle = CONFIG.colors.text; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(ex.p[0]), Y(ex.p[1]), 6, 0, 7); ctx.stroke(); label(ex.title, X(ex.p[0]), Y(ex.p[1]) + 26, CONFIG.colors.text, "center", 16); }
      if (show >= 1) label(`PC1 and PC2 keep ${pct(M2.pca_explained_2d, 0)} of the information`, cx, 18, CONFIG.colors.muted, "center", 13);
    }
    if (!pca) {
      pts.forEach((p, i) => {
        if (!M2.y[i]) return;
        const early = M2.year[i] <= 2019, late = M2.year[i] >= 2022;
        if (!early && !late) return;
        ctx.fillStyle = early ? CONFIG.colors.light : CONFIG.colors.accent; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 2.8, 0, 7); ctx.fill();
      });
    }
    if (t < 1) mapFrame = requestAnimationFrame(frame);
  };
  mapFrame = requestAnimationFrame(frame);
}

// ---------- screen 5: business value (a scenario: sliders → ad revenue + editor time, our model minus the editors' rule) ----------
const BIZ = [   // the sliders; tex + note open under each one
  { id: "k", label: "Slots per quarter", short: "slots per quarter", sym: "K", min: 0, max: K_STEPS.length - 1, step: 1, value: posOfK(CONFIG.slots), real: kAt, fmt: (v) => v.toLocaleString(),
    tex: String.raw`\mathrm{Top}_K(q) = \text{the } K \text{ highest-scored new serials of quarter } q`, note: (b) => `${b.k} slots are ${pct(b.k / b.perQ, 1)} of the ${Math.round(b.perQ).toLocaleString()} new serials a quarter. Same slider as "slots per quarter" in the top bar.` },
  { id: "lambda", label: "Share of a pick's views the slot brings", short: "views from the slot", sym: "λ", min: 0.01, max: 0.5, step: 0.01, value: 0.10, fmt: (v) => pct(v, 0),
    tex: String.raw`\text{views from the slot} = \lambda \times \text{views of the pick}`, note: () => "Not knowable from this data: the picks would partly have been read anyway. An A/B pilot would measure it." },
  { id: "cpm", label: "Ad revenue per 1,000 views", short: "per 1,000 views", sym: "CPM", min: 0.5, max: 10, step: 0.5, value: 3, fmt: (v) => "$" + v.toFixed(2),
    tex: String.raw`\text{revenue per view} = \text{CPM} / 1000`, note: () => "What the platform earns from 1,000 page views." },
  { id: "short", label: "Serials editors screen with the model", short: "editors screen", sym: "S", min: 50, max: 1000, step: 10, value: 200, fmt: (v) => String(v),
    tex: String.raw`\text{hours saved} = \sum_q \big(N_q - S\big) \times m / 60`,
    note: (b) => `Without the model, editors would screen all ${Math.round(b.perQ).toLocaleString()} new serials a quarter (N); with it, only the model's top S. The deck uses 200.` },
  { id: "mins", label: "Minutes to screen one serial", short: "per serial", sym: "m", min: 5, max: 60, step: 5, value: 10, fmt: (v) => v + " min",
    tex: String.raw`\text{hours} = \text{serials} \times m / 60`, note: () => "Time for an editor to look at one new serial. The deck assumes 10 minutes." },
  { id: "wage", label: "Editor cost per hour", short: "editor hour", sym: "w", min: 15, max: 120, step: 5, value: 30, fmt: (v) => "$" + v,
    tex: String.raw`\text{cost} = \text{hours} \times w`, note: () => "Salary and overhead of one editor hour. The deck assumes $30." },
];
const bizVal = (id) => Number($(`biz-${id}`).value);
const bizA = () => ({ lambda: bizVal("lambda"), cpm: bizVal("cpm"), short: bizVal("short"), mins: bizVal("mins"), wage: bizVal("wage") });   // the current assumptions
const bizReal = (b, raw) => (b.real ? b.real(raw) : raw);   // slider position → value (only the slots slider differs)
let sensVar = "k";
function bizInit() {
  const box = $("biz-sliders");
  for (const b of BIZ) {
    const row = el("div", { class: "bz" }), top = el("div", { class: "bz-top" });
    top.append(el("b", {}, b.label), el("span", { class: "bz-sym" }, b.sym), el("span", { class: "bz-val", id: `biz-${b.id}-val` }, b.fmt(bizReal(b, b.value))));
    const input = el("input", { type: "range", id: `biz-${b.id}`, min: b.min, max: b.max, step: b.step, value: b.value });
    const det = el("details", { class: "bz-more" }), sum = el("summary", {}, "formula");
    det.append(sum, el("div", { class: "formula small", id: `biz-${b.id}-fx` }), el("p", { class: "note", id: `biz-${b.id}-note` }));
    det.addEventListener("toggle", () => fitFormulas(det));
    row.append(top, input, det); box.append(row);
    let queued = false;
    input.addEventListener("input", () => {
      $(`biz-${b.id}-val`).textContent = b.fmt(bizReal(b, Number(input.value)));
      if (b.id === "k") { $("f-k").value = input.value; $("f-k-val").textContent = slots(); }   // one slots slider for the whole page
      if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; b.id === "k" ? render() : renderBiz(pool()); }); }
    });
    input.addEventListener("change", () => input.blur());
    const vb = el("button", { "data-sens": b.id, class: "schip" });
    vb.addEventListener("click", () => { sensVar = b.id; renderSens(pool()); });
    $("sens-vars").append(vb);
  }
  $("sens-slider").addEventListener("input", (e) => { const src = $(`biz-${sensVar}`); src.value = e.target.value; src.dispatchEvent(new Event("input")); });
  $("sens-slider").addEventListener("change", (e) => e.target.blur());
  const sens = (open) => {   // the drawer takes the sliders' place; its own button brings them back
    $("biz-drawer").classList.toggle("open", open); $("biz-drawer").parentElement.classList.toggle("drawer-open", open);
    setTimeout(() => { renderSens(pool()); fitFormulas($("biz")); }, 520);   // after the drawer has opened
  };
  $("sens-open").addEventListener("click", () => sens(true));
  $("sens-close").addEventListener("click", () => sens(false));
}
let bizCache = { key: "" };
function bizPrep(byQ) {   // per quarter: both rankings with running sums of views and hits, so any K is cheap
  const key = [...byQ.keys()].join(",") + $("f-authors").value;
  if (bizCache.key === key) return bizCache;
  const s = B.score.lr_SAT, r = B.score.rule_hot_genre, run = (ix, f) => { const a = [0]; ix.forEach((i, j) => a.push(a[j] + f(i))); return a; };
  const qs = [...byQ.values()].map((idx) => {
    const ms = [...idx].sort((a, c) => s[c] - s[a] || tie(a, c)), rs = [...idx].sort((a, c) => r[c] - r[a] || tie(a, c));
    return { n: idx.length, mV: run(ms, (i) => B.views[i]), rV: run(rs, (i) => B.views[i]) };
  });
  bizCache = { key, qs };
  return bizCache;
}
function bizNumbers(byQ, k) {   // views of each side's top K per quarter, and how many new serials there are to screen
  const out = { vm: 0, vr: 0, n: 0, nq: byQ.size, k };
  for (const q of bizPrep(byQ).qs) { const kk = Math.min(k, q.n); out.vm += q.mV[kk]; out.vr += q.rV[kk]; out.n += q.n; }
  out.perQ = out.n / out.nq;
  return out;
}
function bizMoney(bz, a) {   // ad revenue: model minus rule (the deck's headline); editor time, as in the deck: screen the model's top S, not every new serial
  const adM = bz.vm * a.lambda * a.cpm / 1000, adR = bz.vr * a.lambda * a.cpm / 1000;
  const screened = Math.min(bz.n, bz.nq * a.short), hours = (bz.n - screened) * a.mins / 60;
  return { adM, adR, ad: adM - adR, screened, hours, costAll: bz.n * a.mins / 60 * a.wage, costShort: screened * a.mins / 60 * a.wage, labor: hours * a.wage };
}
const roundEven = (x) => { const f = Math.floor(x), d = x - f; return d > 0.5 || (d === 0.5 && f % 2) ? f + 1 : f; };   // rounds .5 to even, like Python (so hours match the deck)
const usd = (v) => (v < 0 ? "−$" : "$") + Math.abs(Math.round(v)).toLocaleString();
function renderBiz(byQ) {
  if (!B.views) { $("biz-cap").textContent = "This data.js has no view counts: rerun modeling/05_demo_data.py."; return; }   // an older data.js must not break the rest of the page
  const k = slots(), bz = bizNumbers(byQ, k), a = bizA(), mo = bizMoney(bz, a);
  $("biz-k").value = posOfK(k); $("biz-k-val").textContent = k.toLocaleString();
  const per = bz.nq === 4 ? "per year" : bz.nq === 1 ? "per quarter" : `over ${bz.nq} quarters`;
  const money = $("biz-money"); money.replaceChildren();
  const cell = (cls, label, value, sub) => {   // sub is text, or [big, rest]: the big part (the views) is set larger so it reads on video
    const c = el("div", { class: "mcell " + cls }), s = el("small");
    if (Array.isArray(sub)) s.append(el("em", {}, sub[0]), sub[1]); else s.textContent = sub;
    c.append(el("span", {}, label), el("b", {}, value), s); return c;
  };
  const rowOf = (title, cells) => { const r = el("div", { class: "mrow" }); r.append(el("div", { class: "mtitle" }, title)); cells.forEach((c, i) => { if (i) r.append(el("div", { class: "m-op" }, i === 1 ? "−" : "=")); r.append(c); }); money.append(r); };
  rowOf("Ad revenue", [cell("model", "Our model's picks", usd(mo.adM), [`${(bz.vm / 1e6).toFixed(1)}M views`, " × λ × CPM"]), cell("rule", "Editors' rule picks", usd(mo.adR), [`${(bz.vr / 1e6).toFixed(1)}M views`, " × λ × CPM"]), cell("net", "gained", "+" + usd(mo.ad), bz.vr > 0 ? [`${(bz.vm / bz.vr).toFixed(1)}×`, ` the rule's views, ${per}`] : per)]);
  rowOf("Editor time (apart)", [cell("rule", "Screening every new serial", usd(mo.costAll), `${bz.n.toLocaleString()} serials × m × w`), cell("model", `Screening the model's top ${a.short}`, usd(mo.costShort), `${mo.screened.toLocaleString()} serials × m × w`), cell("net", "saved", "+" + usd(mo.labor), `${roundEven(mo.hours).toLocaleString()} hours ${per}`)]);
  const tot = el("div", { class: "mtotal" }); tot.append(el("span", {}, `Extra ad revenue ${per}`), el("b", {}, "+" + usd(mo.ad)), el("small", {}, "the deck's headline. Editor time saved is shown apart and not added: it is measured against screening every new serial, not against the rule"));
  money.append(tot);
  for (const b of BIZ) { const f = $(`biz-${b.id}-fx`); if (!f.dataset.done) { f.dataset.done = 1; FX[`biz_${b.id}`] = [b.tex]; fx(`biz_${b.id}`, f); } $(`biz-${b.id}-note`).textContent = b.note(bz); }
  const n = (v) => Math.round(v).toLocaleString().replace(/,/g, "{,}");
  FX.biz = [String.raw`\text{Extra ad revenue} = \sum_{q}\big(V^{\text{model}}_{q} - V^{\text{rule}}_{q}\big)\,\lambda\,\tfrac{\text{CPM}}{1000}`,
            "Vq: total views (June 2025) of the K serials each side picks in quarter q · Nq: new serials in quarter q · as in modeling/08_impact.py",
            String.raw`\text{Editor time saved} = \sum_{q}\big(N_{q} - S\big)\,\tfrac{m}{60}\,w`];
  FX.biz_num = [String.raw`= \big[${n(bz.vm)} - ${n(bz.vr)}\big] \times ${a.lambda.toFixed(2)} \times \tfrac{${a.cpm}}{1000} = \$${n(mo.ad)}`, "",
                String.raw`\text{editor time: } \big[${n(bz.n)} - ${n(mo.screened)}\big] \times \tfrac{${a.mins}}{60} \times ${a.wage} = \$${n(mo.labor)}\ \ (${n(roundEven(mo.hours))}\text{ hours})`];
  fx("biz", $("fx-biz")); fx("biz_num", $("fx-biz-num")); fitFormulas($("biz"));
  $("biz-cap").textContent = `${quarterLabel(byQ)} · ${whoLabel()} · our model = logistic regression on S+A+T. Defaults as in the deck (modeling/08_impact.py): 10% lift, $3 per 1,000 views, 200 serials screened, 10 minutes, $30 an hour. Views are a June 2025 snapshot and would partly have come anyway, so these are assumptions to be measured in a pilot.`;
  if ($("biz-drawer").classList.contains("open")) renderSens(byQ);
}
const SENS_LINES = [   // lines in the sensitivity chart; click a legend entry to show or hide it
  { key: "adM", label: "our model's picks: ad revenue", color: "#4c9bff", on: true },
  { key: "adR", label: "editors' rule picks: ad revenue", color: "#7f8287", on: true },
  { key: "ad", label: "ad revenue gained (the gap)", color: "#d27a50", on: true, bold: true },
  { key: "labor", label: "editor time saved (apart)", color: "#f08fc0", on: false, dash: true },
];
function renderSens(byQ, retry = true) {   // money (per year) as one assumption moves; the others stay put
  const a = bizA(), k = slots();
  document.querySelectorAll("[data-sens]").forEach((b) => {   // the chips show the current values (the sliders may be folded away)
    const sp = BIZ.find((x) => x.id === b.dataset.sens);
    b.classList.toggle("on", b.dataset.sens === sensVar);
    b.replaceChildren(el("small", {}, sp.short), el("b", {}, `${sp.sym} = ${sp.fmt(sp.id === "k" ? k : a[sp.id])}`));
  });
  const lg = $("sens-legend"); lg.replaceChildren();   // the legend first: its height decides how tall the chart can be
  SENS_LINES.forEach((l) => {
    const i = el("i"); i.style.background = l.dash ? `repeating-linear-gradient(90deg, ${l.color} 0 7px, transparent 7px 11px)` : l.color;
    const sp = el("button", { class: "lg" + (l.on ? " on" : "") }); sp.append(i, l.label);
    sp.addEventListener("click", () => { l.on = !l.on; renderSens(pool()); });
    lg.append(sp);
  });
  const spec = BIZ.find((b) => b.id === sensVar), logK = sensVar === "k";
  const nMax = Math.max(...bizPrep(byQ).qs.map((q) => q.n));   // slots can go up to every serial of the quarter
  const lo = spec.min, hi = logK ? nMax : spec.max, xs = [];
  if (logK) { for (let x = 10; x < nMax; x += x < 300 ? 5 : 20) xs.push(x); xs.push(nMax); } else for (let i = 0; i <= 48; i++) xs.push(lo + (i / 48) * (hi - lo));
  const pts = xs.map((x) => { const aa = { ...a }; if (!logK) aa[sensVar] = x; return { x, ...bizMoney(bizNumbers(byQ, logK ? x : k), aa) }; });
  const cur = logK ? Math.min(k, nMax) : a[sensVar], curPt = { x: cur, ...bizMoney(bizNumbers(byQ, k), a) };
  const box = $("sens-chart"), { s, W, H } = newChart(box), L = 74, R = 24, T = 18, Bm = 48, on = SENS_LINES.filter((l) => l.on);
  const ymax = Math.max(1, ...pts.flatMap((p) => on.map((l) => p[l.key]))) * 1.08, ymin = Math.min(0, ...pts.flatMap((p) => on.map((l) => p[l.key])));
  const X = logK ? (x) => L + (Math.log(x / 10) / Math.log(hi / 10)) * (W - L - R) : (x) => L + ((x - lo) / (hi - lo)) * (W - L - R);
  const Y = (y) => T + (1 - (y - ymin) / (ymax - ymin)) * (H - T - Bm);
  const g = svg("g", { class: "grid" }, s);
  for (let i = 0; i <= 4; i++) { const v = ymin + (i / 4) * (ymax - ymin); svg("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v) }, g); svg("text", { class: "muted", x: L - 8, y: Y(v) + 4, "text-anchor": "end" }, s, "$" + Math.round(v / 1000) + "k"); }
  const ticks = logK ? [10, 25, 50, 100, 250, 500, 1000, nMax] : [0, 1, 2, 3, 4].map((i) => lo + (i / 4) * (hi - lo));
  ticks.forEach((v) => svg("text", { class: "muted", x: X(v), y: H - Bm + 20, "text-anchor": "middle" }, s, logK ? v.toLocaleString() : spec.fmt(v)));
  svg("text", { class: "muted", x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle" }, s, logK ? `${spec.label} (K, log scale: up to all ${nMax.toLocaleString()} serials of a quarter)` : `${spec.label} (${spec.sym})`);
  for (const l of on) svg("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p[l.key])}`).join(""), class: "sline" + (l.bold ? " hot" : ""), style: `stroke:${l.color};stroke-width:${l.bold ? 3.5 : 2.4};${l.dash ? "stroke-dasharray:7 5" : ""}` }, s);
  svg("line", { x1: X(cur), x2: X(cur), y1: T, y2: H - Bm, class: "scur" }, s);
  if (SENS_LINES.find((l) => l.key === "ad").on) { svg("circle", { cx: X(cur), cy: Y(curPt.ad), r: 5, class: "sdot" }, s); svg("text", { class: "val", x: X(cur) + 10, y: Y(curPt.ad) + 18 }, s, `now ${usd(curPt.ad)}`); }
  let note = "Every unit adds the same amount: the money is a straight line in this assumption.";
  if (logK) {   // the peak of the net value, and the turning point on the way up (smoothed: single blockbusters make the curve jagged)
    const sm = pts.map((p, i) => { const w = pts.slice(Math.max(0, i - 3), i + 4); return w.reduce((t, q) => t + q.ad, 0) / w.length; });
    const iPeak = sm.indexOf(Math.max(...sm)), pk = pts[iPeak];
    const up = sm.slice(0, iPeak + 1), y0 = Math.min(...up), y1 = Math.max(...up);
    let iKnee = 0, best = -1; up.forEach((v, i) => { const dd = (v - y0) / (y1 - y0 || 1) - i / Math.max(1, up.length - 1); if (dd > best) { best = dd; iKnee = i; } });
    const kn = pts[iKnee];
    if (SENS_LINES.find((l) => l.key === "ad").on) {
      svg("circle", { cx: X(pk.x), cy: Y(pk.ad), r: 7, class: "speak" }, s);
      svg("text", { class: "kneetxt", x: X(pk.x), y: Y(pk.ad) - 14, "text-anchor": "middle" }, s, `peak: K ≈ ${pk.x} → ${usd(pk.ad)} a year`);
      svg("circle", { cx: X(kn.x), cy: Y(kn.ad), r: 5, class: "skneedot" }, s);
      svg("text", { class: "kneetxt dim", x: X(kn.x) - 8, y: Y(kn.ad) - 10, "text-anchor": "end" }, s, `turning point ≈ ${kn.x}`);
    }
    note = `The ad revenue gained rises fast up to about ${kn.x} slots a quarter, peaks at about ${pk.x} (${usd(pk.ad)} a year) and then falls: with very many slots both lists hold nearly the same serials, so the gap closes (at all ${nMax.toLocaleString()} serials it is $0).`;
  }
  $("sens-cap").textContent = note;
  if (retry && box.clientHeight + 2 < H) return renderSens(byQ, false);   // the caption grew and squeezed the chart: draw it once more at the new height
  const sl = $("sens-slider"), src = $(`biz-${sensVar}`);   // the drawer's own slider moves the chosen assumption
  Object.assign(sl, { min: src.min, max: src.max, step: src.step }); sl.value = src.value; $("sens-val").textContent = spec.fmt(bizReal(spec, Number(src.value)));
}

// ---------- page ----------
function render() {
  syncBuilder();
  const byQ = pool(), k = slots(), model = $("f-model").value;
  $("f-authors").classList.toggle("warn", $("f-authors").value === "first");   // make a non-default author filter hard to miss
  renderBuilder(byQ, k);
  renderPicks(byQ, k, model);
  renderChecks(byQ);
  renderBiz(byQ);
}

const SCREENS = ["title", "why", "hit", "model", "checks", "picks", "biz", "p2"];
const barOffset = (id) => (["model", "checks", "picks"].includes(id) ? $("filters").offsetHeight : 0);   // the filter bar sticks over these three
const topOf = (id) => $(id).getBoundingClientRect().top + scrollY - barOffset(id);
function currentScreen() {
  return SCREENS.reduce((best, id, k) => (Math.abs(topOf(id) - scrollY) < Math.abs(topOf(SCREENS[best]) - scrollY) ? k : best), 0);
}
function jump(k) {   // straight to a screen behind a short black fade, without scrolling past the ones between
  const veil = $("veil"), html = document.documentElement;
  veil.classList.add("on");
  setTimeout(() => {
    cancelAnimationFrame(navFrame); navIndex = null; html.style.scrollSnapType = "none";
    scrollTo(0, topOf(SCREENS[k]));
    requestAnimationFrame(() => { html.style.scrollSnapType = ""; veil.classList.remove("on"); });
  }, 220);
}
let navFrame = 0, navIndex = null;   // our own smooth scroll: works even while the cover's wall is animating
function goTo(k) {
  const html = document.documentElement, from = scrollY, to = topOf(SCREENS[k]), t0 = performance.now(), dur = 750;
  cancelAnimationFrame(navFrame); navIndex = k; html.style.scrollSnapType = "none";   // snapping would fight the animation
  const step = (now) => {
    const t = Math.min(1, (now - t0) / dur), e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    scrollTo(0, from + (to - from) * e);
    if (t < 1) navFrame = requestAnimationFrame(step); else { navIndex = null; html.style.scrollSnapType = ""; }
  };
  navFrame = requestAnimationFrame(step);
}

function init() {
  if (!D) { document.body.prepend(el("p", { style: "padding:20px" }, "Could not load data.js. Run modeling/05_demo_data.py to create docs/demo/data.js next to this page.")); return; }
  document.documentElement.style.setProperty("--c-cover", CONFIG.blocks.C);
  FX.shift = [String.raw`\cos(\Delta_{\text{hit}},\, \Delta_{\text{all}}) = \frac{\Delta_{\text{hit}} \cdot \Delta_{\text{all}}}{\lVert \Delta_{\text{hit}} \rVert\, \lVert \Delta_{\text{all}} \rVert} = ${D.story_map.shift_cos_1536.toFixed(2)}, \quad \Delta_g = \bar t_{g,\,2024} - \bar t_{g,\,2017}`,
              "measured in all 1,536 numbers, so it holds for every tab below · t̄: average of a group's vectors that year (hits, or all new serials; no watermarked serials) · 1 = same direction, 0 = unrelated"];
  const fq = $("f-quarter");
  fq.append(el("option", { value: "Y2023" }, "2023, all quarters"));
  D.quarters.forEach((q, i) => q.startsWith("2023") && fq.append(el("option", { value: i }, q)));
  fq.append(el("option", { value: "Y2024" }, "2024, all quarters (not settled)"));
  D.quarters.forEach((q, i) => q.startsWith("2024") && fq.append(el("option", { value: i }, q + " (not settled)")));
  fq.value = CONFIG.quarter;
  for (const [m, label] of Object.entries(D.models)) $("f-model").append(el("option", { value: m }, label));
  Object.assign($("f-k"), { min: 0, max: K_STEPS.length - 1, step: 1 }); $("f-k").value = posOfK(CONFIG.slots);
  $("f-model").value = CONFIG.model; $("f-k-val").textContent = slots();
  ["f-quarter", "f-authors"].forEach((id) => $(id).addEventListener("change", render));
  $("f-model").addEventListener("change", () => { manualMode(); render(); });
  let queued = false;   // the slider re-renders at most once per frame
  $("f-k").addEventListener("input", () => { $("f-k-val").textContent = slots(); if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; render(); }); } });
  $("f-k").addEventListener("change", () => $("f-k").blur());   // so the arrow keys go back to moving between screens
  document.querySelectorAll("[data-story]").forEach((b) => b.addEventListener("click", () => {
    const s = CONFIG.stories[b.dataset.story]; $("f-model").value = s.model; $("f-authors").value = s.authors;
    document.querySelectorAll("[data-story]").forEach((x) => x.classList.toggle("on", x === b));
    manualMode(); render();
  }));
  document.querySelectorAll("[data-layer]").forEach((b) => b.addEventListener("click", () => { players.picks.pause(); setLayer(b.dataset.layer); }));
  $("dim").addEventListener("click", () => { players.picks.pause(); setDim(!$("covers").classList.contains("dim")); });
  $("covers-all").addEventListener("click", () => { showAllCovers = !showAllCovers; render(); });
  document.querySelectorAll("[data-taste]").forEach((b) => b.addEventListener("click", () => { tasteView = b.dataset.taste; renderTaste(true); }));

  players.title = makePlayer("title", [() => restart($("title"), "in")], []);
  players.p2 = makePlayer("p2", [() => restart($("p2"), "in")], []);
  players.biz = makePlayer("biz", [() => restart($("biz-money"), "in")], []);
  watchWidth($("models-chart"), () => drawShares(pool(), slots()));
  $("detail-drawer").addEventListener("transitionend", (e) => e.target === e.currentTarget && e.propertyName === "width" && drawShares(pool(), slots()));   // and once more when the drawer stops
  watchWidth($("sens-chart"), () => $("biz-drawer").classList.contains("open") && renderSens(pool()));
  $("detail-close").addEventListener("click", () => setDetails(false));
  $("detail-open").addEventListener("click", () => setDetails(true));
  document.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => jump(SCREENS.indexOf(b.dataset.go))));
  fx("target", $("fx-target")); fx("share", $("fx-share")); fx("lift", $("fx-lift")); fx("shift", $("fx-shift"));
  setTimeout(() => fitFormulas(), 300);
  players.why = makePlayer("why", [0, 1, 2, 3].map((s) => () => whyShow(s)), CONFIG.waits.why);
  players.hit = makePlayer("hit", [0, 1, 2].map((s) => () => animShow(s)), CONFIG.waits.hit);
  players.model = makePlayer("model", [modelIntro], []);
  players.picks = makePlayer("picks", [() => { setDim(false); setLayer("picks"); }, () => setLayer("hits"), () => setLayer("both"), () => setDim(true)], CONFIG.waits.picks);
  players.checks = makePlayer("checks", [playChecks], []);
  document.querySelectorAll("[data-play]").forEach((b) => b.addEventListener("click", () => players[b.dataset.play].toggle()));
  document.querySelectorAll("[data-step-of]").forEach((b) => b.addEventListener("click", () => players[b.dataset.stepOf].jump(Number(b.dataset.step))));
  document.addEventListener("click", (e) => e.target.closest("button")?.blur());   // keep the space bar for Play / Pause

  document.addEventListener("keydown", (e) => {   // ↓ / ↑ move between screens, space plays or pauses the current one
    if (e.target.closest("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    const cur = navIndex ?? currentScreen();   // quick repeated presses keep counting from where the scroll is heading
    if (e.key === "ArrowDown" || e.key === "PageDown") { e.preventDefault(); goTo(Math.min(SCREENS.length - 1, cur + 1)); }
    else if (e.key === "ArrowUp" || e.key === "PageUp") { e.preventDefault(); goTo(Math.max(0, cur - 1)); }
    else if (e.key === " ") { e.preventDefault(); players[SCREENS[currentScreen()]].toggle(); }
  });
  // every content screen gets a 1920 × 1080 stage that is scaled to the window, like a slide; the cover and Project Two fill the window as they are
  for (const id of ["why", "hit", "model", "checks", "picks", "biz"]) {
    const scr = $(id), st = el("div", { class: "stage" + (scr.classList.contains("full") ? " full" : "") });
    while (scr.firstChild) st.append(scr.firstChild);
    scr.append(st); scr.classList.add("scaled");
  }
  const setBar = () => {
    const W = innerWidth, H = innerHeight, z = Math.min(W / 1920, H / 1080), root = document.documentElement.style;
    root.setProperty("--z", z); root.setProperty("--bar", 66 * z + "px");
    document.querySelectorAll(".stage").forEach((st) => {
      const dh = st.classList.contains("full") ? 1080 : 1014, sh = st.parentElement.clientHeight;
      st.style.transform = `translate(${(W - 1920 * z) / 2}px, ${Math.max(0, (sh - dh * z) / 2)}px) scale(${z})`;
    });
  };
  setBar();
  heroInit(); whyInit(); layoutWall(); bizInit();
  animInit();
  render();
  // arriving at a screen plays it from the start; leaving it jumps to its end state
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.target.id === "title") e.target.classList.toggle("live", e.isIntersecting);
    e.isIntersecting ? players[e.target.id].play(true) : players[e.target.id].finish();
  }), { threshold: 0.6 });
  SCREENS.forEach((id) => io.observe($(id)));
  addEventListener("scroll", () => $("peek").classList.remove("on"), { passive: true });   // the floating cover card never lingers on other screens
  let t; window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(setBar, 100); });   // the stages keep their 1920 × 1080 layout; only the scale changes
}
init();
