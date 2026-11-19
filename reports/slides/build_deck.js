// Royal Road Breakout: pitch deck (10 slides) and 2-minute video storyboard (6 frames).
// Every number is read from modeling/out/*.json, so rerun 03, 06, 07 and then this script.
// Run: NODE_PATH=<folder with pptxgenjs> node reports/slides/build_deck.js
// Figures: reports/slides/make_figs.py (UMAP), reports/slides/shot_atlas.js (Atlas screenshot), img/rr_*.png (Internet Archive, June 2025).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/Users/syuanbo/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/28299c02-a80d-46a6-a9e2-f968fb84eb2d/02c5729a-765e-429b-9195-15f54699b4a0/skills/pptx/scripts/apply_theme.js");

const REPO = path.resolve(__dirname, "../..");
const OUT = path.join(REPO, "reports/slides");
const IMG = path.join(OUT, "img");
const J = (f) => JSON.parse(fs.readFileSync(path.join(REPO, "modeling/out", f)));
const R = J("results.json"), I = J("impact.json"), DR = J("drift_auc.json");

const pct = (x) => Math.round(x * 100);
const usdK = (x) => "$" + Math.round(x / 1000) + "k";
const K = I.assumptions.slots_per_quarter;
const P = I.precision_at_slots;
const main = "lr_SAT", rule = "rule_hot_genre";
const M = I.money.base;
const E = I.editor;
const ex = I.example_candidates[0];
const debutShare = R.by_group.first_time_author.n / R.by_group.all.n;
const pngSize = (f) => { const b = fs.readFileSync(f); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };
const fit = (f, h) => { const s = pngSize(path.join(IMG, f)); return h * s.w / s.h; };

const THEME = {
  name: "Royal Road Breakout", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "111111", lt1: "FFFFFF", dk2: "000000", lt2: "F2F2F2", accent1: "D9480F", accent2: "595959", accent3: "8C8C8C",
            accent4: "BFBFBF", accent5: "404040", accent6: "262626", hlink: "D9480F", folHlink: "595959" },
};
const HEX = { accent: "D9480F", dark: "595959", mid: "8C8C8C", light: "BFBFBF", grid: "D9D9D9", axis: "595959", ink: "111111" };
const TEAM = "Yinqi (Erich) Yu / Hongyu Yue / Syuan-Bo Huang";

function setup(pres) {
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  pres.defineSlideMaster({ title: "Title dark", background: { color: C.text2 }, objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.2, w: 8.8, h: 1.7, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.1, w: 8.8, h: 1.5, fontSize: 16, color: "BFBFBF", valign: "top", align: "left", margin: 0 }, text: "" } },
  ] });
  pres.defineSlideMaster({ title: "Content", background: { color: C.background1 }, objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.95, fontSize: 24, bold: true, color: C.text2, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Royal Road Breakout, 15.071 Project 1", options: { x: 0.5, y: 5.25, w: 6, h: 0.22, fontSize: 9, color: "8C8C8C", margin: 0 } } },
  ], slideNumber: { x: 9.1, y: 5.25, w: 0.4, h: 0.22, fontSize: 9, color: "8C8C8C", align: "right" } });
  pres.defineSlideMaster({ title: "Closing dark", background: { color: C.text2 }, objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.95, fontSize: 24, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
  ], slideNumber: { x: 9.1, y: 5.25, w: 0.4, h: 0.22, fontSize: 9, color: "BFBFBF", align: "right" } });
  return C;
}

const chartBase = (extra) => ({
  catAxisLabelColor: HEX.axis, valAxisLabelColor: HEX.axis, catAxisLabelFontSize: 12, valAxisLabelFontSize: 10,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  valGridLine: { style: "none" }, catGridLine: { style: "none" }, valAxisHidden: true,
  showValue: true, dataLabelFontSize: 12, dataLabelColor: HEX.ink, dataLabelFormatCode: "0%", valAxisLabelFormatCode: "0%", valAxisMinVal: 0,
  ...extra,
});

function stat(s, C, x, y, w, big, label, color, name, size = 40) {
  s.addText(big, { x, y, w, h: size / 60, fontSize: size, bold: true, color, fontFace: "Cambria", margin: 0, isTextBox: true, objectName: name + " number" });
  s.addText(label, { x, y: y + size / 60 + 0.05, w, h: 0.7, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: name + " label" });
}
const box = (s, pres, C, x, y, w, h, name) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.background2 }, line: { type: "none" }, rectRadius: 0.08, objectName: name });

// =====================================================================  DECK
function buildDeck() {
  const pres = new pptxgen();
  pres.title = "Royal Road Breakout";
  const C = setup(pres);
  let s;

  // 1. Cover
  pres.addSection({ title: "Opening" });
  s = pres.addSlide({ masterName: "Title dark", sectionTitle: "Opening" });
  s.addText("Spotting tomorrow's hits at chapter 3", { placeholder: "title" });
  s.addText(`A recommendation model for new Royal Road serials\nMIT 15.071 The AI Edge, Project 1, Fall 2026\n${TEAM}`, { placeholder: "body" });
  s.addNotes("We are pitching to Royal Road's recommendation team. Thousands of serials launch every quarter and only a few get a spot on lists like Rising Stars. We built a model that reads what exists when chapter 3 is published and ranks which serials will become hits.");

  // 2. Platform and problem
  pres.addSection({ title: "Problem" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Problem" });
  s.addText(`Every quarter, ${E.books_per_quarter.toLocaleString("en-US")} new serials compete for ${K} slots, and the top 10% get ${pct(I.money.views_share_top10)}% of all reading`, { placeholder: "title" });
  const h2 = 2.2, fW = fit("rr_fiction_page.png", h2), rW = fit("rr_rising_stars.png", h2), x0 = 0.5 + (9 - fW - rW - 0.25) / 2;
  s.addImage({ path: path.join(IMG, "rr_fiction_page.png"), x: x0, y: 1.35, w: fW, h: h2, altText: "Royal Road serial page with tags, Follow button, Patreon link and a STUB label", objectName: "Serial page screenshot" });
  s.addImage({ path: path.join(IMG, "rr_rising_stars.png"), x: x0 + fW + 0.25, y: 1.35, w: rW, h: h2, altText: "Royal Road Rising Stars list, up to 50 serials per genre", objectName: "Rising Stars screenshot" });
  [[`${E.books_per_quarter.toLocaleString("en-US")}`, "new serials a quarter, posted chapter by chapter (2023)"],
   [`${pct(I.money.views_share_top10)}%`, "of all reading goes to the top 10% of each quarter's serials"],
   [`${pct(debutShare)}%`, "come from first-time authors with no track record"]].forEach(([b, l], i) =>
    stat(s, C, 0.5 + i * 3.05, 3.8, 2.85, b, l, i === 1 ? C.accent1 : C.text1, `Problem stat ${i + 1}`, 30));
  s.addNotes(`Royal Road is a free site where authors publish web novels one chapter at a time; readers find new serials through lists like Rising Stars, which shows up to ${K} serials per genre. In 2023 about ${E.books_per_quarter} serials launched each quarter, and the top 10% of each quarter received ${pct(I.money.views_share_top10)}% of all views, so getting the few slots right matters. Screenshots: royalroad.com via the Internet Archive, June 2025, the month of our data snapshot.`);

  // 3. Data
  pres.addSection({ title: "AI solution" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "AI solution" });
  s.addText("At chapter 3 we already know the tags, the author's track record, and the story itself", { placeholder: "title" });
  const cols = [
    ["Structured data", [["Tags and content warnings", "64 tags, e.g. LitRPG, Progression, Romance\n6 warnings, e.g. Profanity, Graphic Violence", "In model"], ["Author's track record", "first book or not, earlier serials", "In model"], ["Posting speed of chapters 1 to 3", "days between chapters", "In model"]]],
    ["Unstructured data", [["First three chapters", "OpenAI text-embedding-3-small\n1,536 dimensions", "In model"], ["Blurb", "edited after success", "Excluded"], ["Cover image", "CLIP ViT-L/14; only 42% had one by chapter 3", "Tried"]]],
  ];
  cols.forEach(([head, rows], c) => {
    const x = 0.5 + c * 4.6;
    box(s, pres, C, x, 1.35, 4.4, 3.65, `${head} card`);
    s.addText(head, { x: x + 0.25, y: 1.5, w: 3.9, h: 0.35, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: `${head} title` });
    rows.forEach(([name, sub, status], r) => {
      const y = 2.0 + r * 0.72;
      s.addText([{ text: name, options: { bold: true, breakLine: true } }, { text: sub, options: { fontSize: 11, color: "595959" } }],
        { x: x + 0.25, y, w: 2.85, h: 0.62, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `${name} text` });
      const inModel = status === "In model";
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 3.2, y: y + 0.05, w: 0.98, h: 0.3, rectRadius: 0.15, fill: { color: inModel ? C.accent1 : C.background1 },
        line: { color: inModel ? C.accent1 : "8C8C8C", width: 0.75 }, objectName: `${name} chip` });
      s.addText(status, { x: x + 3.2, y: y + 0.05, w: 0.98, h: 0.3, fontSize: 10, bold: true, color: inModel ? C.background1 : "595959", align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `${name} chip text` });
    });
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.35, y: 4.2, w: 3.9, h: 0.65, fill: { color: "E6E6E6" }, line: { color: "BFBFBF", width: 0.75, dashType: "dash" }, objectName: "Cover placeholder" });
  s.addText("Cover examples from team analysis (to add)", { x: 5.35, y: 4.2, w: 3.9, h: 0.65, fontSize: 11, italic: true, color: "595959", align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "Cover placeholder text" });
  s.addNotes("Everything the model uses exists the moment chapter 3 is published. Structured inputs: 64 genre tags such as LitRPG and Romance, 6 content warnings such as profanity, the author's history, and how fast the first chapters came out. Unstructured input: the text of the first three chapters, turned into numbers by an embedding model. We tried covers with CLIP: only 42% of books had a cover in place by chapter 3, and the gain was not significant, so covers stay out. Blurbs are excluded because authors edit them after success.");

  // 4. Embedding
  s = pres.addSlide({ masterName: "Content", sectionTitle: "AI solution" });
  s.addText("Turning the first three chapters into numbers puts similar stories next to each other", { placeholder: "title" });
  const uh = 3.75, uw = fit("umap_hits.png", uh);
  s.addImage({ path: path.join(IMG, "umap_hits.png"), x: 0.5, y: 1.3, w: uw, h: uh, altText: "Map of 45,245 serials placed by the text of their first three chapters; future top-10% serials in orange", objectName: "UMAP map" });
  const ax = 0.5 + uw + 0.35, aw = 9.5 - ax, ah = aw * 1000 / 1600;
  s.addImage({ path: path.join(IMG, "atlas_screenshot.png"), x: ax, y: 1.4, w: aw, h: ah, altText: "Embedding Atlas web app showing all serials", objectName: "Atlas screenshot" });
  s.addText([
    { text: "Explore all 45,245 serials in our interactive atlas", options: { bold: true, breakLine: true } },
    { text: "godspeedhuang.github.io/royalroad-breakout", options: { color: C.accent1, hyperlink: { url: "https://godspeedhuang.github.io/royalroad-breakout/" }, breakLine: true } },
    { text: "Each serial becomes a 1,536-dimension vector\nwith OpenAI text-embedding-3-small." },
  ], { x: ax, y: 1.5 + ah, w: aw, h: 1.9, fontSize: 12, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true, objectName: "Atlas caption" });
  s.addNotes("Each dot is a serial, placed by the meaning of its first three chapters, so similar stories sit together. Orange dots became hits. Hits are spread across genres but denser in some regions such as LitRPG, which is why reading the text helps beyond the tags. The interactive atlas is public on GitHub Pages.");

  // 5. Model comparison
  pres.addSection({ title: "Does it work" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Does it work" });
  s.addText(`Reading the chapters lifts hits per slot from ${pct(P.lr_SA.value)}% to ${pct(P[main].value)}%; a neural network may add a few more`, { placeholder: "title" });
  const bars = [["Random", P.random.value, HEX.light], ["Rule: popular genres first", P[rule].value, HEX.mid], ["Logistic regression: tags + author", P.lr_SA.value, HEX.dark],
    ["Logistic regression: + chapter text", P[main].value, HEX.accent], ["Gradient boosting: same inputs", P.gb_SAT.value, HEX.dark], ["Neural network (MLP): same inputs", P.mlp_SAT.value, HEX.dark]];
  s.addChart(pres.charts.BAR, [{ name: "Hits per slot", labels: bars.map(b => b[0]), values: bars.map(b => b[1]) }], chartBase({
    x: 0.4, y: 1.3, w: 6.0, h: 3.8, barDir: "bar", barGapWidthPct: 45, chartColors: bars.map(b => b[2]), showLegend: false,
    dataLabelPosition: "outEnd", valAxisMaxVal: 0.9, catAxisOrientation: "maxMin", catAxisLabelFontSize: 11,
    showTitle: true, title: `Share of ${K} slots per quarter that go to future hits (2023 test year)`, titleFontSize: 11, titleColor: HEX.axis, objectName: "Model comparison chart",
  }));
  const d = I.diff_pp, d2 = I.diff_pp_2023_2024;
  s.addText([
    { text: `${pct(P.lr_SA.value)}% → ${pct(P[main].value)}%`, options: { bold: true, color: C.accent1, fontSize: 24, breakLine: true } },
    { text: `after adding the chapter text, a significant gain (95% CI +${d.text.ci95[0]}% to +${d.text.ci95[1]}%)`, options: { breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: `${pct(P.gb_SAT.value)}% and ${pct(P.mlp_SAT.value)}%`, options: { bold: true, color: "595959", fontSize: 24, breakLine: true } },
    { text: `for boosting and the neural network. Not significant in 2023; over 2023 and 2024 the neural network leads by ${d2.mlp_vs_lr.diff}% (95% CI +${d2.mlp_vs_lr.ci95[0].toFixed(1)}% to +${d2.mlp_vs_lr.ci95[1].toFixed(1)}%). We keep the simple, explainable model and recheck at each yearly retrain.`, options: {} },
  ], { x: 6.7, y: 1.5, w: 2.8, h: 3.4, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "Model takeaways" });
  s.addNotes(`With ${K} slots per quarter, the best simple rule fills ${pct(P[rule].value)}% of them with future hits. Logistic regression on tags and author history reaches ${pct(P.lr_SA.value)}%, and adding the text of the first chapters reaches ${pct(P[main].value)}%. Gradient boosting and a neural network on the same inputs score a little higher. In 2023 alone the gap is within noise; adding 2024 as a second check, the neural network stays ahead by about ${d2.mlp_vs_lr.diff}%, but 2024 books are still young and we chose this comparison after seeing the results. So we keep logistic regression, which is cheaper, faster to retrain, and easier to explain, and we recheck the neural network at every yearly retrain. Author history alone adds ${d.author.diff}% of slots.`);

  // 6. Result
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Does it work" });
  s.addText(`The model more than doubles the hits per slot: ${Math.round(I.hits_per_quarter[main])} vs ${Math.round(I.hits_per_quarter[rule])} per quarter`, { placeholder: "title" });
  stat(s, C, 0.5, 1.5, 3.3, `${Math.round(I.hits_per_quarter[main])} vs ${Math.round(I.hits_per_quarter[rule])}`, `future hits among ${K} slots per quarter, model vs best rule (2023 test year)`, C.accent1, "Hits per quarter", 48);
  s.addText(`Model: ${pct(P[main].value)}% of slots (95% CI ${pct(P[main].ci95[0])}% to ${pct(P[main].ci95[1])}%). Random picks: ${pct(P.random.value)}%.`, { x: 0.5, y: 3.35, w: 3.3, h: 0.8, fontSize: 12, color: "595959", margin: 0, valign: "top", isTextBox: true, objectName: "CI note" });
  const G = I.by_group;
  s.addChart(pres.charts.BAR, [
    { name: "Rule: popular genres first", labels: ["First-time authors", "Returning authors"], values: [G.first_time_author[rule], G.returning_author[rule]] },
    { name: "Model", labels: ["First-time authors", "Returning authors"], values: [G.first_time_author[main], G.returning_author[main]] },
  ], chartBase({ x: 4.1, y: 1.3, w: 5.5, h: 3.8, barDir: "col", barGapWidthPct: 70, chartColors: [HEX.mid, HEX.accent], showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.axis,
    dataLabelPosition: "outEnd", valAxisMaxVal: 0.9, showTitle: true, title: "Share of slots that go to future hits, by author group", titleFontSize: 11, titleColor: HEX.axis, objectName: "Author group chart" }));
  s.addNotes(`Out of ${K} slots a quarter, the model places about ${Math.round(I.hits_per_quarter[main])} future hits against ${Math.round(I.hits_per_quarter[rule])} for the best rule. First-time authors, two thirds of new serials, get slots in proportion to their share; among them the model hits ${pct(G.first_time_author[main])}% versus ${pct(G.first_time_author[rule])}% for the rule. The README reports the same comparison with a slot count equal to the number of hits (about 200 a quarter): 48% versus 26%. Fewer slots make the model's edge larger.`);

  // 7. Day 30
  const D30 = I.day30;
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Does it work" });
  s.addText(`A second look at day 30 sharpens the list: ${pct(D30.test[main])}% to ${pct(D30.test.day30)}% of slots`, { placeholder: "title" });
  const med = J("day30.json").describe_trainval.median_chapters_first_30d;
  stat(s, C, 0.5, 1.5, 3.2, `${med.top10} vs ${med.others}`, "chapters posted in the first 30 days, future hits vs others (median)", C.accent1, "Cadence", 40);
  s.addText([
    { text: "Add the first month's posting rhythm and re-rank.", options: { bold: true, breakLine: true } },
    { text: `2023: ${pct(D30.test[main])}% to ${pct(D30.test.day30)}% (95% CI of the gain +${D30.test.diff_ci95[0]}% to +${D30.test.diff_ci95[1]}%); first-time authors ${pct(D30.test_first_time[main])}% to ${pct(D30.test_first_time.day30)}%.`, options: { breakLine: true } },
    { text: "Use it as a second pass and A/B test it first." },
  ], { x: 0.5, y: 3.15, w: 3.2, h: 1.9, fontSize: 12, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 5, isTextBox: true, objectName: "Day 30 commentary" });
  s.addChart(pres.charts.BAR, [
    { name: "Chapter 3 model", labels: ["2023 (test)", "2024 (young books)"], values: [D30.test[main], D30.recent_2024[main]] },
    { name: "Day 30 model", labels: ["2023 (test)", "2024 (young books)"], values: [D30.test.day30, D30.recent_2024.day30] },
  ], chartBase({ x: 4.0, y: 1.3, w: 5.6, h: 3.8, barDir: "col", barGapWidthPct: 70, chartColors: [HEX.dark, HEX.accent], showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.axis,
    dataLabelPosition: "outEnd", valAxisMaxVal: 1.0, showTitle: true, title: `Share of ${K} slots per quarter that go to future hits`, titleFontSize: 11, titleColor: HEX.axis, objectName: "Day 30 chart" }));
  s.addNotes("Readers follow serials they trust to keep updating, so the first month's posting rhythm is a strong signal. Re-ranking at day 30 improves the top of the list. 2024 books are young and their labels are still settling, so the 2024 gain may be overstated.");

  // 8. Money
  pres.addSection({ title: "Does it matter" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Does it matter" });
  s.addText(`Better picks are worth about ${usdK(M.extra_revenue_per_year)} a year in ad revenue at ${K} slots a quarter`, { placeholder: "title" });
  const A = I.assumptions;
  stat(s, C, 0.5, 1.45, 3.3, `${M.ratio_model_to_rule}x`, `the reading of the rule's picks: total views of the serials the model puts in the same ${K} slots`, C.text1, "Views ratio", 40);
  stat(s, C, 0.5, 2.95, 3.3, `${usdK(M.extra_revenue_per_year)} a year`, `if a slot lifts a serial's views by ${pct(A.lift.base)}% and ads earn $${A.cpm_usd.base} per 1,000 views (range ${usdK(M.low)} to ${usdK(M.high)})`, C.accent1, "Revenue", 32);
  const bySlots = Object.entries(I.money.by_slots);
  s.addChart(pres.charts.BAR, [{ name: "Extra ad revenue per year", labels: bySlots.map(([k]) => `${k} slots`), values: bySlots.map(([, v]) => v.extra_revenue_per_year.base_base / 1000) }], chartBase({
    x: 4.1, y: 1.3, w: 5.5, h: 3.2, barDir: "col", barGapWidthPct: 60, chartColors: bySlots.map(([k]) => (+k === K ? HEX.accent : HEX.light)), showLegend: false,
    dataLabelPosition: "outEnd", dataLabelFormatCode: '"$"0"k"', valAxisLabelFormatCode: '"$"0"k"', showTitle: true, title: "Extra ad revenue per year by slots per quarter (base case)", titleFontSize: 11, titleColor: HEX.axis, objectName: "Revenue by slots chart",
  }));
  s.addText(`Editors could screen the model's top ${A.editor_shortlist} instead of all ${E.books_per_quarter.toLocaleString("en-US")} new serials each quarter, about ${E.hours_saved_per_year.toLocaleString("en-US")} hours a year (${usdK(E.usd_saved_per_year)} at $${A.editor_hourly_usd}/hour).`,
    { x: 4.1, y: 4.55, w: 5.4, h: 0.55, fontSize: 11, color: "595959", margin: 0, valign: "top", isTextBox: true, objectName: "Editor note" });
  s.addNotes(`Assumptions, all set in modeling/07_impact.py: ${K} slots per quarter; a slot lifts a serial's lifetime views by ${pct(A.lift.base)}% (range ${pct(A.lift.low)} to ${pct(A.lift.high)}%); display ads earn $${A.cpm_usd.base} per 1,000 views (range $${A.cpm_usd.low} to $${A.cpm_usd.high}). The model's picks gather ${M.ratio_model_to_rule} times the views of the rule's picks, so the same slots produce about ${(M.extra_views_per_year / 1e6).toFixed(1)} million extra views a year. This counts only better use of the same slots, not readers who stay on the site longer, so it is conservative. The editor estimate assumes ${A.editor_minutes_per_book} minutes per serial; it shows that human review of every new serial becomes feasible, not money Royal Road spends today.`);

  // 9. Caveats
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Does it matter" });
  s.addText("We left out signals that change after success, and retrain every year", { placeholder: "title" });
  const leak = R.leak_checks.blurb_marker;
  [["Excluded: edited after success", [
    `Blurbs: ${(leak.share_of_test * 100).toFixed(1)}% of 2023 blurbs mention Kindle, Patreon or "Book 2"; ${pct(leak.top10_rate_if_flagged)}% of those are hits`,
    "Covers: books whose cover appeared after chapter 3 hit three times as often (team analysis)",
    "Earlier books' followers: inflated when the new book takes off"], C.text1],
   ["Keep it fresh: retrain yearly", null, C.accent1]].forEach(([h, items, col], i) => {
    const x = 0.5 + i * 4.6;
    box(s, pres, C, x, 1.4, 4.4, 3.6, `Caveat card ${i + 1}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 1.6, w: 0.3, h: 0.3, fill: { color: col }, line: { type: "none" }, objectName: `Caveat dot ${i + 1}` });
    s.addText(h, { x: x + 0.7, y: 1.55, w: 3.5, h: 0.4, fontSize: 16, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: `Caveat title ${i + 1}` });
    if (items) s.addText(items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } })),
      { x: x + 0.25, y: 2.15, w: 3.9, h: 2.7, fontSize: 14, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true, objectName: `Caveat list ${i + 1}` });
  });
  // 右卡：測試年固定 2023，訓練資料越舊 AUC 越低（modeling/04_drift.py）
  const gaps = Object.keys(DR.S["2023"]).sort();
  s.addChart(pres.charts.LINE, [
    { name: "Tags and structure only", labels: gaps, values: gaps.map(g => DR.S["2023"][g].auc) },
    { name: "Chapter text only", labels: gaps, values: gaps.map(g => DR.T["2023"][g].auc) },
  ], chartBase({ x: 5.2, y: 2.0, w: 4.2, h: 2.45, chartColors: [HEX.mid, HEX.accent], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7,
    showValue: false, valAxisHidden: false, valAxisMinVal: 0.7, valAxisMaxVal: 0.85, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00", valGridLine: { color: HEX.grid, size: 0.5 },
    showCatAxisTitle: true, catAxisTitle: "Years between training data and the 2023 test year", catAxisTitleFontSize: 10, catAxisTitleColor: HEX.axis, catAxisLabelFontSize: 11,
    showValAxisTitle: true, valAxisTitle: "AUC", valAxisTitleFontSize: 10, valAxisTitleColor: HEX.axis,
    showLegend: true, legendPos: "t", legendFontSize: 11, legendColor: HEX.axis, objectName: "Drift chart" }));
  s.addText(`Each extra year of data age costs ${Math.abs(DR.S.auc_change_per_year_of_gap).toFixed(3)} AUC for tags, ${Math.abs(DR.T.auc_change_per_year_of_gap).toFixed(3)} for text.`,
    { x: 5.35, y: 4.5, w: 3.9, h: 0.45, fontSize: 12, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "Drift caption" });
  s.addNotes("The dataset is a June 2025 snapshot. Authors edit blurbs and covers after success, and a hit book sends readers back to the author's older work. Using those would look great on paper and fail in production, so we excluded them. Tastes also drift: tags lose predictive power faster than text, so the model should be retrained every year.");

  // 10. Recommendation
  pres.addSection({ title: "Recommendation" });
  s = pres.addSlide({ masterName: "Closing dark", sectionTitle: "Recommendation" });
  s.addText("Pilot it: score at chapter 3, re-rank at day 30, and A/B test half the slots", { placeholder: "title" });
  [["Score at chapter 3", `Fill the ${K} slots from the top of the ranking: about ${Math.round(I.hits_per_quarter[main] - I.hits_per_quarter[rule])} more future hits per quarter than the rule`],
   ["Re-rank at day 30", "Add posting rhythm; swap in debut authors who keep updating"],
   ["A/B test first", "Half the slots by model, half by editors; compare follower and view growth for one quarter"],
   ["Maintain", "Retrain yearly, score on characters not chapter counts so splitting chapters does not pay, and log cover and blurb versions"]].forEach(([h, b], i) => {
    const y = 1.45 + i * 0.9;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y, w: 0.5, h: 0.5, fill: { color: C.accent1 }, line: { type: "none" }, objectName: `Step dot ${i + 1}` });
    s.addText(String(i + 1), { x: 0.5, y, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `Step number ${i + 1}` });
    s.addText(h, { x: 1.25, y: y - 0.02, w: 2.6, h: 0.55, fontSize: 17, bold: true, color: C.background1, margin: 0, valign: "middle", isTextBox: true, objectName: `Step title ${i + 1}` });
    s.addText(b, { x: 3.9, y: y - 0.05, w: 5.6, h: 0.65, fontSize: 14, color: "D9D9D9", margin: 0, valign: "middle", isTextBox: true, objectName: `Step body ${i + 1}` });
  });
  s.addNotes("We recommend a pilot rather than a rollout: our numbers come from history, and in a live system the slots themselves change outcomes. An A/B test on half the slots for one quarter measures the real lift and replaces our revenue assumption with Royal Road's own number.");
  return pres;
}

// Narration, about 280 words; also written to video_script.md.
const VIDEO_SCRIPT = [
  `In ${ex.quarter.slice(0, 4)}, a first-time author launched "${ex.title}" on Royal Road. When chapter 3 went up, our model ranked it number ${ex.rank_in_quarter} out of ${ex.books_in_quarter.toLocaleString("en-US")} new serials that quarter. Today it has ${ex.followers.toLocaleString("en-US")} followers.`,
  `That is the problem we solve. Every quarter about ${E.books_per_quarter.toLocaleString("en-US")} new serials launch, but lists like Rising Stars hold only ${K} per genre, and the top ten percent take ${pct(I.money.views_share_top10)} percent of all reading. Two thirds come from authors with no track record.`,
  `Our model reads what exists at chapter 3: the tags, the author's history, and the first three chapters themselves, turned into numbers so that similar stories sit close together. Here is our atlas of forty-five thousand serials; orange dots became hits. A simple logistic regression did as well as boosting and a neural network.`,
  `On 2023 serials the model never saw, it fills ${pct(P[main].value)} percent of ${K} slots with future hits, against ${pct(P[rule].value)} percent for the best editorial rule. That is ${Math.round(I.hits_per_quarter[main])} hits per quarter instead of ${Math.round(I.hits_per_quarter[rule])}.`,
  `Those picks attract ${M.ratio_model_to_rule} times the reading. If a slot lifts views by ${pct(I.assumptions.lift.base)} percent, that is about ${Math.round(M.extra_revenue_per_year / 1000)} thousand dollars a year in ad revenue, from the same slots.`,
  `We recommend a pilot: score every serial at chapter 3, re-rank at day 30 using posting rhythm, and A/B test half the slots for one quarter.`,
];
const TIMES = ["0:00 to 0:20", "0:20 to 0:35", "0:35 to 1:05", "1:05 to 1:25", "1:25 to 1:45", "1:45 to 2:00"];

// =====================================================================  VIDEO STORYBOARD
function buildStoryboard() {
  const pres = new pptxgen();
  pres.title = "Royal Road Breakout video storyboard";
  const C = setup(pres);
  const frame = (i, title, body, img) => {
    // Text-only frames use the dark title layout with large type; frames with a picture use the light layout.
    const s = pres.addSlide({ masterName: img ? "Content" : "Title dark", sectionTitle: "Video" });
    s.addText(title, { placeholder: "title" });
    if (img) {
      s.addImage({ path: path.join(IMG, img.file), x: img.x, y: 1.3, w: img.w, h: img.h, objectName: "Frame image" });
      s.addText(body, { x: 0.5, y: 1.4, w: img.x - 0.8, h: 3.5, fontSize: 20, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 10, isTextBox: true, objectName: "Frame text" });
    } else {
      // The placeholder's own size wins over call options, so set the size on each run.
      s.addText(body.map((r) => ({ text: r.text, options: { fontSize: 22, paraSpaceAfter: 8, ...r.options } })), { placeholder: "body" });
    }
    s.addText(TIMES[i], { x: 7.5, y: 5.25, w: 1.5, h: 0.22, fontSize: 9, color: "8C8C8C", align: "right", margin: 0, isTextBox: true, objectName: "Timing" });
    s.addNotes(VIDEO_SCRIPT[i]);
  };
  pres.addSection({ title: "Video" });
  frame(0, `"${ex.title}"`, [
    { text: `A first-time author, launched ${ex.quarter.replace("Q", " Q")}`, options: { breakLine: true } },
    { text: `Ranked #${ex.rank_in_quarter} of ${ex.books_in_quarter.toLocaleString("en-US")} new serials at chapter 3`, options: { bold: true, color: C.accent1, breakLine: true } },
    { text: `Today: ${ex.followers.toLocaleString("en-US")} followers` },
  ]);
  frame(1, `${E.books_per_quarter.toLocaleString("en-US")} new serials a quarter, ${K} slots`, [
    { text: `The top 10% get ${pct(I.money.views_share_top10)}% of all reading`, options: { breakLine: true } },
    { text: "Two thirds come from first-time authors" },
  ], { file: "rr_rising_stars.png", x: 5.0, w: 4.5, h: 4.5 * 860 / 1150 });
  frame(2, "Demo: the model reads the first three chapters", "Screen recording: open the atlas, find the example serial, show the hits around it",
    { file: "umap_hits.png", x: 5.2, w: fit("umap_hits.png", 3.7), h: 3.7 });
  frame(3, `${Math.round(I.hits_per_quarter[main])} vs ${Math.round(I.hits_per_quarter[rule])} hits per quarter`, [
    { text: `Out of ${K} slots: model ${pct(P[main].value)}%, best rule ${pct(P[rule].value)}%, random ${pct(P.random.value)}%`, options: { breakLine: true } },
    { text: "Tested on 2023 serials the model never saw" },
  ]);
  frame(4, `About ${usdK(M.extra_revenue_per_year)} a year`, [
    { text: `${M.ratio_model_to_rule}x the reading from the same ${K} slots`, options: { breakLine: true } },
    { text: `Range ${usdK(M.low)} to ${usdK(M.high)} depending on slot lift and ad rates` },
  ]);
  frame(5, "Pilot: A/B test half the slots for one quarter", [
    { text: "Score at chapter 3, re-rank at day 30, retrain yearly", options: { breakLine: true } },
    { text: TEAM, options: { fontSize: 16 } },
  ]);
  return pres;
}

(async () => {
  for (const [pres, file] of [[buildDeck(), "royalroad_breakout_draft.pptx"], [buildStoryboard(), "video_storyboard.pptx"]]) {
    const f = path.join(OUT, file);
    await pres.writeFile({ fileName: f });
    await applyTheme(f, THEME);
    console.log("wrote", f);
  }
  const words = VIDEO_SCRIPT.join(" ").split(/\s+/).length;
  const md = ["# 2-minute video script", "", "Generated by `reports/slides/build_deck.js` from `modeling/out/*.json`; the frames are in `video_storyboard.pptx` (speaker notes hold the same lines).", "",
    `Frame 3 needs a screen recording of the atlas (https://godspeedhuang.github.io/royalroad-breakout/): open it, find "${ex.title}", and show the hits around it.`, "",
    ...VIDEO_SCRIPT.flatMap((t, i) => [`## Frame ${i + 1} (${TIMES[i]})`, "", t, ""]), `Word count: ${words}`, ""];
  fs.writeFileSync(path.join(OUT, "video_script.md"), md.join("\n"));
  console.log("wrote video_script.md,", words, "words");
})();
