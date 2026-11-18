# Experiments log

Everything we tried, what happened, and what we decided. Numbers are on the 2023 test year (8,075 serials, 811 future top-10% hits) unless noted.

**Ground rules for every row.** Train on 2017–2021, pick hyperparameters and the main model on 2022, retrain on 2017–2022, score 2023 once (2024 is a secondary check because its serials have had little time to collect followers). Recall@slots gives each quarter as many slots as it has true hits; random scores 0.10. Differences use a paired bootstrap that resamples serials within each quarter (500 draws, 95% CI). "pp" means percentage points of Recall@slots.

## Main model

| # | Question | What we tried | Result | Decision | Where |
|---|---|---|---|---|---|
| 1 | Does reading the chapters help? | Logistic regression on tags + author (S+A), then add the first-3-chapter embedding (T) | +8.4 pp (CI +4.8 to +11.2); AUC 0.814 → 0.887 | Keep the text | `modeling/01_ml.py`, `03_compare.py` |
| 2 | Does author experience help? | Tags (S) vs tags + "first book or not" and number of earlier books (S+A) | +1.1 pp (CI −0.2 to +3.0), not significant | Keep: cheap, and needed to report first-time authors separately | same |
| 3 | Is a nonlinear model better? | Gradient boosting (S+A + 32 PCA components of T) and a PyTorch MLP on S+A+T, same tuning protocol | Boosting 0.476 (−0.6 pp, CI −3.8 to +1.8); MLP 0.493 (+1.1 pp, CI −0.8 to +2.8). Validation AUC: LR 0.877, MLP 0.876, boosting 0.862 | Main model = logistic regression (chosen on 2022, simplest and explainable) | `modeling/01_ml.py`, `02_dl.py` |
| 4 | Does the headline depend on tags? | Text embedding only (T) | 0.406 (CI 0.376–0.440) vs best rule 0.261 | Headline holds without tags | `modeling/03_compare.py` |
| 5 | How good are simple editorial rules? | Popular genres first, fastest updates first, experienced authors first | 0.261, 0.170, 0.148 | Benchmark = popular genres first | `modeling/01_ml.py` |

## Inputs we excluded (leakage)

| # | Input | What we found | Decision | Where |
|---|---|---|---|---|
| 6 | Blurb embedding (D) | +5.8 pp (CI +3.2 to +8.5), but blurbs are the 2025 version: 7.8% of test blurbs mention Kindle, Patreon, Audible or "Book 2", and 45.5% of those are hits | Excluded; upper bound only (all inputs: 0.573) | `modeling/03_compare.py` |
| 7 | Earlier books' success (P) | +3.3 pp (CI +1.5 to +5.5), but a new hit pulls readers back to the author's earlier books | Excluded; upper bound only | same |
| 8 | Covers uploaded after chapter 3 | 21.2% of these serials are hits vs 7.0% for covers uploaded before chapter 3 and never changed (2017–2022) | Only "valid" covers may be used | `cover/01_cover_pipeline.ipynb` |
| 9 | Anti-piracy watermarks in chapter text | 9.7% of test serials; dropping them leaves AUC at 0.889 (was 0.887) | No re-embedding needed | `embedding/11_watermark_check.py` |

Known limit we cannot fix with this dataset: tags, chapter text and followers are a single 2025 snapshot. Authors may add tags after a book takes off (the few 2016 serials tagged Progression, a tag that spread later, are hits five times as often), and training labels are measured in 2025, so the backtest is not a strict as-of-chapter-3 replay.

## Covers

| # | Question | What we tried | Result | Decision | Where |
|---|---|---|---|---|---|
| 10 | Do covers help, on serials with a valid cover? | Frozen CLIP ViT-L/14, CLIP ViT-B/32, DINOv2, ResNet-50 and simple visual statistics, added to S+A+T; all models trained and tested on valid-cover serials only | CLIP ViT-L/14: 0.289 → 0.360 (+7.1 pp); no gain in 2024 | Superseded by row 11: the 0.289 baseline was trained on too few serials | `cover/02_cover_ablation.ipynb` |
| 11 | Was the 0.289 baseline fair? | Same S+A+T, trained on all serials, scored on the same valid-cover test serials | 0.348 (+5.9 pp over 0.289, CI +0.4 to +9.6). Valid-cover serials are only 5.6% hits, so lift over random is 6.2x vs 4.8x on all serials | Compare covers against 0.348 | `cover/03_cover_merge.ipynb` |
| 12 | Can covers improve the model for all serials? | Two-stage: main model on all serials; a cover adjustment trained on valid-cover serials only, with the main score as a fixed offset and zero mean adjustment | Valid-cover serials: +4.0 pp in 2023 (CI +1.2 to +8.4), 0 in 2024. All serials: 0.482 → 0.481 in 2023, −2.1 pp in 2024 (the adjustment drifts upward on newer covers) | Do not add covers to the main model | same |
| 13 | Fine-tune the image model? | Not tried: only 327 future hits among valid-cover training serials | Too likely to overfit | n/a |

## Retraining and drift

| # | Question | What we tried | Result | Decision | Where |
|---|---|---|---|---|---|
| 14 | How fast does a model go stale? | Fix the test year, train on 2-year windows of increasing age (5,000 serials each) | Each extra year of age: −0.026 AUC for tags, −0.008 for text. The tags drop is mostly a step when the training window predates 2020–21 (tag usage changed) | Retrain at least yearly; rely on text | `modeling/04_drift.py` |
| 15 | Do reader preferences drift? | Direction "top 10% minus others" in the 1,536-d embedding, compared across years, corrected for sampling noise, watermarks removed | Cosine similarity 0.935 eight years apart; market supply and hits move together (cosine 0.90) | Describe as "what wins within a cohort is stable"; snapshot overlap may overstate it | `embedding/05_drift.ipynb`, `11_watermark_check.py` |

## Pending (in the slides, code not yet in this repo)

| # | Analysis | Status |
|---|---|---|
| 16 | Hits per slot with a fixed 50 slots per quarter (53% → 68%, 34 vs 14 per quarter) | Waiting for code |
| 17 | Day-30 re-rank with the first month's posting rhythm (68% → 76%) | Waiting for code |
| 18 | Ad revenue and editor time (about $39k and $36k a year) | Waiting for code |

## Ideas not tried yet

- Gradient boosting on the full 1,536-d embedding instead of 32 PCA components (the early-fusion setup in Lecture 15), and a late-fusion model that stacks separate tags and text scores.
- The chapter-1 vector (`chap1_vec`) alone or alongside the 3-chapter vector: it is already built but unused.
- Interpretable cover features from CLIP text prompts (for example "a professional book cover with title text").
- Choosing the number of slots from a cost-benefit rule (keep adding a slot while the next serial's hit probability times the value of a hit exceeds the cost of a slot).
