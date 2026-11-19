# modeling

The Project 1 main model: rule baselines → machine learning → deep learning, plus a drift test.

## How to run

Needs `embedding/out/books.parquet` and `embedding/out/watermark_books.csv`. The whole folder runs in about 3 minutes (Mac, CPU).

```bash
cd modeling
uv run python features.py     # self-check: split sizes, author history does not peek ahead, metrics are right
uv run python 01_ml.py        # rules, logistic regression, gradient boosting
uv run python 02_dl.py        # PyTorch MLP
uv run python 03_compare.py   # comparison, confidence intervals, groups, leak checks, impact
uv run python 04_drift.py     # fixed test year, training data of different ages
uv run python 05_demo_data.py # data for the demo page (docs/demo/data.js)
```

| File | Output |
|---|---|
| `features.py` | Shared: feature blocks, time split, metrics |
| `01_ml.py`, `02_dl.py` | `out/preds_*.parquet` (every model's score for every book, not in git), `out/selection_*.json` (chosen hyperparameters and validation scores) |
| `03_compare.py` | `out/results.json`, `out/results.md`, `out/figs/model_comparison.png` |
| `04_drift.py` | `out/drift_auc.json`, `out/figs/drift_auc.png` |
| `05_demo_data.py` | `../docs/demo/data.js`: every 2023 and 2024 book's score under each model, with title, tags, cover URL, genre tint and outcome; 20 serials per 2016 quarter and each quarter's hit threshold for the target animation; tag and warning lists; the hyperparameter grids from `selection_*.json`; two example first-three-chapter vectors and the hit direction (2017–2022, no watermarks); a 4,000-serial PCA/UMAP sample with yearly paths and the 1,536-d 2017→2024 shift cosine; screen 0's numbers (including the hits' share of 2023 views and followers) and cover wall; the top pick's first lines (≤ 200 characters, from `atlas.parquet`) with their cl100k_base tokens and its chapter-1 vector; two standardized raw dimensions of the map sample; the cover model's scores from `cover/out/preds_cover.parquet` when present (checked against `cover_merge.json`); views per serial for the business screen; the drift, tag-lift and cover-experiment numbers the page charts. Prints the 50-slot numbers and the ad revenue used in the deck |

## Design

- **Unit**: one book per row, from `embedding/out/books.parquet` (45,245 books first released 2016–2024 with at least 3 chapters).
- **Y**: `top10`, whether followers are at or above the 90th percentile among books first released in the same quarter. Comparing within the quarter removes the "older books have more followers" effect.
- **Split (by time)**:

| | Years | Books | Use |
|---|---|---|---|
| train | 2017–2021 | 17,100 | Training |
| val | 2022 | 5,700 | Hyperparameters and choosing the main model |
| test | 2023 | 8,075 | Scored once at the end |
| recent | 2024 | 12,165 | Supplementary. Observed for only 0.5–1.5 years, so its top 10% is not settled |

  2016 is not used (75% of its books carry watermarks). After tuning, models are refit on 2017–2022 and scored on 2023 and 2024. An author's older book in train and newer book in test is allowed: in production the author's earlier books are visible too.

- **Metrics**: AUC, AP, and **Recall@slots**. Each quarter gets as many slots as it has true hits (about 10%, roughly 203 a quarter in 2023); Recall@slots is the share of those slots that go to a true hit. Random picks score 0.10.
- **Feature blocks** (the letters appear in model names, e.g. `lr_SAT`):

| Code | Content | Dims | Usable? |
|---|---|---|---|
| S | tags, warnings, `is_original`, length of chapters 1–3 and days between them | 75 | Yes |
| A | author experience: first book or not, number of earlier books | 2 | Yes |
| T | first-three-chapter vector `book_vec` | 1,536 | Yes |
| P | whether an earlier book reached its quarter's top 10% | 1 | **Leak-prone**: earlier books' followers are a 2025 snapshot, and a new hit pulls readers back to them |
| D | blurb vector `desc_vec` + blurb length | 1,537 | **Leak-prone**: the blurb is the 2025 version, see below |

  Not usable as features (only known afterwards): `followers`, `total_views`, `average_views`, `favorites`, `ratings`, `n_chapters`, `pct_in_quarter`, `log_followers`, `status`, `cover_url`. Raw follower counts of earlier books are not used either.

- **The main model is not chosen on test scores**: among the three leak-free candidates (`lr_SAT`, `gb_SAT`, `mlp_SAT`) the one with the highest validation AUC wins, which is `lr_SAT`.

## Results (2023 test year, 8,075 books, 811 hits)

![model comparison](out/figs/model_comparison.png)

| Model | AUC | Recall@slots (95% CI) | 2024 |
|---|---|---|---|
| Rule: fastest updates first | 0.625 | 0.170 (0.144–0.196) | 0.195 |
| Rule: experienced authors first | 0.565 | 0.148 (0.127–0.176) | 0.177 |
| Rule: popular genres first | 0.742 | 0.261 (0.239–0.295) | 0.282 |
| LR: S | 0.810 | 0.387 (0.359–0.420) | 0.405 |
| LR: S+A | 0.814 | 0.398 (0.372–0.436) | 0.414 |
| **LR: S+A+T (main)** | **0.887** | **0.482 (0.455–0.515)** | 0.453 |
| Gradient boosting: S+A+T | 0.865 | 0.476 (0.445–0.503) | 0.470 |
| MLP: S+A+T | 0.886 | 0.493 (0.466–0.527) | 0.456 |
| LR: T only | 0.856 | 0.406 (0.376–0.440) | 0.377 |
| MLP: T only | 0.858 | 0.425 (0.395–0.457) | 0.388 |
| LR: S+A+T+P † | 0.896 | 0.515 (0.491–0.548) | 0.490 |
| LR: S+A+T+P+D † | 0.918 | 0.573 (0.547–0.603) | 0.566 |
| LR: D only † | 0.873 | 0.506 (0.481–0.538) | 0.491 |

† Leak-prone; upper bound only. Full numbers in `out/results.json`.

**What each step adds** (Recall@slots, percentage points, paired bootstrap 95% CI):

| Comparison | Difference | 95% CI | Reading |
|---|---|---|---|
| Main model vs best rule | +22.1 | +18.2 to +25.3 | Model clearly beats the rule |
| Add author experience (S → S+A) | +1.1 | −0.2 to +3.0 | Not significant |
| Add chapter text (S+A → S+A+T) | +8.4 | +4.8 to +11.2 | Significant: the text helps |
| Gradient boosting vs LR | −0.6 | −3.8 to +1.8 | No difference |
| MLP vs LR | +1.1 | −0.8 to +2.8 | No difference |
| Add earlier books' success († P) | +3.3 | +1.5 to +5.5 | Significant, but leak-prone |
| Add blurb († D) | +5.8 | +3.2 to +8.5 | Significant, but leak-prone |

**By author group** (main model `lr_SAT`):

| Group | Books | AUC | Recall@slots | Best rule |
|---|---|---|---|---|
| All | 8,075 | 0.887 | 0.482 | 0.261 |
| First-time authors | 5,383 | 0.878 | 0.430 | 0.246 |
| Returning authors | 2,692 | 0.894 | 0.536 | 0.317 |

First-time authors are harder to call, but the model still beats the rule by a wide margin.

**Impact** (2023: about 2,019 new books and 203 slots per quarter on average):

| Ranking | Future hits per quarter | Eventual followers of the picks |
|---|---|---|
| Random | 20 | about 42k |
| Popular genres first | 53 | about 118k |
| Main model | 98 | about 238k |
| Perfect foresight | 203 | about 372k |

Followers are an after-the-fact number, used only to show whether slots went to the right books; they are not an effect caused by recommending.

## Drift (`04_drift.py`)

![drift](out/figs/drift_auc.png)

Fix the test year and train on data of different ages (2-year windows, 5,000 books each). Within the same test year, each extra year of training-data age costs:

- Tags and other structured features only: 0.026 AUC.
- First-three-chapter text only: 0.008 AUC.

Rules built on tags go stale fast; the text signal lasts longer. Part of the tags decline is the tagging system itself changing (Progression, for example, only became common later), not only reader taste. Recommendation for the team: retrain at least once a year.

## Leak checks

- **The blurb (D) leaks.** In the test year, 7.8% of blurbs mention Amazon, Kindle, Patreon, Audible, "Book 2" or Discord. 45.5% of those books are hits (10% overall), and the flag alone has an AUC of 0.652. Authors mostly add these words after a book takes off, so D stays out of the main model. Dropping those books lowers the blurb-only model's AUC from 0.873 to 0.837.
- **Watermarks do not affect the model.** 9.7% of test books contain watermarks; 11.1% of them are hits (about the overall rate), and the flag alone has an AUC of 0.506. Dropping them gives a main-model AUC of 0.889 (was 0.887), so there is no need to redo the embeddings for modeling.
- **Not ruled out**: tags and chapter text are also 2025 versions, and authors may have edited them later (added tags, author notes in chapters). This dataset cannot recover the earlier versions.

## Known limitations

- On the deep learning side the text encoder is frozen; only the MLP on top is trained, and it does no better than LR. Going further would mean fine-tuning a text model (`chapters.parquet` has the text), an order of magnitude more expensive.
- Author experience counts only books in the sample (2016–2024, at least 3 chapters); earlier books, or books with fewer than 3 chapters, are not counted.
- A teammate's earlier version (2021–2023, at least 5 chapters, 13,998 books) reported Recall@10% of about 50% using titles and blurbs. Here it is 48% without blurbs and 57% with them, the same order of magnitude, but the samples and splits differ, so the numbers are not directly comparable.
- 2016–2017 have only 52–60 top-10% books per quarter; books with 0 followers tie in the within-quarter ranking.
- Gradient boosting is not exactly reproducible across machines: tiny floating-point differences in the PCA step change which of four near-tied settings wins on 2022. Rerun on another machine (Apple M5 Pro, scikit-learn 1.9.1) it picked learning rate 0.03 with 31 leaves and scored 0.465 instead of 0.476; still not better than LR. The rules, all LR models and the MLP reproduce exactly.
- The editors' rule has many tied scores (books with the same tags score the same), so its hit count can move by one book with the tie-break. The demo breaks ties by book id.

## Covers

All cover experiments are in `cover/`; see `cover/README.md`. Covers show some signal among books with a valid cover in 2023 (+4.0 points) but none in 2024, and merging them into the model for all books does not help (0.482 vs 0.481), so the main model stays S+A+T. Everything we tried and decided is in `EXPERIMENTS.md` at the repo root.
