# Royal Road Breakout: project context for Claude

MIT 15.071 The AI Edge (Fall 2026) Project 1, team of three. **Due 2026-11-20.**
Course requirements: `docs/project-brief.md`. Modeling design: `modeling/README.md`. Public overview: `README.md`.

## In one sentence

The stakeholder is Royal Road's recommendation team: many new serials, few recommendation slots. Using only what is available when chapter 3 is published (text of the first three chapters, tags, whether the author is writing a first book), predict whether a book ends up in the top 10% of followers among books first released in the same quarter. The result that matters most is for **first-time authors**, who have no earlier books to go on.

The deck's three acts: (1) new authors get buried → (2) how many more future hits the model picks than a baseline → (3) does taste drift, and how often must the model be retrained. **The main model is the lead; the taste study is supporting.**

## Status

| Part | Status |
|---|---|
| EDA (`eda/`) | Done |
| Embeddings and taste-drift analysis (`embedding/`) | Done |
| Public Atlas (`docs/atlas/`) | Live: https://godspeedhuang.github.io/royalroad-breakout/ (main branch /docs, redeploys on push) |
| Main prediction model (`modeling/`) | First version done: rules, LR, gradient boosting, MLP. Main model `lr_SAT`, 2023 test year AUC 0.887, Recall@slots 0.482 (best rule 0.261) |
| AUC drift test (fixed test year, training data of different ages) | Done: each extra year of training-data age costs 0.026 AUC for tags, 0.008 for text |
| Covers (`cover/01`, `02`, teammate Yinqi Yu on Colab) | Cover status, embeddings from 4 frozen image models, ablation done: on valid-cover books only, adding CLIP ViT-L/14 to S+A+T takes Recall@slots from 0.289 to 0.360. Results in `cover/out/`; cover embeddings on the teammate's Google Drive |
| Covers merged into the main model (`cover/03_cover_merge.ipynb`) | Done. Plan 1: on the same valid-cover test books, the main model trained on all books already scores 0.348, so covers must be compared against 0.348, not 0.289. Plan 2 (two-stage merge): covers add 4.0 points among valid-cover books in 2023 (significant) and 0 in 2024; merged into the ranking of all books there is no gain (0.482 vs 0.481 in 2023, −2.1 in 2024) because the cover adjustment drifts upward on newer books. Recommendation (team has not decided): no covers in the main model; report covers as exploratory |
| Walkthrough (`walkthrough/01–06`) and experiment log (`EXPERIMENTS.md`) | 01–05 done; 06 waits for SyuanBo's code for 50 slots, day-30 re-rank and revenue |
| Deck, video | Draft deck exists (10 slides); its numbers for 50 slots, day 30 and revenue have no code in the repo yet. Video not started |

## Key facts about the data

- Source: HF `OmniAICreator/RoyalRoad-1.61M`, one row per chapter, 86,959 books, 57,997 with at least 3 chapters.
- `followers`, `total_views`, `average_views` are a **single snapshot** from about 2025-06-16, identical on every chapter of a book; there is no growth curve.
- The five score columns and `pages` are entirely empty; `status` is 82.5% empty.
- Older books have more followers (books out for under 3 months reach the top 10% 2.7% of the time, over 3 years 12.8%), so Y must be a **within-quarter rank**.
- Project sample: first released 2016–2024 with at least 3 chapters → 45,245 books, 135,735 chapters.

## Decisions already made (do not reopen)

- Dataset: Royal Road, not AniList or Bangumi.
- Y = top 10% of followers within the release quarter (`top10`).
- Only the first three chapters: later chapters are written after a book takes off, which would reverse cause and effect.
- One vector per chapter; chapters over 8,000 tokens are split evenly and recombined by token-weighted mean; at most 20,000 tokens for a book's first three chapters.
- Book vector = token-weighted mean of the three chapters, L2-normalized. The blurb gets its own vector and is not mixed into the book vector.
- Model `text-embedding-3-small` (1,536-d) through MIT Parley's OpenAI-compatible API. Parley has no Batch API, so requests run in parallel.
- Vectors are stored as parquet, no vector database.
- Trend conclusions use numbers computed in 1,536 dimensions; 2D plots are for illustration only. PCA for trajectories, UMAP for clusters, no t-SNE.
- Visualization tool: Apple Embedding Atlas (`uvx embedding-atlas`), not SQLRooms.
- Author history is computed from `author_url` in the data; author pages are not scraped.
- Author features are only "number of earlier books" and "first book or not" (counting only books first released earlier). Earlier books' followers are a 2025 snapshot and a new hit pulls readers back to them, so "earlier book reached the top 10%" is flagged as leak-prone and raw follower counts of earlier books are not used. "Experienced authors first" is a business-rule baseline, and results are reported separately for first-time and returning authors.

## Known pitfalls

- **Watermarks**: about 6,507 books contain anti-piracy sentences in chapter text (mentioning Amazon or Royal Road); 75% of 2016 books, 10–15% in other years. The apparent "2016 → 2017 jump" was almost entirely caused by them. Checked for modeling: models train from 2017, and dropping watermarked books from the test year leaves the main model's AUC unchanged (0.889 vs 0.887), so the embeddings do not need to be redone for modeling. Matching is by substring and may flag books that mention these words normally.
- **Cite the watermark-free drift numbers**: the preference direction eight years apart has similarity 0.935 (`watermark_check.json`), not the old 0.85. Numbers in `drift_summary.json`, `direction_summary.json` and `reports/artifacts/workflow*.html` are from before watermarks were removed.
- **Blurbs leak**: blurbs are the 2025 version. 7.8% of test-year blurbs mention Amazon, Kindle, Patreon, Audible, "Book 2" and the like, and 45.5% of those books are hits. So the blurb vector (D) and earlier books' success (P) stay out of the main model and serve only as an upper bound (recall 0.573 with them). Tags and chapter text may also have been edited later; this dataset cannot recover earlier versions.
- **Covers leak**: `time=` in the cover URL is the upload time of the current cover. Books whose cover appeared or changed after chapter 3 have a 21% hit rate in 2017–2022, against 7% for valid covers (uploaded before chapter 3 and never changed). So a cover model can only be trained on valid covers; filling zeros for books without a valid cover and adding covers to the main model would leak this status. Valid covers are almost all from 2020 onward (earlier URLs have no `time=`).
- Supply and hits move in almost the same direction in 1,536 dimensions (cosine 0.90). "The market moves one way and hits do not follow" is an artifact of the PCA plot; do not say it.
- PCA 2D explains only 10.4% of the variance.
- 2024 has the shortest observation time (0.5–1.5 years) and is only supplementary; 2023 (1.5–2.5 years) is the test year. 2016–2017 have only 52–60 top-10% books per quarter.
- The falling lift of the Progression tag reflects the tag spreading later, not readers changing their minds.
- `12_explain_direction.py` only produces the watermark-free version (`*_clean`); the concept probes' point estimates and confidence intervals use different estimators (equal-weight mean across years vs. pooled bootstrap). `--plot-only` redraws its figure from the saved JSON without the data or the API.
- Passing `--labels` to Embedding Atlas errors out; `atlas_pca_labels.parquet` is unused.

## A teammate's earlier results

A teammate built an earlier version on the same dataset (not in this repo): 13,998 books from 2021–2023 with at least 5 chapters; metric Recall@10% with 314 slots; logistic regression (structured + title and blurb + body text) about 50%, rule baselines 16–38%; title and blurb added a significant 6.1 points, body text and covers were not significant. This repo's models were written from scratch and do not build on that version; differences are in "Known limitations" in `modeling/README.md`.

## Files and data

- In `embedding/out/` only JSON, CSV and figures are in git. `data/` and all `.parquet` files are gitignored; folders are kept with `.gitkeep`. Exception: `docs/atlas/*/data/dataset.parquet` (needed by Pages). Large files are shared privately with teammates through the cloud; the list is under "Large files" in the README.
- On the original author's machine all data lives in this folder (moved on 2026-11-11 from the old folder `../final project 1/royalroad/`, not a symlink): `data/raw` (12 GB raw data), `data/interim`, `embedding/out/*.parquet`, `embedding/out/emb_parts` (raw vectors from 3,129 requests), about 17 GB in total. The old folder only holds old copies of the code. Other machines only have what was downloaded from HF.
- `books.parquet` (826 MB) is the modeling input. It is shared with `books_meta` and `atlas*.parquet` on the public HF dataset (`embedding/14_publish_hf.sh`), author names included, by the user's decision: the upstream dataset is public anyway. `chapters*.parquet` contain full chapter text and are not uploaded; rebuild them with `01_prepare.py`.
- Cover files from the teammate (`cover_status.parquet`, `cover_emb_*.parquet`) go in `data/cover/`, not in git. Without `cover_status.parquet`, `cover/03_cover_merge.ipynb` uses the book list in `cover/out/clusters/cover_clusters.parquet` as the valid-cover list (clustering used valid covers only). Cover images belong to their authors and must not go into the repo; `cover/out/clusters/cluster_*.png` are thumbnail sheets and are gitignored.
- `walkthrough/` notebooks only read JSON, CSV and figures from the `out/` folders and `reports/eda/`; they do not recompute and run in seconds. Rerun them when upstream results change: `cd walkthrough && for f in 0*.ipynb; do uv run jupyter nbconvert --to notebook --execute --inplace "$f"; done`.
- `embedding/.env` holds the Parley key and is not in git. Do not read it or print it.
- `docs/` is the GitHub Pages content. `docs/atlas/` is produced by `embedding/13_export_atlas.sh`; do not edit it by hand. The public version drops author names and limits excerpts to 300 characters. Exception made on 2026-11-17: the Chinese labels in `docs/atlas/*/data/dataset.parquet` (`kind`, `top10`, centroid titles) were relabeled in place to the English labels the current code produces; nothing else changed.

## Conventions

- Use uv: `uv sync`, `uv run python ...`, `uv add <pkg>`. Do not use pip.
- Scripts are numbered with two digits in run order; paths are always built from `Path(__file__).parent`.
- Talk to SyuanBo in Traditional Chinese, keeping technical terms in English. Everything in the repo is in English: all READMEs, `EXPERIMENTS.md`, this file, code, comments, notebook text and outputs, and figure labels. Notebooks use short one- or two-sentence notes instead of large headings.
- Chinese that intentionally remains: the teammate's Google Drive folder name `封面向量_V区块_交接`; the legacy-label mappings (written as `\u` escapes) in `07_atlas_pca.py`, `09_pca_trajectory.py` and `13_export_atlas.sh`, because the `atlas*.parquet` files on HF still store the old Chinese labels; book titles, blurbs and chapter text in the data (authors' original content). Figures from `eda/01_eda.ipynb` (`reports/eda/figs/`, also embedded in that notebook and in `walkthrough/01`) still have Chinese labels because rerunning the EDA needs the 12 GB raw data. The text outputs of `eda/01` and `cover/01–02` were translated in place (numbers unchanged) because those notebooks cannot be rerun here.
- New numbers must be reproducible from `embedding/out/*.json` or code, not only stated in conversation.
- Documents do not use em dashes (— or ——); use parentheses, commas, colons or separate sentences. En dashes in number ranges (–) are fine.

## After changing code: check the documents

Every time code changes (scripts added, renamed or deleted, parameters changed, new results), read each document below before reporting done and update anything out of date. When reporting, list which documents changed and which were read and needed no change.

| Document | Update when |
|---|---|
| `README.md` | Commands, file locations, sample sizes, reproduction steps or headline results change |
| `CLAUDE.md` | Status table, decisions, known pitfalls, or file and data notes change |
| `modeling/README.md` | Feature, split, metric or baseline design or results change |
| `cover/README.md` | The cover pipeline, file locations or results change |
| `EXPERIMENTS.md` | An experiment is added or finished, or a decision changes |
| `walkthrough/*.ipynb` | Rerun when upstream JSON, CSV or figures change; check numbers written into the text (e.g. takeaways) |
| `docs/index.html` | The Atlas version, sample size or page text changes |
| `docs/project-brief.md` | Only when the course requirements change |

- Numbers in documents follow program output, not memory.
- When adding a `.md` file or a README for a subfolder, add it to this table.
