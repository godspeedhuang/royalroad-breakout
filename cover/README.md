# cover

Can the cover image help predict which serials take off? Notebooks 01 and 02 were built by a teammate (Yinqi Yu) on Colab; 03 asks how covers could be merged into the main model.

**Conclusion: covers carry some signal, but it is not stable, and merging them into the main model for all serials does not help, so the main model stays S+A+T.** The analysis also confirms a leak: authors change covers after a serial takes off.

## Files

| File | What it does | Where it runs |
|---|---|---|
| `01_cover_pipeline.ipynb` | Decide whether each cover is usable → download covers → embed them with 4 frozen image models → visual statistics → self-checks → clustering | Colab (T4 GPU), reads and writes the teammate's Google Drive |
| `02_cover_ablation.ipynb` | Serials with a valid cover only: S+A+T with and without the cover vector | Colab (CPU); clones this repo and uses `modeling/features.py` |
| `03_cover_merge.ipynb` | Plan 1: why the 0.289 baseline was low. Plan 2: two-stage merge of covers into the main model for all serials | Locally: `cd cover && uv run jupyter nbconvert --to notebook --execute --inplace 03_cover_merge.ipynb` (under a minute) |

| Output | Content |
|---|---|
| `out/status/` | Cover status: counts and hit rates by year and status, upload time vs chapter 3, a 1-day grace sensitivity check, re-save check, duplicate covers |
| `out/clusters/` | CLIP ViT-L/14 clustering: silhouette, lift per cluster, a 24-cover sample sheet per cluster |
| `out/ablation/` | Results of 02 and paired bootstrap differences |
| `out/cover_merge.json` | All numbers from 03 |
| `out/preds_cover.parquet` | Per-serial two-stage scores (main + cover adjustment) for 2023 and 2024, written by 03; read by `modeling/05_demo_data.py` for the demo's "S+A+T+Cover" bar. Parquet, not in git |

**Inputs for 03** (not in git; put them in `data/cover/`): `cover_status.parquet` and `cover_emb_clip_l14.parquet`, from the teammate's Google Drive folder `RR_project1/cover_v2/封面向量_V区块_交接/` (subfolders `01_status/` and `03_embeddings/`). It also needs `embedding/out/books.parquet` (HF dataset).

## Cover status: which covers are usable

`time=` in the cover URL is the upload time of the **current** cover. Compared with the chapter-3 release time:

| Status | Meaning | Serials | Hit rate (2017–2022) |
|---|---|---|---|
| `valid` | Uploaded before chapter 3, never changed | 19,006 | 7.0% |
| `after_ch3` | Added or changed after chapter 3 | 10,590 | **21.2%** |
| `no_timestamp` | No `time=` in the URL (uploaded before April 2020), cannot tell | 7,243 | 11.6% |
| `no_custom_cover` | Platform default image | 8,304 | 1.9% |
| `image_changed` | Replaced again after the 2025-06 snapshot | 102 | 7.1% |

Serials whose cover changed after chapter 3 are hits three times as often as valid-cover serials: authors really do change covers once a serial takes off. So a cover model can only be trained on valid covers, and those are almost all from 2020 onward.

## Results

**02: valid-cover serials only** (2023 test year, 4,507 serials, 253 hits)

| Model | AUC | Recall@slots | Gain from covers (95% CI) |
|---|---|---|---|
| S+A+T | 0.858 | 0.289 | |
| + CLIP ViT-L/14 | 0.870 | 0.360 | +7.1 (+2.7 to +10.7) |
| + CLIP ViT-B/32 | 0.868 | 0.344 | +5.5 (+1.5 to +8.9) |
| + DINOv2 | 0.858 | 0.328 | +4.0, AUC unchanged |
| + ResNet-50, simple visual statistics | | 0.304, 0.289 | Not significant |

In 2024, CLIP ViT-L/14 adds nothing (0.345 vs 0.345). The clustering silhouette is only 0.02: covers do not fall into clear groups.

**03, plan 1: why 0.289 was low**

On the same valid-cover test serials, the main model trained on **all** serials already scores 0.348 (+5.9 points over training on valid covers only, CI +0.4 to +9.6). The rest of the gap is the hit rate: only 5.6% of these serials are hits (10% overall), and relative to random the model does better here (6.2x) than on all serials (4.8x). So covers should be compared against 0.348; the S+A+T+V model from 02 (0.360) is only 1.2 points higher and not significant.

**03, plan 2: two-stage merge**

The main model is trained on all serials as before. A cover module is trained on valid-cover serials only: it starts from the main score as a fixed offset and learns how much to add or subtract. Cover vectors are standardized and there is no intercept, so the adjustment averages 0 over training serials. Serials without a valid cover keep the main score.

| Comparison | 2023 | 2024 |
|---|---|---|
| Valid-cover serials: main → with cover adjustment | 0.348 → 0.387 (+4.0, CI +1.2 to +8.4) | 0.351 → 0.351 |
| All serials: main → with cover adjustment | 0.482 → 0.481 (CI crosses 0) | 0.453 → 0.432 (−2.1, significant) |
| Shuffled cover vectors | −2.4 (CI crosses 0) | |

Merging into the ranking of all serials does not help, mainly because the adjustment drifts upward on newer serials (mean +0.076 in 2023, +0.122 in 2024), pushing all valid-cover serials, the group with the lowest hit rate, up the list. Centering per quarter (a diagnostic added after seeing the results) removes most of the drift, but what remains also crosses 0: valid-cover serials hold only 31% of 2023 hits.

## Notes

- Paths in 01 and 02 are hard-coded to Colab and the teammate's Google Drive; change them to run elsewhere.
- Re-downloading covers gives different results: authors keep changing covers, so the later you download, the fewer valid covers remain.
- Cover images belong to their authors and must not go into the repo. `out/clusters/cluster_*.png` are sheets of cover thumbnails and are listed in `.gitignore`.
- Everything in these notebooks is in English. 01 and 02 ran on Colab and cannot be rerun here, so their saved log output was translated in place (numbers unchanged). The Drive folder name `封面向量_V区块_交接` is a real path and stays as is.
