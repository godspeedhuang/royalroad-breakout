# Royal Road Breakout：給 Claude 的專案脈絡

MIT 15.071 The AI Edge（Fall 2026）Project 1，三人小組。**截止 2026-11-20。**
課程要求見 `docs/project-brief.md`，建模設計見 `modeling/README.md`，對外說明見 `README.md`。

## 一句話

Stakeholder 是 Royal Road 的推薦團隊：新連載很多、推薦位有限。用第 3 章發布當下拿得到的資訊（前三章文字、tags、作者是不是第一次寫書），預測一本書最後會不會進「同一季開始連載的書裡 followers 前 10%」。成績最重要的是**第一次寫書的作者**那一組，因為他們沒有前作可以參考。

簡報的三幕：(1) 新作者被淹沒 → (2) 模型比 baseline 多挑中幾本熱門書 → (3) 口味會不會漂移、模型多久要重訓。**主模型是主角，口味研究是配角。**

## 現在的狀態

| 部分 | 狀態 |
|---|---|
| EDA（`eda/`） | 完成 |
| Embedding 與口味漂移分析（`embedding/`） | 完成 |
| 公開版 Atlas（`docs/atlas/`） | 已上線：https://godspeedhuang.github.io/royalroad-breakout/（main 分支 /docs，push 後自動重新部署） |
| 主預測模型（`modeling/`） | 完成第一版：規則、LR、gradient boosting、MLP。主模型 `lr_SAT`，2023 測試年 AUC 0.887、Recall@名額 0.482（最好的規則 0.261） |
| AUC 漂移測試（固定測試年、換訓練年代） | 完成：訓練資料每舊一年，tags 模型 AUC 掉 0.026，文字模型掉 0.008 |
| 簡報、影片 | 還沒做 |

## 資料的關鍵事實

- 來源：HF `OmniAICreator/RoyalRoad-1.61M`，一章一列，86,959 本書，至少 3 章的 57,997 本。
- `followers`、`total_views`、`average_views` 是約 2025-06-16 的**單一快照**，同一本書每一章的值都一樣，沒有成長曲線。
- 五個 score 欄位和 `pages` 全空；`status` 82.5% 是空的。
- 連載越久 followers 越多（連載不到 3 個月的書進前 10% 的比例 2.7%，超過 3 年的 12.8%），所以 Y 一定要用**同季排名**。
- 本專案樣本：2016–2024 年開始連載、至少 3 章 → 45,245 本，135,735 章。

## 已經做過的決定（不要重新討論）

- 資料集用 RoyalRoad，不用 AniList、Bangumi。
- Y = 同季 followers 前 10%（`top10`）。
- 只看前三章：後面的章節是紅了之後才寫的，會倒果為因。
- 每章一個向量；超過 8,000 token 的章平均切塊再依 token 數加權合併；每本書前三章上限 20,000 token。
- 書的向量 = 三章依 token 數加權平均後 L2 正規化。簡介另外一個向量，不混進書的向量。
- 模型 `text-embedding-3-small`（1,536 維），經 MIT Parley 的 OpenAI 相容 API。Parley 沒有 Batch API，用並行請求。
- 向量存 parquet，不用向量資料庫。
- 趨勢的結論以 1,536 維算出的數字為準，2D 圖只用來說明。PCA 看軌跡，UMAP 看分群，不用 t-SNE。
- 視覺化工具用 Apple Embedding Atlas（`uvx embedding-atlas`），不用 SQLRooms。
- 作者歷史用資料裡的 `author_url` 自己算，不去爬作者頁。
- 作者特徵只放「以前寫過幾本書」「是不是第一次寫書」（只算更早開始連載的書）。前作的 followers 是 2025 年快照，這本書紅了會回頭拉高前作，所以「前作是不是前 10%」要標注有回推問題，前作的 followers 原始數字不用。「老作者優先」當 business-rule baseline，成績分成第一次寫書和有前作兩組報。

## 已知的坑

- **浮水印**：約 6,507 本書的章節含防盜句子（提到 Amazon、Royal Road），2016 年 75% 的書有，其他年 10–15%。之前看到的「2016→2017 大跳動」幾乎全是它造成的。建模時檢查過：模型從 2017 年起訓練，測試年拿掉含浮水印的書後主模型 AUC 不變（0.889 對 0.887），所以不用為了建模重做 embedding。比對用子字串，可能誤標正常提到這些字的書。
- **引用漂移數字要用去浮水印版**：偏好方向相隔 8 年的相似度是 0.935（`watermark_check.json`），不是舊的 0.85。`drift_summary.json`、`direction_summary.json` 和 `reports/artifacts/workflow*.html` 裡的數字都是去浮水印前的。
- **簡介有 leak**：簡介是 2025 年的版本。測試年 7.8% 的書簡介裡有 Amazon、Kindle、Patreon、Audible、"Book 2" 這類字，其中 45.5% 是熱門書。所以簡介向量（D）和前作表現（P）不放進主模型，只當上限參考（加上去 Recall 到 0.573）。tags 和章節內容也可能被事後改過，這個資料集無法還原。
- 供給和熱門的移動方向在 1,536 維幾乎平行（cosine 0.90）。「市場往一邊走、熱門書沒跟著走」是 PCA 圖造成的錯覺，不要這樣講。
- PCA 2D 只解釋 10.4% 的變異。
- 2024 年觀察時間最短（0.5–1.5 年），只當補充；2023 年（1.5–2.5 年）是測試年。2016–2017 年每季前 10% 只有 52–60 本。
- Progression 這個 tag 的 lift 下降，是標籤後來才普及，不是讀者變心。
- `12_explain_direction.py` 只產生去浮水印版（`*_clean`）；概念探針的點估計和信賴區間用的估計式不同（各年等權平均 vs. 合併後 bootstrap）。
- `--labels` 傳給 Embedding Atlas 會報錯，`atlas_pca_labels.parquet` 沒在用。

## 隊友已有的結果

一位隊友先前用同一個資料集做過一版（不在這個 repo）：2021–2023 年、至少 5 章的 13,998 本；指標是 314 個推薦位的 Recall@10%；logistic regression（結構化 + 標題簡介 + 內文）約 50%，規則 baseline 16–38%；標題簡介的增量顯著（+6.1 pp），內文和封面不顯著。這個 repo 的模型是從零寫的，沒有接這一版；差異寫在 `modeling/README.md` 的「已知限制」。

## 檔案與資料

- `embedding/out/` 裡只有 JSON、CSV、圖進 git。`data/` 和所有 `.parquet` 都被 gitignore，資料夾用 `.gitkeep` 保留；例外是 `docs/atlas/*/data/dataset.parquet`（Pages 要用）。大檔透過雲端私下傳給隊友，清單在 README 的「Large files」。
- 資料都在這個資料夾裡（2026-11-11 從舊資料夾 `../final project 1/royalroad/` 搬過來，不是 symlink）：`data/raw`（12 GB 原始資料）、`data/interim`、`embedding/out/*.parquet`、`embedding/out/emb_parts`（3,129 個請求的原始向量），合計約 17 GB。舊資料夾只剩程式碼的舊副本。
- `books.parquet`（826 MB）是建模的輸入。它和 `books_meta`、`atlas*.parquet` 放在公開的 HF dataset 分享（`embedding/14_publish_hf.sh`），作者名一起公開，使用者決定的：上游資料集本來就公開。`chapters*.parquet` 有章節全文，不上傳，要用就跑 `01_prepare.py` 重建。
- `embedding/.env` 放 Parley 金鑰，不進 git。不要讀它、不要印出來。
- `docs/` 是 GitHub Pages 的內容。`docs/atlas/` 由 `embedding/13_export_atlas.sh` 產生，不要手改。公開版拿掉了作者名，節錄限 300 字。

## 慣例

- 用 uv：`uv sync`、`uv run python ...`、`uv add <pkg>`。不要用 pip。
- 腳本用兩位數編號表示執行順序；路徑一律從 `Path(__file__).parent` 算。
- 跟 SyuanBo 對話用繁體中文，專有名詞保留英文。README 用英文（課程交付物）。
- 新的數字要能從 `embedding/out/*.json` 或程式重現，不要只寫在對話裡。
- 文件不用破折號（— 或 ——），改用括號、逗號、冒號或拆句。數字區間的 – 照用。

## 改完程式碼之後：檢查文件

每次改完程式碼（新增、改名或刪除腳本，改參數，產生新結果），回報完成之前先把下面每份文件看過一次，過時的地方一起改。回報時列出改了哪些文件、哪些看過但不用改。

| 文件 | 什麼情況要改 |
|---|---|
| `README.md` | 指令、檔案位置、樣本數、重現步驟、主要結果變了 |
| `CLAUDE.md` | 狀態表、已做的決定、已知的坑、檔案與資料的說明變了 |
| `modeling/README.md` | 特徵、切分、指標、baseline 的設計或結果變了 |
| `docs/index.html` | Atlas 的版本、樣本數、頁面上的說明變了 |
| `docs/project-brief.md` | 只有課程要求變了才改 |

- 文件裡的數字以程式輸出為準，不要憑記憶改。
- 新增 `.md` 檔或子資料夾的 README 時，把它加進這張表。
