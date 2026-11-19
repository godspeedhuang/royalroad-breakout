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
| 驗證年改進實驗（`05_experiments.py`，2026-11-16） | 完成，只看 2022 驗證年。維度、三章並列、只用第一章、鄰居特徵、文風統計、連續目標、排序 loss 都沒有穩定贏主模型；只有「三章等權平均」小贏（驗證 AUC 0.8775 → 0.881），還沒採用。細節在 `modeling/README.md`「驗證年實驗」 |
| 第 30 天版本（`06_day30.py`，2026-11-16） | 完成：主模型加前 30 天更新節奏。驗證年 Recall@名額 0.454 → 0.506，但測試年 2023 只有 0.482 → 0.490（+0.7，不顯著），2024 年 0.453 → 0.516（2024 標籤不穩，可能高估）。當第二輪確認，不取代主模型。換成每季 50 個版位時，2023 是 68% → 76%（+8.0，CI +0.5 到 +14.0），名額越少差距越明顯 |
| 商業影響（`07_impact.py`，2026-11-16） | 完成：每季固定 50 個版位（對應 Rising Stars 每類型最多 50 本）。2023 模型挑中 68%、規則 28%（每季 34 對 14 本）；版位上的書累積瀏覽量是規則的 5.5 倍；base case（lift 10%、CPM $3）每年多約 $3.9 萬廣告收入，範圍 $1.3 萬到 $13 萬；編輯初篩一年省約 1,200 小時（補充）。簡報主數字用這裡的 50 個版位，README 繼續報 Recall@名額 |
| 簡報、影片 | `reports/slides/royalroad_breakout_draft.pptx`：10 頁 pitch，英文，標題都是一句 statement，有講者備註；黑白灰加一個橘色，不用「·」。`video_storyboard.pptx`：2 分鐘影片 6 個畫面，逐字稿在 `video_script.md`（約 224 字）。三份都由 `reports/slides/build_deck.js` 讀 `modeling/out/*.json` 產生。還沒做：第 3 頁封面範例圖（等隊友）、影片的 Atlas 螢幕錄影和旁白錄音（誰錄未定） |

## 連結

- GitHub（公開）：https://github.com/godspeedhuang/royalroad-breakout
- Atlas（GitHub Pages）：https://godspeedhuang.github.io/royalroad-breakout/ ，第一次載入約 110 MB、要等一分鐘左右
- HF dataset（公開）：https://huggingface.co/datasets/syuanbo/royalroad-breakout ，放 `books`、`books_meta`、`atlas`、`atlas_pca` 四個 parquet，路徑和 repo 一樣
- GitHub 帳號是 `godspeedhuang`，HF 帳號是 `syuanbo`，兩個不同，不要互換。

## 待辦（2026-11-13 整理）

- [ ] 簡報 5–10 頁、2 分鐘影片，對象是 Royal Road 推薦團隊（11/20 截止）
- [ ] README 縮回一頁：課程要「one-page README」，現在約 1,350 字。embedding 細節可移到 `embedding/README.md`（要加進下面的文件表）
- [ ] 隊友：把兩位加成 GitHub collaborator（不加的話信裡要改成「先 fork 再發 PR」），再寄說明信。2026-11-11 擬好的信：連 repo 首頁、`uv sync`、README 的「Large files」、`modeling/out/results.md`、Atlas 入口頁；向量成本寫 about $7（3.55 億 token），不是 $8
- [ ] Does it matter：現在只換算成「每季多挑中幾本熱門書」（98 對規則 53），還沒換成營收或讀者時間

模型方面可以再試、但不是必要的：

- 三章等權平均取代 token 加權平均（`05_experiments.py` 驗證年小贏，改 `features.py` 的 T 後要重跑 01–04 並更新文件數字）
- 近年資料加權、LR／MLP／排序 loss 的 ensemble：還沒測
- 作者經驗只算樣本內的前作（2016–2024、至少 3 章），可從 `data/interim/fictions_47.parquet` 補完整
- fine-tune 文字模型（`chapters.parquet` 有原文）。凍結向量上的 MLP 跟 LR 沒差，先不做
- LLM 評分開頭（structured output）：唯一可能補上「寫得好不好」的方向；有名的書 LLM 可能認得，要遮書名作者並檢查冷門書
- 已經試過、不要再試：換 `text-embedding-3-large`、三章並列、只用第一章、鄰居特徵、文風統計、連續目標（同季排名）、依排名配對的排序 loss（見 `modeling/README.md`）
- 測試年目前是 2023。原本的設計寫「2023–2024 不適合當測試年」，後來只排除 2024；要改回 2022 只動 `modeling/features.py` 的 `TRAIN_YEARS / VAL_YEAR / TEST_YEAR`

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

- `embedding/out/` 裡只有 JSON、CSV、圖進 git。`data/` 和所有 `.parquet` 都被 gitignore，資料夾用 `.gitkeep` 保留；例外是 `docs/atlas/*/data/dataset.parquet`（Pages 要用）。要花錢重算的向量放在公開的 HF dataset，下載指令在 README 的「Large files」；其他大檔用程式重建，不用 Google Drive。
- 資料都在這個資料夾裡（2026-11-11 從舊資料夾 `../final project 1/royalroad/` 搬過來，不是 symlink）：`data/raw`（12 GB 原始資料）、`data/interim`、`embedding/out/*.parquet`、`embedding/out/emb_parts`（3,129 個請求的原始向量），合計約 17 GB。舊資料夾只剩程式碼的舊副本。
- `books.parquet`（826 MB）是建模的輸入。它和 `books_meta`、`atlas*.parquet` 放在公開的 HF dataset 分享（`embedding/14_publish_hf.sh`），作者名一起公開，使用者決定的：上游資料集本來就公開。`chapters*.parquet` 有章節全文，不上傳，要用就跑 `01_prepare.py` 重建。`modeling/05_experiments.py` 要用 `chapters.parquet` 的逐章向量和原文；它由 `03_pool_project.py` 產生，需要本機的 `emb_parts`，所以隊友沒辦法直接重跑 05。
- `embedding/.env` 放 Parley 金鑰，不進 git。不要讀它、不要印出來。
- `docs/` 是 GitHub Pages 的內容。`docs/atlas/` 由 `embedding/13_export_atlas.sh` 產生，不要手改。公開版拿掉了作者名，節錄限 300 字。

## 慣例

- 用 uv：`uv sync`、`uv run python ...`、`uv add <pkg>`。不要用 pip。
- 腳本用兩位數編號表示執行順序；路徑一律從 `Path(__file__).parent` 算。
- 跟 SyuanBo 對話用繁體中文，專有名詞保留英文。README 用英文（課程交付物）。
- 新的數字要能從 `embedding/out/*.json`、`modeling/out/*.json` 或程式重現，不要只寫在對話裡。
- Commit 依功能分開（環境、EDA、embedding、modeling、docs 各自一個），訊息用英文。
- 公開上傳到外部網站（HF 上傳、`gh repo create --public`）會被 auto mode 擋，給使用者完整指令讓他自己跑。
- 這個資料夾常有好幾個 Claude session 同時在改。動文件前先重讀，不要用舊內容整份覆寫。
- 文件不用破折號（— 或 ——），改用括號、逗號、冒號或拆句。數字區間的 – 照用。簡報和對外文字也不用間隔號「·」當分隔。

## 簡報與影片怎麼更新

- **金額和版位的假設只改 `modeling/07_impact.py` 最上面那一區**（版位數、lift、CPM、編輯時間與時薪），不要直接改簡報上的數字。
- 更新順序：`03_compare.py`、`06_day30.py`、`07_impact.py`（模型或資料變了才要重跑前面的）→ `uv run python reports/slides/make_figs.py`（UMAP 圖）→ `NODE_PATH=<pptxgenjs 所在的 node_modules> node reports/slides/build_deck.js`，一次產生簡報、影片分鏡和 `video_script.md`。pptxgenjs 沒放進 repo，裝在本機 scratchpad，換機器要先 `npm install pptxgenjs@3.12.0`。
- 圖片在 `reports/slides/img/`：`rr_*.png` 是 Internet Archive 2025 年 6 月的 royalroad.com（網站本身有 Cloudflare 機器人檢查，不要用 headless 繞過）；`atlas_screenshot.png` 由 `reports/slides/shot_atlas.js` 用可見的 Chrome 截（headless 沒有 WebGPU，地圖是空的），要先在 `docs/` 開 `python3 -m http.server 8071`。
- 簡報標題都寫成一句 statement（可以有數字），不要寫成「Model: xxx」這種標籤。主數字用每季 50 個版位；README 用 Recall@名額（名額 = 熱門書數，約 200），兩者在講者備註裡有說明。

## 改完程式碼之後：檢查文件

每次改完程式碼（新增、改名或刪除腳本，改參數，產生新結果），回報完成之前先把下面每份文件看過一次，過時的地方一起改。回報時列出改了哪些文件、哪些看過但不用改。

| 文件 | 什麼情況要改 |
|---|---|
| `README.md` | 指令、檔案位置、樣本數、重現步驟、主要結果變了 |
| `CLAUDE.md` | 狀態表、已做的決定、已知的坑、檔案與資料的說明變了 |
| `modeling/README.md` | 特徵、切分、指標、baseline 的設計或結果變了 |
| `docs/index.html` | Atlas 的版本、樣本數、頁面上的說明變了 |
| `docs/project-brief.md` | 只有課程要求變了才改 |
| `reports/slides/`（簡報、分鏡、`video_script.md`） | `modeling/out/` 的結果或 `07_impact.py` 的假設變了：重跑 `build_deck.js`，再用 PowerPoint 看一次版面 |

- 文件裡的數字以程式輸出為準，不要憑記憶改。
- 新增 `.md` 檔或子資料夾的 README 時，把它加進這張表。
