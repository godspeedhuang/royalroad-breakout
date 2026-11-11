# 15.071 Project 1 要求（課程 handout 重點）

來源：15.071 The AI Edge, Fall 2026, *Team project description*。這是整理稿，不是原文。

## 時程

| 項目 | 日期 |
|---|---|
| **Project 1：Predictive AI** | **2026-11-20** |
| Project 2：Generative and agentic AI | 2026-12-08 |
| Demo Day（選出的專案展示） | 2026-12-09 |

三人一組。兩個 project 可以做不同題目，也可以同一個題目一路做下去。

## Project 1 要做什麼

用 predictive AI 解決一個商業或政策問題，二選一：

- tabular data 的 machine learning **加上** 非結構化資料的 deep learning，合成一個專案；或
- 更有企圖心的 deep learning 專案，用非結構化、可能是 multimodal 的資料。

核心工作：

1. 把問題轉成定義清楚的 prediction task，選好 inputs 和 outcome。
2. 準備資料。
3. 嚴謹的 model selection 與 validation。
4. 跟有意義的 benchmark 比：predictive baseline、business rule、其他建模方法。
5. 量化「預測變準」帶來的實際影響（營收、成本、效率、服務、風險），鼓勵用視覺化呈現。

## 每個專案要回答的四個問題

1. **What problem are we solving?** target stakeholder、現在的流程或決策、可以改善的機會。
2. **What can AI do?** 取得合適的資料，設計 AI 解法。
3. **Does it work?** 用合適的 benchmark 嚴謹評估。
4. **Does it matter?** 把技術表現換算成管理或社會上的影響。

## 交付物

| 交付物 | 對象 | 內容 |
|---|---|---|
| 2 分鐘影片 | target stakeholder | 自成一體的 demo 或 pitch |
| 簡報 5–10 頁 | target stakeholder | 問題、AI 解法（資料、建模方法、主要發現）、商業影響、建議 |
| GitHub repo + 一頁 README | 對方組織的 data scientist／ML engineer | 可執行的程式碼、重現結果的說明 |

影片和簡報回答「為什麼重要」，repo 回答「怎麼做的」。

## 評分取向（handout 明講的）

- 目的不是做出高 accuracy 的模型或複雜的 workflow，而是**用 AI 解決問題並證明實用價值**。
- 範圍要有企圖心但聚焦，幾週內能做完有意義的分析與評估。
- **簡單但有價值的模型，勝過技術複雜但價值不清楚的系統。**
- 好專案 = 有意義的問題 + 設計好的解法 + 嚴謹評估 + 有說服力的影響 + 清楚溝通。
- 要公開放進作品集的專案，不能含機密、專有、可識別個人或其他受限的資料。
