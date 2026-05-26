# System Prototype：WSI 自動區域選取 + 細胞分類 ✕ Alovas 平台整合

> **狀態**：Prototype 規劃中（2026-05-25）
> **目標受眾**：開發者 / 研究人員

---

## 一、目的

讓醫師在 Alovas 病理平台上傳一張 WSI（Whole Slide Image）後，系統能：

1. **自動選出最具代表性的幾塊 ROI（Region of Interest）**，省去醫師手動框選
2. 針對這些 ROI **快速跑完 Cellpose 偵測 + EfficientNetV2 分類**
3. 以視覺化方式呈現每塊 ROI 的 **cell 類別分布**（各類比例 bar chart + 細胞位置圖）
4. 醫師可在平台上**快速瀏覽摘要結果**，決定是否需要進一步全片掃描或人工複查

---

## 二、整體架構

```
┌─────────────────────────────────────────────────────────────────┐
│                        Alovas 平台                               │
│                                                                  │
│   醫師上傳 WSI  ──►  Webhook / API call  ──►  System Backend    │
│                                                                  │
│   System Backend 回傳結果後，Alovas 在 WSI viewer 上疊加標注     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      System Backend                              │
│                                                                  │
│  Step 1  ROI Selector                                            │
│          ├─ 讀 WSI thumbnail（低倍，快速）                        │
│          ├─ 組織區域偵測（排除空白、折疊、染色不均區域）            │
│          └─ 選出 N 塊（預設 5 塊）最具代表性的 ROI                 │
│                                                                  │
│  Step 2  Cell Detection（Cellpose）                              │
│          └─ 對每塊 ROI 跑 Cellpose，輸出 cell mask / bbox        │
│                                                                  │
│  Step 3  Cell Classification（EfficientNetV2）                   │
│          └─ 對每個 detected cell crop 跑分類器                   │
│             （stage2 fine-tuned model）                          │
│                                                                  │
│  Step 4  Result Aggregation                                      │
│          ├─ 每塊 ROI 的 cell 類別比例統計                         │
│          ├─ 全 ROI 合計比例（用來判斷是否異常）                    │
│          └─ 轉成 Alovas annotation JSON 格式                     │
│                                                                  │
│  Step 5  Response to Alovas                                      │
│          ├─ 各 ROI 的 bounding box（讓 Alovas 畫框）              │
│          ├─ 每個 cell 的 bbox + 分類結果                          │
│          └─ 每塊 ROI 的比例摘要 + 全片摘要                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、ROI 自動選取策略（Step 1 細節）

ROI 選取是整個 prototype 的關鍵創新點。好的 ROI 應具備：

| 條件 | 說明 |
|------|------|
| 組織密度高 | 排除空白玻片區域（HSV 飽和度 + 亮度閾值） |
| 細胞分布均勻 | 避免嚴重重疊或過於稀疏的區域 |
| 染色品質正常 | 排除染色太淡或過深的異常區塊 |
| 空間分散 | N 塊 ROI 在 WSI 上盡量分散，提高代表性 |
| 大小固定 | 每塊 ROI 固定大小（e.g. 2000×2000 px @ 40x），確保統計可比較 |

**候選演算法**：
- **快速版**：Thumbnail 降採樣 → 灰階 + Otsu 二值化 → 找密度最高的格子（sliding window）→ greedy 挑分散的 top-N
- **進階版**：加入 Cellpose 在 thumbnail 層做初步 cell count，優先選細胞數多的區域

---

## 四、與 Alovas 平台的銜接方式（待確認）

> ⚠️ 以下為 prototype 假設，實際銜接方式需與 Alovas 開發端確認

| 項目 | 假設規格 |
|------|---------|
| 觸發方式 | Alovas 在醫師上傳/開啟 WSI 後呼叫 Webhook（HTTP POST） |
| 輸入格式 | WSI 路徑（共用 NAS / 物件儲存 URL）或直接傳檔 |
| 輸出格式 | JSON（參考 `cell_cls/convert_to_platform_json.py` 的格式）|
| 顯示方式 | Alovas 在 WSI viewer 上疊加 ROI 框 + cell 標點 + 比例 panel |
| 非同步 | 分析可能需 1~5 分鐘，應支援 async（先回 202，完成後 callback 或輪詢）|

---

## 五、輸出格式草稿

```json
{
  "wsi_id": "case_001",
  "analysis_time_sec": 73.2,
  "rois": [
    {
      "roi_id": 0,
      "x": 12400, "y": 8200,
      "width": 2000, "height": 2000,
      "cell_count": 182,
      "class_distribution": {
        "NEU": 0.61, "LYM": 0.22, "MON": 0.08,
        "EOS": 0.04, "BAS": 0.01, "EBL": 0.02, "OTH": 0.02
      },
      "cells": [
        {"cell_id": 0, "x": 12510, "y": 8340, "w": 45, "h": 42,
         "pred_class": "NEU", "confidence": 0.94}
      ]
    }
  ],
  "summary": {
    "total_cells_analyzed": 912,
    "merged_distribution": {
      "NEU": 0.59, "LYM": 0.24, "MON": 0.09,
      "EOS": 0.04, "BAS": 0.01, "EBL": 0.02, "OTH": 0.01
    },
    "alerts": ["MON 比例偏高（>8%）", "BAS 需人工確認"]
  }
}
```

---

## 六、資料夾結構規劃

```
system/
├── README.md                   ← 本文件
├── roi_selector/               ← ROI 自動選取模組
│   ├── __init__.py
│   ├── tissue_detector.py      ← 組織 vs 空白區域偵測
│   └── roi_picker.py           ← 從候選格子選出最終 N 塊 ROI
├── pipeline/                   ← 整合 detection + classification
│   ├── __init__.py
│   ├── detect.py               ← 封裝 Cellpose 呼叫
│   ├── classify.py             ← 封裝 EfficientNetV2 推論
│   └── run_roi.py              ← 對單一 ROI 跑完整 pipeline
├── api/                        ← 對外介面（與 Alovas 銜接）
│   ├── __init__.py
│   ├── server.py               ← FastAPI server（接收 Webhook）
│   ├── schemas.py              ← Request / Response Pydantic models
│   └── format_output.py        ← 轉成 Alovas annotation JSON
├── viz/                        ← 本地視覺化（開發用）
│   └── plot_roi_results.py
└── demo/                       ← Prototype demo 腳本
    └── run_demo.py             ← 給一張 WSI 跑完整流程並存圖
```

---

## 七、Prototype 開發優先順序

| 優先度 | 模組 | 說明 |
|--------|------|------|
| 🔴 P0 | `roi_selector` | 最核心的新功能，先做最簡單的版本（thumbnail + 密度 grid）|
| 🔴 P0 | `pipeline/run_roi.py` | 串接現有 Cellpose + 分類器，對單一 ROI 輸出結果 |
| 🟡 P1 | `api/server.py` | FastAPI 包裝，方便與 Alovas 或任何前端對接 |
| 🟡 P1 | `api/format_output.py` | 輸出格式對齊 Alovas 需求 |
| 🟢 P2 | `viz/` | 本地 debug 視覺化，開發期間驗用 |
| 🟢 P2 | `demo/run_demo.py` | 一鍵跑 demo，用於展示與討論 |

---

## 八、依賴的現有資源

| 資源 | 路徑 | 說明 |
|------|------|------|
| Stage 1 backbone | `cell_cls/runs/stage1_v3/best.pt` | ⚠️ 禁止覆蓋 |
| Stage 2 分類器 | `cell_cls/runs/stage2_v3/best.pt` | Prototype 優先使用 |
| Cellpose 設定 | `cell_cls/configs.py` | 沿用現有參數 |
| WSI 讀取 | `openslide` | conda 環境已安裝 |
| 類別對應 | `cell_cls/configs.py` → `CLASS_NAMES` | |

---

## 九、開放問題（待釐清）

- [ ] Alovas Webhook API 規格（輸入欄位、認證方式、callback URL）
- [ ] WSI 檔案存放位置（NAS 路徑？物件儲存？）
- [ ] 醫師端希望看到的 UI 元素（框的顏色、比例呈現方式）
- [ ] 每次分析幾塊 ROI 合適？（初始建議：5 塊，可設定）
- [ ] 同步 vs 非同步：分析時間可接受的上限？
- [ ] 是否需要儲存每次分析結果供後續 fine-tune 使用？

---

*Last updated: 2026-05-25*
