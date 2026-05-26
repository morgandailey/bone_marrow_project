# 專案紀錄：骨髓血球自動分類 on WSI

> 這份文件是給 Claude / LLM 看的專案狀態紀錄。
> **每次有重要進展都要更新這份文件**，讓下一個對話能立刻接手。
> 最後更新：2026-05-25

---

## 🆕 System Prototype（Alovas 平台整合）

`system/` 資料夾存放 Alovas 平台整合 prototype 的所有研究與程式碼。

**目標**：醫師在 Alovas 上傳 WSI → 系統自動選出 N 塊最具代表性 ROI → 跑 Cellpose + EfficientNetV2 → 回傳每塊 ROI 的細胞分佈給醫師快速預覽。

詳見 [system/README.md](system/README.md)。

---

## ⚠️ 絕對禁止事項

**`cell_cls/runs/stage1_v3/best.pt` 不可覆蓋、不可刪除、不可修改。**

這是在 MLL 171K 張骨髓細胞上預訓練的 backbone，是整個系統的基石。每次 fine-tune 都必須從這個檔案出發，不能從任何 stage2 版本繼續。

---

## 專案目標（已確立）

**快速告知醫師哪些 cell 類別比例異常（偏高/偏低）**

- 不需要逐顆完美分類
- 各類別比例估計可信 → 觸發異常警示 → 醫師最終確認
- 醫師有一個平台可縮放查看 cell 及其周圍環境，並按 true/false 確認

---

## 核心流程（已確立）

```
WSI (.mrxs)
  → Cellpose 偵測所有細胞
  → EfficientNetV2 分類（Stage 1 or Stage 2）
  → 篩出 MON / BAS 等稀少類別候選
  → convert_to_platform_json.py 轉成平台格式
  → 醫師在平台確認（可縮放看周圍 context）
  → 確認的 cell 放入 wsi_crops/
  → 合併後重新 80/20 split → fine-tune Stage 2_v3
  → 迭代改善
  → 分類器足夠穩定後，大量產生高品質 pseudo labels（bbox + class）
  → 用 pseudo labels 訓練更快速的 YOLO / detection model
  → 部署時用 detection model 快速輸出全 WSI 細胞分布與異常比例
```

### 方法定位

- **短中期主線：Classification pipeline**
  - Cellpose 負責「找細胞」，EfficientNetV2 負責「分細胞類別」
  - 先把 MON / BAS 等臨床重點稀少類別候選品質做穩
  - 產出比例估計與醫師複查清單，建立可迭代的本地標注資料閉環
- **長期部署方向：Detection pipeline**
  - 當 classification model 的候選品質足夠高，才用它大量產生 pseudo labels
  - pseudo labels 經過信心度、top-3 margin、尺寸/context 規則與醫師抽查後，作為 YOLO 訓練資料
  - YOLO 不是目前直接拿稀疏人工標注訓練，而是作為「分類器穩定後」的高速推論模型

---

## 環境與部署

### Conda 環境
```bash
conda activate project1   # Python 3.10, PyTorch 2.6, CUDA 12.4
```

### 必要套件
```bash
pip install torchvision timm openslide-python openslide-bin cellpose \
            Pillow scikit-image opencv-python scikit-learn numpy pandas matplotlib tqdm
```
完整清單（含版本號）：`requirements.txt`

### Cellpose 注意事項
- 使用 `CellposeModel`（cellpose 3.x API），`diameter=None`（自動估計直徑）
- cyto2 在骨髓 WSI patch 上視覺確認效果良好（2026-05-23）
- 背景過濾條件：`mean < 235` 且 `std > 15` 才算有組織的 patch

### SLURM（NCHC 國網）
| 項目 | 值 |
|------|----|
| `--account` | `MST114560` |
| 一般 job partition | `normal`（H100，最長 2d） |
| 快速測試 | `dev`（H100，最長 1h） |
| 大量訓練 | `normal2`（H200，最長 2d） |

### 常用投遞指令
```bash
# WSI 全圖掃描（conf=0.7，兩張 WSI，用 stage2_v3）
sbatch cell_cls/scripts/run_wsi_scan.sh          # → runs/wsi_scan_s2v3/

# WSI 全圖掃描（conf=0.5，S23，用 stage2_v3）
sbatch cell_cls/scripts/run_wsi_scan_s23_conf05.sh   # → runs/wsi_scan_s23_s2v3/

# Stage 2 fine-tune（讀 configs.py 設定）
sbatch cell_cls/scripts/run_train_stage2.sh
```

---

## 目錄結構

```
project1/
├── cell_cls/                    # 主要分類系統
│   ├── core/                    # 共用模組
│   │   ├── dataset.py           # Dataset, CLASSES, MERGE_MAP, transforms
│   │   └── configs.py           # Stage 1 / Stage 2 超參數設定
│   ├── data/                    # 資料前處理
│   │   ├── crop_wsi_cells.py    # 從 WSI annotation JSON 裁切單顆細胞
│   │   ├── merge_wsi_crops.py   # 合併多次標注
│   │   └── check_annotations.py # 標注品質檢查
│   ├── train/                   # 訓練腳本
│   │   ├── train_stage1.py      # Stage 1 訓練（MLL 資料集）
│   │   └── train_stage2.py      # Stage 2 fine-tune（WSI 本地標注）
│   ├── eval/                    # 評估 / 視覺化
│   │   ├── eval_stage1.py       # Stage 1 評估
│   │   ├── test_on_wsi.py       # WSI domain shift 評估
│   │   ├── plot_performance.py  # Per-class F1 比較圖
│   │   └── plot_ppt.py          # PPT 用 heatmap + 資料量圖
│   ├── infer/                   # 推論
│   │   ├── wsi_detect_classify.py  # WSI 全圖掃描（Cellpose + 分類）
│   │   ├── convert_to_platform_json.py  # 候選 JSON → 平台格式
│   │   └── test_cellpose.py     # Cellpose 效果目視確認
│   ├── scripts/                 # SLURM 投遞腳本
│   │   ├── run_wsi_scan.sh      # conf=0.7，兩張 WSI，stage2_v3
│   │   ├── run_wsi_scan_s23_conf05.sh  # conf=0.5，S23，stage2_v3
│   │   ├── run_train_stage2.sh  # Stage 2 fine-tune
│   │   ├── train_stage1.slurm
│   │   └── train_stage2.slurm
│   ├── logs/                    # SLURM job 輸出
│   ├── wsi_crops/               # 人工確認的標注資料（train/ + val/）
│   └── runs/
│       ├── stage1_v3/           # ★ Stage 1 預訓練（MLL，macro F1=0.807）
│       │   ├── best.pt          # ← 不可動
│       │   └── eval_report.json
│       ├── stage2_v2/           # 舊版 fine-tune（WSI，macro F1=0.429）
│       ├── stage2_v3/           # ★ 現役最佳 fine-tune（WSI，macro F1=0.480）
│       │   └── best.pt
│       ├── wsi_scan/            # stage1 時代 conf=0.7 掃描結果
│       ├── wsi_scan_conf05/     # stage1 時代 conf=0.5 S23 掃描結果
│       └── archive/             # 廢棄版本（stage1, stage1_v2, stage2, cellpose_test）
├── archive/                     # 封存：sparsedet/（SparseDet ICCV 2023）、yolo_det/（舊版 YOLO）
├── data/
│   ├── mll/bone_marrow_cell_dataset/  # MLL 171K 張預訓練資料
│   ├── annotation/              # WSI 人工標注 JSON（ALOVAS 格式）
│   └── wsi/                     # WSI 原始影像（.mrxs）
├── system/                      # Alovas 平台整合 prototype
├── JOURNAL.md
├── README.md
└── requirements.txt             # pip 套件清單（含版本號）
```

---

## 模型現況

### Stage 1（MLL 預訓練）
- 架構：`tf_efficientnetv2_m.in21k_ft_in1k`，17 類
- 資料：MLL 171,374 張，Focal Loss (gamma=2.0)
- 結果（MLL val）：macro F1 = **0.807**
- 弱類別：MMZ(0.535)、BAS(0.612)、NIF(0.644)
- 權重：`cell_cls/runs/stage1_v3/best.pt`
- **WSI domain shift**：MON F1=0.000、BAS F1=0.000（染色差異導致）

### Stage 2（WSI Fine-tune，現役）
- 從 `stage1_v3/best.pt` 出發，凍結 backbone，只訓練 classifier head
- 資料：wsi_crops/ 合併後 80/20 stratified split
- 結果（WSI val）：macro F1 = **0.429**
- 權重：`cell_cls/runs/stage2_v2/best.pt`
- F1=0 的類別：MON、ART、HAC、NIF、OTH、PEB（標注幾乎為 0）

### Stage 2 v3（現役最佳，2026-05-25）
- 新增 MON 38 張（共 41 張）、BAS 17 張（共 60 張）後重訓
- 結果：macro F1 = **0.480**（+0.051），**MON F1 = 0.667**（從 0 到有），BAS F1 = 0.476
- Early stopping at epoch 47，Focal Loss 已足夠，未加 WeightedRandomSampler
- 權重：`cell_cls/runs/stage2_v3/best.pt`
- SLURM log：`cell_cls/logs/train_s2v3_212995.out`

---

## 資料現況

### 本地標注（wsi_crops/）

| 類別 | train | val | 合計 | 目標 | 狀態 |
|---|---|---|---|---|---|
| EBO | 213 | 1,085 | 1,298 | — | ✅ 充足 |
| NGB | 206 | 755 | 961 | — | ✅ 充足 |
| MMZ | 162 | 595 | 757 | — | ✅ 充足 |
| MYB | 161 | 525 | 686 | — | ✅ 充足 |
| EOS | 112 | 477 | 589 | — | ✅ 充足 |
| LYT | 121 | 337 | 458 | — | ✅ 充足 |
| NGS | 76 | 336 | 412 | — | ✅ 充足 |
| PLM | 66 | 133 | 199 | — | ✅ 充足 |
| PMO | 22 | 81 | 103 | — | ✅ 充足 |
| **BLA** | 26 | 118 | **144** | ~300 | ⚠️ F1=0.456，形態混淆（BLA/PMO 難分）+ domain shift |
| **BAS** | 17 | 43 | **60** | ~150 | ⚠️ 不足，但 v3 已提升至 F1=0.476 |
| **MON** | 38 | 3 | **41** | ~100 | ⚠️ 仍不足，但 v3 已提升至 F1=0.667 |

> ✅ Stage 2 v3 已將 `wsi_crops/train/` 與 `wsi_crops/val/` 合併後重新做 80/20 stratified split；但 MON/BAS 樣本數仍偏少，後續仍需補標注。

### WSI Scan 候選

| 掃描 | WSI | conf | MON 候選 | BAS 候選 |
|---|---|---|---|---|
| wsi_scan/ | 0000201584 + S23 | 0.7 | 4,245 | 38 |
| wsi_scan_conf05/ | S23 | 0.5 | 2,569 | 28 |

> **MON 候選假陽性分析（2026-05-25 人工確認）**：有真正的 MON 存在，但大部分候選實為 MMZ、NGB、NGS。S23 後段（玻片下半部）幾乎無明顯 MON，推測前段混有較多外周血（hemodilution），後段才是純骨髓。後段 ~800 個 MON 候選幾乎全是假陽性，送醫師確認效率極低，需加二次篩選。

---

## 已知問題

### 1. Domain Shift
- MLL 染色（深紫 BAS）vs WSI 染色（LYT 偏藍紫）→ LYT 被誤判為 BAS
- 根本解：累積 WSI 本地標注 fine-tune

### 2. Cell Crop 未做物理尺寸正規化（Size Feature 丟失）
- `dataset.py` 的 `Resize((224, 224))` 讓所有 cell 縮到同尺寸
- 巨核細胞（bbox ~120 px）和單核細胞（bbox ~25 px）進模型後「看起來一樣大」，大小這個重要診斷線索完全丟失
- 額外風險：`megakayocyte` 在 `crop_wsi_cells.py` 的 `SKIP` 裡（訓練集沒有），但推論時 Cellpose 還是會偵測到並送入分類器，可能被誤判為 MON/BAS
- **受影響檔案**：`cell_cls/wsi_detect_classify.py` L151–156、`cell_cls/crop_wsi_cells.py` L75–85
- **建議修法**：改成以 centroid 為中心、固定物理尺寸（建議 30 µm）裁切，需先確認 WSI 的 `openslide.mpp-x`（@ 40× ≈ 0.25 µm/px → CROP_PX ≈ 120 px），不足大小補白邊；改後需同步重新生成 `wsi_crops/` 並重訓 stage2

### 4. BLA（骨髓母細胞）F1 偏低（0.456）
- 臨床重要性極高：**BLA ≥ 20% 即 AML 診斷標準**
- 雖有 144 張標注（比 MON/BAS 多），但 Precision 0.464 / Recall 0.448 都低，表示是形態混淆問題而非數量問題
- 主因：BLA vs PMO 都是不成熟骨髓細胞，圓核 + 細緻染色質 + 核仁，crop 圖難以區分
- 加上 WSI domain shift（MLL 上 F1=0.862 → WSI 上 0.456）
- 改善方向：①確認 144 張標注品質（BLA/PMO 邊界案例請醫師重審）②增加到 ~300 張③長期加入 context window

### 3. wsi_crops/ 圖片命名混用，重複偵測只能靠 hash
- 舊圖（手選自 wsi_scan）是流水號命名，無法對應回座標；新版腳本已改為 `{wsi}_x{x}_y{y}.jpg`
- 每次加新圖後必須跑 `python cell_cls/data/dedup_crops.py` 做 MD5 hash 比對
- 根本解無法實施：38 張 MON、17 張 BAS 是手選的，沒有 annotation JSON，刪掉找不回來

### 4. 某些 Cell 需要上下文
- 早期 EBO vs LYT：裁切圖幾乎一樣
- MON vs MMZ：大小是主要差異，裁切圖看不出
- 平台的縮放功能部分解決此問題

---

## 平台整合

醫師平台的 annotation JSON 格式：
```json
{
  "annotation": [
    {
      "name": "monocyte",
      "caption": "Monocyte",
      "type": "MultiPolygon",
      "index": 1,
      "partOfGroup": "DEFAULT",
      "coordinates": [[x0,y0],[x0,y1],[x1,y1],[x1,y0]],
      "temporary": false,
      "probabilities": [
        {"cls": "monocyte", "prob": 0.72},
        {"cls": "lymphocyte", "prob": 0.18},
        {"cls": "metamyelocyte", "prob": 0.05}
      ]
    }
  ]
}
```

轉換方式：
```bash
python cell_cls/convert_to_platform_json.py \
    --input cell_cls/runs/wsi_scan/ \
    --output_dir platform_json/
```

> `probabilities` 欄位平台目前不使用，待平台改版後可自動建議分類。

---

## Class Name 對應表

| 代碼 | 平台名稱 | 中文 |
|---|---|---|
| MON | monocyte | 單核球 |
| BAS | basophil | 嗜鹼性球 |
| LYT | lymphocyte | 淋巴球 |
| NGS | neutrophil_segmented | 分葉核中性球 |
| NGB | neutrophil_band | 桿狀核中性球 |
| MMZ | metamyelocyte | 後骨髓球 |
| MYB | myelocyte | 骨髓球 |
| EOS | eosinophil | 嗜酸性球 |
| EBO | erythroblast | 正常紅血球前驅 |
| PEB | proerythroblast | 早期紅血球前驅 |
| PMO | promyelocyte | 前骨髓球 |
| PLM | plasma_cell | 漿細胞 |
| BLA | blast | 骨髓母細胞 |
| NIF | immature_cell | 不成熟細胞 |
| HAC | hairy_cell | 毛細胞 |
| OTH | other | 其他 |
| ART | artifact | 偽影 |

> ⚠️ `neutrophil_segmented`、`neutrophil_band` 等名稱未確認，需比對平台實際使用的名稱。

---

## 細胞形態辨認口訣

### BAS（嗜鹼性球）
```
✓ 深紫/藍黑色粗大顆粒（最關鍵）
✓ 細胞質幾乎被顆粒佔滿
✓ 核被顆粒蓋住或不規則分葉
✗ 細胞質乾淨 → 不是 BAS
✗ 圓核深藍 → LYT
✗ 橘紅顆粒 → EOS
```

### MON（單核球）
```
✓ 細胞大（比 LYT 大 1.5–2 倍）
✓ 核不規則，腎形/子彈形/馬蹄形，有皺摺感
✓ 細胞質灰藍、量多、乾淨
✓ 染色質 fine-grained 滑順，不結塊
✗ 深藍核 → LYT
✗ 染色質 coarse 聚集 → MMZ/MYB
✗ 細胞質有粉紅顆粒 → MMZ
✗ 核明顯 banded → NGB
```

---

## 各類別骨髓正常占比與異常意義

| 代碼 | 正常占比 | 增多代表 |
|---|---|---|
| NGS | 7–25% | 細菌感染、CML |
| NGB | 10–35% | 敗血症（核左移） |
| EBO | 10–30% | 溶血/缺鐵性貧血 |
| MYB | 5–20% | CML、感染 |
| MMZ | 5–15% | CML、感染 |
| LYT | 5–20% | ALL、CLL、病毒感染 |
| PMO | 2–8% | AML-M3（可達 ~90%） |
| BLA | < 5% | AML（≥20% 即診斷） |
| PLM | < 2% | 多發性骨髓瘤（≥10%） |
| MON | 1–3% | CMML、AML-M5 |
| EOS | 1–4% | 過敏、寄生蟲感染 |
| BAS | < 1% | **CML（診斷標準之一）** |

---

## 下一步行動（按優先順序）

- [ ] **用 `stage2_v3/best.pt` 重新跑 WSI scan**：比較 MON/BAS 候選數量與假陽性型態是否改善
- [ ] **測試 convert_to_platform_json.py** 輸出格式是否符合平台需求，並確認平台 class name 列表（`neutrophil_segmented` 等名稱需核實）
- [ ] **把 WSI scan 候選餵進平台**，讓醫師確認 MON/BAS 候選，回收 true/false 結果
- [ ] **加二次篩選規則**：bbox size、top-3 margin、區域位置、固定物理尺寸 crop/context，降低 MON 假陽性
- [ ] **持續累積 MON/BAS 本地標注後 fine-tune 下一版 Stage 2**
- [ ] **當分類器品質足夠穩定後，批次產生 pseudo labels**（bbox + class + confidence），抽查/清理後用於訓練 YOLO detection model

---

## 廢棄/舊版（可刪除）

| 路徑 | 狀態 | 說明 |
|---|---|---|
| cell_cls/runs/stage1/ | 可刪 | os.walk bug，只用了 1,306 張 |
| cell_cls/runs/stage1_v2/ | 可刪 | WeightedRandom 過採樣，效果差 |
| cell_cls/runs/stage2/ | 可刪 | train/val 比例反向，F1=0.316 |
| archive/yolo_det/ | 封存 | 稀疏標注問題，已放棄此方向 |
| archive/sparsedet/ | 封存 | 無法執行，環境不相容 |
