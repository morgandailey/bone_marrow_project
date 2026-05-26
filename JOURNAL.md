# 實驗日誌

---

## 2026-04-28

### 今天做了什麼

- 整理專案目錄結構
  - 刪除 `cell_detection0`（v1 舊版，已被 `yolo_det` 取代）
  - 刪除 `data/`（舊的 patch 資料，可重新產生）
  - 重新命名：`cell_detection` → `yolo_det`、`old_code` → `sparsedet`
  - 將 `annotation/` 和 `wsi/` 整合到 `data/` 下
- 分析兩個模型的設計與結果
  - YOLOv8（yolo_det）：Macro F1 = 0.205，問題根源是稀疏標注
  - SparseDet（sparsedet）：AP50 = 61.46，但用的是不同資料集且目前無法執行
- 釐清核心問題：稀疏標注導致 YOLO 訓練訊號錯誤，換更大模型無法解決
- 確定新方向：先訓 Classification Model，同步進行方法 A（自動標注→YOLO）和方法 B（比例輸出）
- 找到適合的 open dataset：Bone-Marrow-Cytomorphology_MLL（171K 張，骨髓，13 類對應）

### 明天從哪接著做

- [ ] 下載 MLL 資料集（TCIA 或 Kaggle，Kaggle 較方便）
- [ ] 從現有標注裁切 600 顆細胞（排除 unknown/deleted）
- [ ] 開始規劃 Classification Model 架構

---

## 2026-05-04

### 今天做了什麼

- 確認新方向：Classification Model（單顆細胞分類）
  - 方法 B（先做）：Cellpose → 裁切 → 分類 → 輸出比例
  - 方法 A（後做）：分類結果自動產生標注 → 訓 YOLO
  - 兩個方法共用同一條主線，可同步進行
- 找到並下載 MLL 骨髓資料集
  - 路徑：`/work/u4001296/project1/data/mll/bone_marrow_cell_dataset/`
  - 21 類，171,374 張，6.8GB
  - 嚴重類別不平衡（ABE 只有 8 張，NGS 有 29K 張）
- 確定模型架構：**EfficientNetV2-M**（timm 載入 ImageNet 預訓練）
- 確定訓練策略：兩階段 Transfer Learning
  - Stage 1：MLL 21 類，171K 張，完整訓練
  - Stage 2：加入 megakaryocyte，共 22 類，用自己的 600 顆 fine-tune（backbone 凍結）
- 確認先跑 Stage 1，看效果再決定要不要 Stage 2

### 類別對應整理

| MLL 類別 | 你的類別 | 備註 |
|:---|:---|:---|
| BLA | myeloblast | ✓ |
| PMO | promyelocyte | ✓ |
| MYB | myelocyte | ✓ |
| MMZ | metamyelocyte | ✓ |
| NGB | band neutrophil | ✓ |
| NGS | segmented neutrophil | ✓ |
| MON | monocyte | ✓ |
| EOS + ABE | eosinophil | 合併 |
| BAS | basophil | ✓ |
| LYT + LYI | lymphocyte | 合併 |
| PLM | plasma cell | ✓ |
| EBO + PEB | normoblast | 合併 |
| megakaryocyte | megakaryocyte | MLL 沒有，Stage 2 才加 |
| ART/NIF/OTH/FGC/HAC/KSC | other | 合併兜底 |

### 明天從哪接著做

- [x] 寫 Stage 1 訓練程式（EfficientNetV2-M + timm + Weighted Loss）
- [x] 寫 slurm 腳本
- [ ] 從現有標注裁切 600 顆細胞（Stage 2 準備）

---

## 2026-05-05

### 今天做了什麼

- 完成 `cell_cls/` 訓練程式
  - `configs.py`：超參數設定（model、lr、batch size、AMP、early stopping）
  - `dataset.py`：BoneMarrowDataset，21 類，stratified split
  - `train_stage1.py`：完整訓練流程（AMP、Weighted CrossEntropyLoss、CosineAnnealingLR、early stopping）
  - `train_stage1.slurm`：slurm 提交腳本
- 修正兩個 bug：
  - timm 模型名稱需帶 pretrained tag → 改用 `tf_efficientnetv2_m.in21k_ft_in1k`
  - `build_model` 硬寫模型名稱 → 改從 config 讀取
- **跑完 Stage 1 訓練**
  - Early stopping 在 epoch 20 觸發
  - Best val F1 = **0.7815**（21 類，Macro F1）
  - 權重存於 `cell_cls/runs/stage1/best.pt`

### 明天從哪接著做

- [x] 看 per-class F1，找出哪幾類分最差
- [ ] 決定要不要做 Stage 2（加自己的 600 顆細胞 fine-tune）
- [ ] 從現有標注裁切 600 顆細胞（Stage 2 準備）
- [ ] 開始規劃方法 B pipeline（Cellpose → 裁切 → 分類 → 比例輸出）

---

## 2026-05-05

### 今天做了什麼

- 分析 Stage 1 結果，發現嚴重問題：per-class F1 幾乎全是 0
  - 實際上只有少數類別（BAS、HAC、KSC 等）有 F1 > 0
  - 多數類別（NGS 29K、EBO 27K、ART 19K）全是 0
  - 訓練時顯示的 val F1 = 0.78 是假的（sklearn 只對出現的類別取平均）
- 釐清根本原因：Weighted Loss 的 1/count 算法差距 3600 倍，導致模型主動放棄學多數類別
- 進行第二次訓練（stage1_v2）：
  - 合併樣本 < 100 的類別（ABE→EOS、LYI→LYT、KSC+FGC→OTH），21 類縮為 17 類
  - Loss weight 改 sqrt（差距從 3600 倍縮到 60 倍）
  - 加 WeightedRandomSampler（少數類別過採樣到 2500 次/epoch）
  - 加強 Augmentation（rotation 30°、ColorJitter 加 hue）
  - 修正 val F1 指標（加 labels 參數，強制對全 17 類取平均）
- stage1_v2 結果：Best val F1 = 0.2785（epoch 10，early stop at 15）
  - 仍然只有 5 類有 F1 > 0（EOS/BAS/HAC/OTH/LYT）
  - 多數類別依然 F1 = 0
  - 問題：oversample 讓少數類別主導訓練，模型遇到不確定的就猜少數類別

### 下一步：換 Focal Loss

- Focal Loss 自動讓「已經學好的樣本」貢獻更少梯度，把訓練資源集中在還沒學好的類別
- 不需要手動設 weight 或 oversample，比現在的方式更乾淨
- 計畫：移除 WeightedRandomSampler，換 Focal Loss，重跑訓練

---

## 2026-05-06

### 今天做了什麼（續）

- 發現根本 bug：`os.listdir` 只讀一層目錄，但大類別（NGS、ART、EBO 等）資料夾內有子目錄
  - 導致前三次訓練實際上只用了 1,306 張圖（BAS/HAC/OTH/LYT/EOS），而非 171K 張
  - 所有前三次的結果（包括 val F1=0.78 假高、F1=0.27 真低）都是在 1,306 張上跑的，沒有參考價值
- 修正：改用 `os.walk` 遞迴搜尋，確認找到 171,374 張
- 同時修正損壞圖片的處理（`__getitem__` 加 try-except）
- 第一個 epoch 驗證：val F1 = **0.7154**，確認資料讀取正常
- 改用 sbatch 提交（job 188573，節點 hgpn05），等 email 通知結果

### stage1_v3 結果（2026-05-06）

- **Best val Macro F1 = 0.8074**（epoch 22，early stop at 27）
- 全部 17 類都有 F1，不再有 F1=0 的類別

| 類別 | F1 | 備註 |
|---|---|---|
| EBO | 0.947 | |
| EOS | 0.966 | |
| NGS | 0.921 | |
| LYT | 0.915 | |
| PLM | 0.929 | |
| ART | 0.898 | |
| PMO | 0.848 | |
| BLA | 0.862 | |
| HAC | 0.821 | |
| OTH | 0.816 | |
| MON | 0.764 | |
| PEB | 0.765 | |
| NGB | 0.755 | 與 NGS/MMZ 互混 |
| MYB | 0.725 | 與 PMO/MMZ 互混 |
| NIF | 0.644 | 定義本來就模糊 |
| BAS | 0.612 | val 只有 44 張 |
| MMZ | 0.535 | 骨髓分化中間態，與 NGB/MYB 難分 |

- MMZ/NGB/MYB 互相混淆符合預期（骨髓分化系列形態相近）
- BAS 樣本太少（44 張）是 F1 低的主因

#### 每類被猜成什麼（前 3 名，全名版）

| 真實類別 | F1 | 第1猜（正確） | 第2猜 | 第3猜 |
|---|---|---|---|---|
| Artefact | 0.898 | Artefact 90% | Lymphocyte 2% | Not Identifiable 2% |
| Basophil | 0.612 | Basophil 59% | Eosinophil 11% | Segmented Neutrophil 9% |
| Blast | 0.862 | Blast 86% | Lymphocyte 4% | Promyelocyte 3% |
| Erythroblast | 0.947 | Erythroblast 95% | Lymphocyte 2% | Proerythroblast 1% |
| Eosinophil | 0.966 | Eosinophil 97% | Segmented Neutrophil 1% | Artefact 0% |
| Hairy Cell | 0.821 | Hairy Cell 78% | Lymphocyte 17% | Artefact 5% |
| Lymphocyte | 0.915 | Lymphocyte 92% | Artefact 2% | Blast 2% |
| Metamyelocyte | 0.535 | Metamyelocyte 50% | Band Neutrophil 19% | Myelocyte 16% |
| Monocyte | 0.764 | Monocyte 76% | Blast 7% | Lymphocyte 4% |
| Myelocyte | 0.725 | Myelocyte 73% | Promyelocyte 17% | Metamyelocyte 3% |
| Band Neutrophil | 0.755 | Band Neutrophil 74% | Segmented Neutrophil 16% | Metamyelocyte 5% |
| Segmented Neutrophil | 0.921 | Segmented Neutrophil 92% | Band Neutrophil 5% | Artefact 1% |
| Not Identifiable | 0.644 | Not Identifiable 61% | Artefact 10% | Lymphocyte 10% |
| Other | 0.816 | Other 82% | Artefact 11% | Blast 5% |
| Proerythroblast | 0.765 | Proerythroblast 74% | Erythroblast 12% | Blast 9% |
| Plasma Cell | 0.929 | Plasma Cell 92% | Erythroblast 3% | Lymphocyte 3% |
| Promyelocyte | 0.848 | Promyelocyte 86% | Myelocyte 7% | Blast 2% |

### 目前狀態

- stage1_v3 訓練完成，`cell_cls/runs/stage1_v3/best.pt`
- WSI domain shift 嚴重（F1=0.101），確認需要 Stage 2

---

## 2026-05-06

### Stage 2 設計筆記

#### 模型架構說明

EfficientNetV2-M 分兩段：

```
圖片 (224×224) → [Backbone] → 1280 個特徵數字 → [分類層] → 17 類機率
```

**Backbone**：把圖片壓縮成抽象特徵向量（核的形狀、顆粒感、核膜厚薄等）。這些特徵是通用的，德國資料學到的在台灣資料上依然有效。

**分類層**：一層線性加權，把 1280 個特徵數字轉成 17 個類別分數。Stage 1 的權重是根據 MLL 德國資料調出來的，台灣 WSI 染色條件不同導致特徵分布偏移，所以分類層的決策跑掉了。

#### Stage 2 策略

- **凍結 Backbone**：特徵提取層完全不動
- **只訓練分類層**：用我們的 1,169 顆標注細胞重新校準
- 資料：`annotation/train/`（~1,169 顆），test 用 `annotation/vali/`（~4,488 顆）
- 凍結 Backbone 的好處：就算標注有少量噪音，影響範圍只限分類層，不會破壞已學到的形態特徵

#### 下一步

- [x] 寫 Stage 2 訓練腳本
- [x] 跑完後用 vali 分割評估

### Stage 2 結果（2026-05-06）

- **Best val Macro F1 = 0.316**（best epoch 43，凍結 Backbone，只訓練分類層）
- 資料：train 分割 ~1,169 顆（微調），vali 分割 ~4,488 顆（評估）

| 類別 | Stage 1 | Stage 2 | 變化 |
|---|---|---|---|
| EOS | 0.545 | **0.830** | +0.285 |
| NGB | 0.117 | **0.658** | +0.541 |
| EBO | 0.118 | **0.671** | +0.553 |
| MMZ | 0.031 | **0.578** | +0.547 |
| MYB | 0.012 | **0.576** | +0.564 |
| PLM | 0.316 | **0.524** | +0.208 |
| LYT | 0.000 | **0.466** | +0.466 |
| NGS | 0.342 | **0.497** | +0.155 |
| ART | 0.000 | 0.000 | train 無樣本 |
| HAC | 0.000 | 0.000 | train 無樣本 |
| MON | 0.000 | 0.000 | train 樣本極少 |
| PEB | 0.000 | 0.000 | train 樣本極少 |

- F1=0 的類別都是 train 分割裡幾乎沒有樣本的類別，不是模型問題
- 權重：`cell_cls/runs/stage2/best.pt`

---

---

## 2026-05-23

### 今天做了什麼

- 確認 Cellpose cyto2 在骨髓 WSI patch 上分割效果良好（目視確認）
- 寫 `wsi_detect_classify.py`：WSI 全圖掃描（tile → Cellpose → 分類 → 候選 JSON）
- 跑 WSI scan（conf=0.7，兩張 WSI），輸出 `runs/wsi_scan/`
  - MON 候選：4,245 顆（0000201584: 614、S23: 3,631）
  - BAS 候選：38 顆
- 發現 WSI domain shift 嚴重：MON/BAS 幾乎全被 LYT/MMZ/NGB 誤判

### 明天從哪接著做

- 跑 conf=0.5 補更多候選
- 規劃 Stage 2 fine-tune（本地 WSI 標注）

---

## 2026-05-24

### 今天做了什麼

- 跑 conf=0.5 S23 WSI scan，輸出 `runs/wsi_scan_conf05/`
  - MON 候選：2,569 顆、BAS 候選：28 顆
- 新增 `convert_to_platform_json.py`：候選 JSON → 平台 annotation 格式（含 top3 probabilities）
- 重新訓練 Stage 2 v2（重新 80/20 split）：
  - Best val F1 = **0.429**（+0.113 vs stage2）
  - MON F1 仍 = 0（train 只有 2 張）
  - 權重：`runs/stage2_v2/best.pt`

---

## 2026-05-25

### 今天做了什麼

- 人工確認 WSI scan 候選：有真正 MON，但大多數是 MMZ/NGB/NGS 假陽性
  - S23 後段幾乎無 MON，推測前段混有外周血（hemodilution）
- 手動從 S23 scan 結果挑選明顯 MON 移入 wsi_crops/train/
  - MON: 2 → **38 張**、BAS: 14 → **17 張**
- 重新訓練 Stage 2 v3：
  - Best val F1 = **0.480**（epoch 37，early stop at 47）
  - **MON F1 = 0.667**（從 0 到有，38 張就足夠）
  - BAS F1 = 0.476（+0.143）
  - 權重：`runs/stage2_v3/best.pt`
- 更新 WSI scan 腳本改用 stage2_v3（`--ckpt runs/stage2_v3/best.pt`）
- 整理 cell_cls/ 目錄結構：
  - Python 檔分類到 `core/` `data/` `train/` `eval/` `infer/`
  - 投遞腳本移到 `scripts/`
  - 廢棄 runs 移到 `runs/archive/`

### 目前狀態

- 現役模型：`runs/stage2_v3/best.pt`（WSI macro F1=0.480）
- 待改善：BAS（F1=0.476）、BLA（F1=0.456，形態混淆）、MON（F1=0.667，樣本仍不足）
- 下一步：用 stage2_v3 重跑 WSI scan → 送平台讓醫師確認 → 累積更多標注 → fine-tune v4

---

## cell_cls/ 檔案說明

| 檔案 | 用途 |
|---|---|
| `configs.py` | 所有超參數設定（stage1 和 stage2） |
| `dataset.py` | MLL 資料集的 Dataset class、17 類定義、MERGE_MAP、transforms |
| `train_stage1.py` | Stage 1 訓練主程式（MLL 171K 張，Focal Loss） |
| `train_stage1.slurm` | Stage 1 的 slurm 提交腳本 |
| `eval_stage1.py` | Stage 1 評估腳本，輸出 per-class F1 和 confusion 分析 |
| `crop_wsi_cells.py` | 從 WSI 裁切標注細胞存到磁碟（`wsi_crops/train/` 和 `wsi_crops/val/`） |
| `train_stage2.py` | Stage 2 訓練主程式（凍結 Backbone，只訓練分類層） |
| `train_stage2.slurm` | Stage 2 的 slurm 提交腳本 |
| `test_on_wsi.py` | 用完整 WSI 標注測試模型（不分 train/val，用於 domain shift 診斷） |
| `check_annotations.py` | 用模型掃描標注，找出預測與標注差異大的可疑樣本 |
| `wsi_crops/` | 裁切後的細胞圖片（train/val，按類別分資料夾） |
| `runs/stage1_v3/` | Stage 1 訓練結果（best.pt、history.json、eval_report.json） |
| `runs/stage2/` | Stage 2 訓練結果（best.pt、history.json、eval_report.json） |