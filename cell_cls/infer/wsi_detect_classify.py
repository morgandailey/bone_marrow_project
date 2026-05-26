"""
wsi_detect_classify.py

偵測整張 WSI 上的所有細胞，篩出 MON / BAS 候選供醫師複查。

流程：
  1. 分塊 (tile) 讀取 WSI（TILE_SIZE x TILE_SIZE，相鄰 tile 重疊 OVERLAP px）
  2. 每塊用 Cellpose (cyto2) 偵測細胞 mask
  3. 只保留 centroid 落在 tile 「有效區」內的 cell（避免 overlap 邊界重複計算）
  4. 裁切細胞並批次送入 Stage 1 EfficientNetV2 分類
  5. 篩出 MON / BAS 且 softmax 信心度 >= conf_thresh 的細胞
  6. 存裁切圖 + 匯總 JSON（WSI 座標、類別、信心度）

使用方式：
  python wsi_detect_classify.py                          # 預設兩張 WSI
  python wsi_detect_classify.py --wsi /path/to/a.mrxs   # 指定單張

輸出（--output_dir 預設 cell_cls/runs/wsi_scan）：
  <output_dir>/crops/<WSI_STEM>/<MON|BAS>/<idx>.jpg
  <output_dir>/<WSI_STEM>_candidates.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.measure import regionprops
import timm
import openslide
from cellpose import models as cp_models

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
from dataset import CLASSES, VAL_TRANSFORM  # noqa: E402

# ── 常數 ──────────────────────────────────────────────────────────────────── #
PADDING     = 10
TILE_SIZE   = 2048
OVERLAP     = 64        # tile 重疊量；有效區 guard = OVERLAP // 2
CONF_THRESH = 0.7
TARGET_CLS  = {'MON', 'BAS'}
BATCH_SIZE  = 128
MODEL_NAME  = 'tf_efficientnetv2_m.in21k_ft_in1k'
NUM_CLASSES = len(CLASSES)   # 17


# ── 分類器 ─────────────────────────────────────────────────────────────────── #

def load_classifier(ckpt_path: str, device: torch.device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt['model_state'])
    return model.to(device).eval()


def classify_crops(model, crops: list, device: torch.device) -> list:
    """
    crops : list of PIL RGB images
    return: list of (cls_name: str, confidence: float, top3: list)
            top3 = [['CLS', prob], ['CLS', prob], ['CLS', prob]]
    """
    results = []
    for i in range(0, len(crops), BATCH_SIZE):
        batch = crops[i : i + BATCH_SIZE]
        tensors = torch.stack([VAL_TRANSFORM(c) for c in batch]).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            logits = model(tensors)
        probs              = F.softmax(logits, dim=1)
        confs, idx         = probs.max(dim=1)
        top3_vals, top3_idx = probs.topk(3, dim=1)
        for conf, cls_idx, t3v, t3i in zip(
            confs.cpu().tolist(), idx.cpu().tolist(),
            top3_vals.cpu().tolist(), top3_idx.cpu().tolist()
        ):
            top3 = [[CLASSES[i], round(v, 4)] for i, v in zip(t3i, t3v)]
            results.append((CLASSES[cls_idx], float(conf), top3))
    return results


# ── Cellpose → bounding box ────────────────────────────────────────────────── #

def masks_to_boxes(masks: np.ndarray) -> list:
    """masks: H×W int (0=bg, 1..N=cell). 回傳 [(y0,x0,y1,x1), ...]"""
    return [(p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3])
            for p in regionprops(masks)]


# ── 主流程 ─────────────────────────────────────────────────────────────────── #

def process_wsi(wsi_path: str, ckpt_path: str, output_dir: str,
                cp_model, device: torch.device,
                conf_thresh: float = CONF_THRESH) -> list:

    slide     = openslide.OpenSlide(wsi_path)
    W, H      = slide.dimensions        # level-0 full resolution
    stem      = os.path.splitext(os.path.basename(wsi_path))[0]
    crops_root = os.path.join(output_dir, 'crops', stem)
    for cls in TARGET_CLS:
        os.makedirs(os.path.join(crops_root, cls), exist_ok=True)

    clf_model  = load_classifier(ckpt_path, device)
    candidates = []
    counters   = {cls: 0 for cls in TARGET_CLS}
    json_path  = os.path.join(output_dir, f'{stem}_candidates.json')

    stride = TILE_SIZE - OVERLAP
    guard  = OVERLAP // 2
    xs = list(range(0, W, stride))
    ys = list(range(0, H, stride))
    n_tiles = len(xs) * len(ys)
    print(f'  Slide size: {W} x {H}  →  {n_tiles} tiles '
          f'({len(xs)} x {len(ys)}, stride={stride})')

    tile_idx = 0
    for ty in ys:
        for tx in xs:
            tile_idx += 1
            tw = min(TILE_SIZE, W - tx)
            th = min(TILE_SIZE, H - ty)

            tile_pil = slide.read_region((tx, ty), 0, (tw, th)).convert('RGB')
            tile_np  = np.array(tile_pil)

            # Cellpose
            masks, _, _ = cp_model.eval(tile_np, diameter=None, channels=[0, 0])

            all_boxes = masks_to_boxes(masks)
            if not all_boxes:
                continue

            # 只保留 centroid 在 有效區 內的 cell
            vx0 = guard if tx > 0     else 0
            vy0 = guard if ty > 0     else 0
            vx1 = tw - guard if (tx + tw) < W else tw
            vy1 = th - guard if (ty + th) < H else th

            valid_boxes = [
                (by0, bx0, by1, bx1)
                for (by0, bx0, by1, bx1) in all_boxes
                if vx0 <= (bx0 + bx1) / 2 < vx1
                and vy0 <= (by0 + by1) / 2 < vy1
            ]
            if not valid_boxes:
                continue

            # 裁切
            crops, meta = [], []
            for (by0, bx0, by1, bx1) in valid_boxes:
                cx0 = max(0, bx0 - PADDING)
                cy0 = max(0, by0 - PADDING)
                cx1 = min(tw, bx1 + PADDING)
                cy1 = min(th, by1 + PADDING)
                crops.append(tile_pil.crop((cx0, cy0, cx1, cy1)))
                meta.append({
                    'x': tx + cx0, 'y': ty + cy0,
                    'w': cx1 - cx0, 'h': cy1 - cy0,
                })

            # 分類
            preds = classify_crops(clf_model, crops, device)

            # 篩選
            n_kept = 0
            for (cls_name, conf, top3), cell_meta, crop_pil in zip(preds, meta, crops):
                if cls_name in TARGET_CLS and conf >= conf_thresh:
                    wsi_stem = os.path.splitext(os.path.basename(wsi_path))[0]
                    fname = f'{wsi_stem}_x{cell_meta["x"]}_y{cell_meta["y"]}.jpg'
                    save_path = os.path.join(crops_root, cls_name, fname)
                    crop_pil.save(save_path, quality=95)
                    candidates.append({
                        'wsi':       os.path.basename(wsi_path),
                        'cls':       cls_name,
                        'conf':      round(conf, 4),
                        'top3':      top3,
                        'x':         cell_meta['x'],
                        'y':         cell_meta['y'],
                        'w':         cell_meta['w'],
                        'h':         cell_meta['h'],
                        'crop_path': os.path.relpath(save_path, output_dir),
                    })
                    counters[cls_name] += 1
                    n_kept += 1

            if n_kept > 0:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        {'wsi': os.path.basename(wsi_path),
                         'total': len(candidates),
                         'MON':   counters['MON'],
                         'BAS':   counters['BAS'],
                         'conf_thresh': conf_thresh,
                         'results': candidates},
                        f, indent=2, ensure_ascii=False
                    )

            if tile_idx % 100 == 0 or n_kept > 0:
                total_cand = sum(counters.values())
                print(f'  [{tile_idx:4d}/{n_tiles}] '
                      f'tile=({tx},{ty})  '
                      f'cells={len(valid_boxes)}  kept={n_kept}  '
                      f'total_cand={total_cand} '
                      f'(MON={counters["MON"]} BAS={counters["BAS"]})')

    slide.close()

    print(f'\n  Done: MON={counters["MON"]}  BAS={counters["BAS"]}')
    print(f'  JSON → {json_path}')
    return candidates


# ── CLI ────────────────────────────────────────────────────────────────────── #

def parse_args():
    ap = argparse.ArgumentParser(description='WSI cell detection + MON/BAS classification')
    ap.add_argument('--wsi', nargs='+', default=[
        '/work/u4001296/project1/data/wsi/0000201584.mrxs',
        '/work/u4001296/project1/data/wsi/slide-2025-10-01T13-44-33-R1-S23.mrxs',
    ], help='WSI 路徑（可多個）')
    ap.add_argument('--ckpt', default=
        '/work/u4001296/project1/cell_cls/runs/stage1_v3/best.pt',
        help='Stage 1 模型權重')
    ap.add_argument('--output_dir', default=
        '/work/u4001296/project1/cell_cls/runs/wsi_scan',
        help='輸出目錄')
    ap.add_argument('--conf', type=float, default=CONF_THRESH,
        help='MON/BAS 篩選信心度門檻（預設 0.7）')
    ap.add_argument('--cellpose_model', default='cyto2',
        choices=['cyto', 'cyto2', 'cyto3'],
        help='Cellpose 模型類型')
    return ap.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading Cellpose ({args.cellpose_model})...')
    cp_model = cp_models.CellposeModel(
        gpu=torch.cuda.is_available(),
        model_type=args.cellpose_model,
    )

    for wsi_path in args.wsi:
        if not os.path.exists(wsi_path):
            print(f'[SKIP] not found: {wsi_path}')
            continue
        print(f'\n=== {os.path.basename(wsi_path)} ===')
        process_wsi(wsi_path, args.ckpt, args.output_dir,
                    cp_model, device, conf_thresh=args.conf)

    print('\nAll done.')


if __name__ == '__main__':
    main()
