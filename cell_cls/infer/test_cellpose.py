"""
test_cellpose.py

從 WSI 取幾個代表性區塊，跑 Cellpose 並把偵測結果疊圖存檔，
供目視確認分割是否準確。

用法：
  python test_cellpose.py                          # 預設參數
  python test_cellpose.py --diameter 40            # 指定細胞直徑 (px)
  python test_cellpose.py --model cyto3            # 換模型
  python test_cellpose.py --wsi /path/to/a.mrxs   # 指定 WSI

輸出：
  cell_cls/runs/cellpose_test/
    patch_<n>_raw.jpg      -- 原始 tile
    patch_<n>_overlay.jpg  -- 偵測結果疊圖（彩色 mask + bbox）
    patch_<n>_bbox.jpg     -- 只畫 bbox（比較乾淨）
    summary.json           -- 各 patch 偵測數量統計
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
from skimage.measure import regionprops
from skimage.color import label2rgb
import openslide
from cellpose import models as cp_models

# ── 預設參數 ────────────────────────────────────────────────────────────────── #
DEFAULT_WSI   = '/work/u4001296/project1/data/wsi/0000201584.mrxs'
DEFAULT_OUT   = '/work/u4001296/project1/cell_cls/runs/cellpose_test'
PATCH_SIZE    = 1024          # 取樣 tile 大小（比訓練時小，方便快速測試）
N_PATCHES     = 6             # 隨機取幾個 patch（+ 1 個固定中心 patch）
RANDOM_SEED   = 42


# ── 工具函式 ────────────────────────────────────────────────────────────────── #

def is_tissue(patch_pil, min_std: float = 15.0, max_mean: float = 235.0) -> bool:
    """
    判斷 patch 是否含有組織（非背景）。
    背景通常是接近純白（mean > 235）或純黑（std < 15）。
    """
    gray = np.array(patch_pil.convert('L'), dtype=np.float32)
    return float(gray.mean()) < max_mean and float(gray.std()) > min_std


def sample_patches(slide, n: int, size: int, seed: int, max_tries: int = 500):
    """
    在 WSI 有效範圍內取 n 個含組織的 patch 左上角座標。
    先固定取中心一塊，再隨機取 n 塊（跳過背景）。
    """
    W, H = slide.dimensions
    rng  = random.Random(seed)
    coords = []

    # 固定中心
    cx, cy = W // 2 - size // 2, H // 2 - size // 2
    coords.append((cx, cy))

    tries = 0
    while len(coords) - 1 < n and tries < max_tries:
        x = rng.randint(0, W - size - 1)
        y = rng.randint(0, H - size - 1)
        patch = slide.read_region((x, y), 0, (size, size)).convert('RGB')
        tries += 1
        if is_tissue(patch):
            coords.append((x, y))

    found = len(coords) - 1
    if found < n:
        print(f'  [WARNING] 只找到 {found}/{n} 個含組織的 patch（嘗試 {tries} 次）')
    return coords


def run_cellpose(img_np: np.ndarray, cp_model, diameter):
    """回傳 masks (H×W int), 各 cell props list"""
    masks, flows, styles = cp_model.eval(
        img_np, diameter=diameter, channels=[0, 0]
    )
    props = regionprops(masks)
    return masks, props


def draw_overlay(img_np: np.ndarray, masks: np.ndarray, props) -> np.ndarray:
    """彩色 mask 疊圖"""
    overlay = label2rgb(masks, image=img_np, bg_label=0, alpha=0.35)
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    # 畫每個 cell 的 bbox（紅框）
    pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil)
    for p in props:
        y0, x0, y1, x1 = p.bbox
        draw.rectangle([x0, y0, x1, y1], outline=(255, 50, 50), width=2)
    return np.array(pil)


def draw_bbox_only(img_np: np.ndarray, props) -> np.ndarray:
    """只畫 bbox，不改動原圖"""
    pil  = Image.fromarray(img_np.copy())
    draw = ImageDraw.Draw(pil)
    for p in props:
        y0, x0, y1, x1 = p.bbox
        draw.rectangle([x0, y0, x1, y1], outline=(0, 220, 0), width=2)
    return np.array(pil)


def save_figure(raw, overlay, bbox, out_prefix: str, patch_id: int,
                n_cells: int, diameter_used):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(raw);     axes[0].set_title('原始 patch')
    axes[1].imshow(overlay); axes[1].set_title(f'Cellpose mask overlay  (n={n_cells})')
    axes[2].imshow(bbox);    axes[2].set_title(f'BBox only  diameter={diameter_used:.1f}')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(f'{out_prefix}_compare.jpg', dpi=120, bbox_inches='tight')
    plt.close(fig)

    Image.fromarray(raw).save(f'{out_prefix}_raw.jpg',     quality=95)
    Image.fromarray(overlay).save(f'{out_prefix}_overlay.jpg', quality=95)
    Image.fromarray(bbox).save(f'{out_prefix}_bbox.jpg',    quality=95)


# ── main ────────────────────────────────────────────────────────────────────── #

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wsi',      default=DEFAULT_WSI)
    ap.add_argument('--out',      default=DEFAULT_OUT)
    ap.add_argument('--model',    default='cyto2',
                    choices=['cyto', 'cyto2', 'cyto3'])
    ap.add_argument('--diameter', type=float, default=None,
                    help='Cellpose 細胞直徑 (px)，None = 自動估計')
    ap.add_argument('--n',        type=int, default=N_PATCHES,
                    help='隨機取樣 patch 數量（不含中心 patch）')
    ap.add_argument('--size',     type=int, default=PATCH_SIZE,
                    help='每個 patch 的大小 (px)')
    ap.add_argument('--gpu',      action='store_true', default=True)
    ap.add_argument('--no_gpu',   action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    use_gpu = args.gpu and not args.no_gpu

    try:
        import torch
        use_gpu = use_gpu and torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    os.makedirs(args.out, exist_ok=True)

    print(f'WSI    : {args.wsi}')
    print(f'Model  : {args.model}  diameter={args.diameter}  GPU={use_gpu}')
    print(f'Output : {args.out}')

    slide = openslide.OpenSlide(args.wsi)
    W, H  = slide.dimensions
    print(f'Slide  : {W} x {H} px (level 0)')

    coords = sample_patches(slide, args.n, args.size, RANDOM_SEED)

    print(f'\nLoading Cellpose ({args.model})...')
    cp_model = cp_models.CellposeModel(gpu=use_gpu, model_type=args.model)

    summary = []
    for i, (x, y) in enumerate(coords):
        patch_pil = slide.read_region((x, y), 0, (args.size, args.size)).convert('RGB')
        patch_np  = np.array(patch_pil)
        label     = 'center' if i == 0 else f'rand{i}'
        print(f'\n[patch {i+1}/{len(coords)}]  ({x}, {y})  label={label}')

        masks, props = run_cellpose(patch_np, cp_model, args.diameter)

        # 估計平均直徑
        diameters = [np.sqrt((p.bbox[2]-p.bbox[0]) * (p.bbox[3]-p.bbox[1]))
                     for p in props] if props else [0]
        avg_diam  = float(np.mean(diameters))
        print(f'  → {len(props)} cells detected  avg_diameter≈{avg_diam:.1f}px')

        out_prefix = os.path.join(args.out, f'patch_{i+1:02d}_{label}')
        overlay = draw_overlay(patch_np, masks, props)
        bbox    = draw_bbox_only(patch_np, props)
        save_figure(patch_np, overlay, bbox, out_prefix, i,
                    len(props), avg_diam)

        summary.append({
            'patch_id':    i + 1,
            'label':       label,
            'x': x, 'y': y,
            'n_cells':     len(props),
            'avg_diameter': round(avg_diam, 1),
        })
        print(f'  Saved → {out_prefix}_compare.jpg')

    slide.close()

    json_path = os.path.join(args.out, 'summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'wsi': os.path.basename(args.wsi),
                   'model': args.model,
                   'diameter_arg': args.diameter,
                   'patches': summary}, f, indent=2, ensure_ascii=False)
    print(f'\nSummary → {json_path}')

    total_cells = sum(p['n_cells'] for p in summary)
    avg_per_patch = total_cells / len(summary)
    print(f'Total cells detected: {total_cells}  '
          f'({avg_per_patch:.0f} per patch avg)')
    print('\nDone. 請開啟 _compare.jpg 目視確認分割品質。')


if __name__ == '__main__':
    main()