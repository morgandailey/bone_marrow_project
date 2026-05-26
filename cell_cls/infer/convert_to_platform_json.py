"""
convert_to_platform_json.py

將 wsi_detect_classify.py 輸出的 *_candidates.json
轉換成平台可吃的 annotation JSON 格式（一個 WSI 一個檔）。

用法：
  # 單檔
  python convert_to_platform_json.py \
      --input runs/wsi_scan/S23_candidates.json

  # 整個資料夾（自動找所有 *_candidates.json）
  python convert_to_platform_json.py \
      --input runs/wsi_scan/ \
      --output_dir platform_json/
"""

import json
import argparse
import os

# ── 短代碼 → 平台 class name ──────────────────────────────────────────────── #
CLASS_NAME_MAP = {
    'MON': 'monocyte',
    'BAS': 'basophil',
    'LYT': 'lymphocyte',
    'NGS': 'neutrophil_segmented',
    'NGB': 'neutrophil_band',
    'MMZ': 'metamyelocyte',
    'MYB': 'myelocyte',
    'EOS': 'eosinophil',
    'EBO': 'erythroblast',
    'PEB': 'proerythroblast',
    'PMO': 'promyelocyte',
    'PLM': 'plasma_cell',
    'BLA': 'blast',
    'NIF': 'immature_cell',
    'HAC': 'hairy_cell',
    'OTH': 'other',
    'ART': 'artifact',
}


# ── 單顆 cell → annotation entry ─────────────────────────────────────────── #

def cell_to_annotation(cell: dict, index: int) -> dict:
    x, y, w, h  = cell['x'], cell['y'], cell['w'], cell['h']
    cls_code     = cell['cls']
    cls_name     = CLASS_NAME_MAP.get(cls_code, cls_code.lower())

    entry = {
        'name':        cls_name,
        'caption':     cls_name.replace('_', ' ').title(),
        'type':        'MultiPolygon',
        'index':       index,
        'partOfGroup': 'DEFAULT',
        'coordinates': [
            [x,     y    ],   # 左上
            [x,     y + h],   # 左下
            [x + w, y + h],   # 右下
            [x + w, y    ],   # 右上
        ],
        'temporary': False,
    }

    # top3 機率（平台目前不吃，但留著之後用）
    if 'top3' in cell:
        entry['probabilities'] = [
            {'cls': CLASS_NAME_MAP.get(c, c.lower()), 'prob': round(p, 4)}
            for c, p in cell['top3']
        ]

    return entry


# ── 轉換單一 JSON 檔 ──────────────────────────────────────────────────────── #

def convert(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', [])
    if not results:
        print(f'  [WARN] 沒有 results，跳過 {input_path}')
        return

    # ROI = 所有候選 cell 的外框
    x_min = min(c['x']           for c in results)
    y_min = min(c['y']           for c in results)
    x_max = max(c['x'] + c['w'] for c in results)
    y_max = max(c['y'] + c['h'] for c in results)

    roi = {
        'name':        'roi',
        'caption':     'Roi',
        'type':        'MultiPolygon',
        'index':       0,
        'partOfGroup': 'DEFAULT',
        'coordinates': [
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_max],
            [x_max, y_min],
        ],
        'temporary': False,
    }

    annotations = [roi] + [
        cell_to_annotation(c, i) for i, c in enumerate(results, start=1)
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'annotation': annotations}, f, indent=4, ensure_ascii=False)

    # 統計各類別數量
    from collections import Counter
    counts = Counter(c['cls'] for c in results)
    count_str = '  '.join(f'{k}={v}' for k, v in sorted(counts.items()))
    print(f'  → {output_path}')
    print(f'     {len(results)} cells  [{count_str}]')


# ── CLI ──────────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(
        description='wsi_detect_classify 輸出 JSON → 平台 annotation JSON'
    )
    ap.add_argument(
        '--input', required=True,
        help='*_candidates.json 檔案，或包含多個 candidates JSON 的目錄'
    )
    ap.add_argument(
        '--output_dir', default='platform_json',
        help='輸出目錄（預設 platform_json/）'
    )
    args = ap.parse_args()

    if os.path.isdir(args.input):
        files = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith('_candidates.json')
        )
        if not files:
            print(f'[ERROR] 找不到 *_candidates.json in {args.input}')
            return
    else:
        files = [args.input]

    for fpath in files:
        stem    = os.path.basename(fpath).replace('_candidates.json', '')
        out     = os.path.join(args.output_dir, f'{stem}.json')
        print(f'Converting {os.path.basename(fpath)} ...')
        convert(fpath, out)

    print('\nDone.')


if __name__ == '__main__':
    main()