import os
import json
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score, classification_report
import timm
import openslide

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from configs import stage1_config as cfg
from dataset import CLASSES, CLASS_TO_IDX, VAL_TRANSFORM

# 標注名稱 → 模型類別（Stage 1 的 17 類）
LABEL_MAP = {
    'normoblast':         'EBO',
    'band neutrophil':    'NGB',
    'metamyelocyte':      'MMZ',
    'myelocyte':          'MYB',
    'lymphocyte':         'LYT',
    'eosinophil':         'EOS',
    'segmented neutrophil': 'NGS',
    'plasma cell':        'PLM',
    'myeloblast':         'BLA',
    'promyelocyte':       'PMO',
    'basophil':           'BAS',
    'monocyte':           'MON',
}
SKIP = {'unknown', 'deleted', 'roi', 'tumor', 'megakayocyte'}

WSI_FILES = [
    ('/work/u4001296/project1/data/wsi/0000201584.mrxs',
     '/work/u4001296/project1/data/wsi/0000201584_20251103.json'),
    ('/work/u4001296/project1/data/wsi/slide-2025-10-01T13-44-33-R1-S23.mrxs',
     '/work/u4001296/project1/data/wsi/slide-2025-10-01T13-44-33-R1-S23.json'),
]

PADDING = 10  # 裁切時四周多留幾個 pixel


def load_annotations(json_path):
    with open(json_path) as f:
        data = json.load(f)
    cells = []
    for a in data['annotation']:
        if a['type'] != 'rectangle':
            continue
        name = a['name']
        if name in SKIP or name not in LABEL_MAP:
            continue
        coords = a['coordinates']
        x0 = min(c[0] for c in coords)
        y0 = min(c[1] for c in coords)
        x1 = max(c[0] for c in coords)
        y1 = max(c[1] for c in coords)
        cells.append({'label': LABEL_MAP[name], 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})
    return cells


def crop_cell(slide, cell, padding=PADDING):
    x0 = max(0, cell['x0'] - padding)
    y0 = max(0, cell['y0'] - padding)
    x1 = cell['x1'] + padding
    y1 = cell['y1'] + padding
    w, h = x1 - x0, y1 - y0
    region = slide.read_region((x0, y0), 0, (w, h)).convert('RGB')
    return region


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(os.path.join(cfg['output_dir'], 'best.pt'), map_location=device)
    model = timm.create_model(cfg['model_name'], pretrained=False, num_classes=cfg['num_classes'])
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()

    all_preds, all_labels = [], []

    for wsi_path, json_path in WSI_FILES:
        print(f'\n{os.path.basename(wsi_path)}')
        slide = openslide.OpenSlide(wsi_path)
        cells = load_annotations(json_path)
        print(f'  Valid cells: {len(cells)}')

        for cell in cells:
            img = crop_cell(slide, cell)
            img_t = VAL_TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                logits = model(img_t)
            pred_idx = logits.argmax(dim=1).item()
            pred_cls = CLASSES[pred_idx]
            true_cls = cell['label']
            if true_cls not in CLASS_TO_IDX:
                continue
            all_preds.append(pred_idx)
            all_labels.append(CLASS_TO_IDX[true_cls])

        slide.close()

    all_class_idx = list(range(cfg['num_classes']))
    print(f'\n=== WSI Test Results ({len(all_labels)} cells) ===')
    print(classification_report(
        all_labels, all_preds,
        labels=all_class_idx, target_names=CLASSES, digits=3
    ))
    per_class_f1 = f1_score(all_labels, all_preds, labels=all_class_idx,
                            average=None, zero_division=0)
    macro_f1 = float(np.mean(per_class_f1))
    print(f'Macro F1 = {macro_f1:.4f}')

    out = {
        'total_cells': len(all_labels),
        'macro_f1': macro_f1,
        'per_class_f1': {CLASSES[i]: float(per_class_f1[i]) for i in range(len(CLASSES))},
    }
    out_path = os.path.join(cfg['output_dir'], 'wsi_test_report.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()