import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)
import timm
import json

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
from configs import stage1_config as cfg
from dataset import BoneMarrowDataset, CLASSES, VAL_TRANSFORM
from train_stage1 import collect_all_samples


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # reproduce same val split
    all_labels = collect_all_samples(cfg['data_dir'])
    all_indices = list(range(len(all_labels)))
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=cfg['val_ratio'], random_state=cfg['seed']
    )
    _, val_idx = next(splitter.split(all_indices, all_labels))

    val_ds = BoneMarrowDataset(cfg['data_dir'], val_idx, transform=VAL_TRANSFORM)
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=True
    )

    ckpt_path = os.path.join(cfg['output_dir'], 'best.pt')
    ckpt = torch.load(ckpt_path, map_location=device)

    model = timm.create_model(cfg['model_name'], pretrained=False, num_classes=cfg['num_classes'])
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()

    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels_list.extend(labels.numpy())

    all_class_idx = list(range(cfg['num_classes']))
    print(f"\nBest epoch: {ckpt['epoch']}  val_f1={ckpt['val_f1']:.4f}\n")
    print("=== Per-class F1 ===")
    report = classification_report(
        all_labels_list, all_preds,
        labels=all_class_idx, target_names=CLASSES, digits=3
    )
    print(report)

    cm = confusion_matrix(all_labels_list, all_preds, labels=all_class_idx)
    per_class_f1 = f1_score(all_labels_list, all_preds, labels=all_class_idx, average=None, zero_division=0)
    sorted_idx = np.argsort(per_class_f1)

    print("=== Worst 5 classes ===")
    for i in sorted_idx[:5]:
        print(f"  {CLASSES[i]:4s}  F1={per_class_f1[i]:.3f}  "
              f"support={cm[i].sum()}")

    print("\n=== 每類被猜成什麼（前 3 名）===")
    for i, cls in enumerate(CLASSES):
        row = cm[i]
        total = row.sum()
        if total == 0:
            continue
        top3 = np.argsort(row)[::-1][:3]
        parts = [f"{CLASSES[j]}:{row[j]}({row[j]/total*100:.0f}%)" for j in top3 if row[j] > 0]
        print(f"  {cls:4s} → {', '.join(parts)}")

    out = {
        'epoch': ckpt['epoch'],
        'val_f1': ckpt['val_f1'],
        'per_class_f1': {CLASSES[i]: float(per_class_f1[i]) for i in range(len(CLASSES))},
    }
    out_path = os.path.join(cfg['output_dir'], 'eval_report.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()