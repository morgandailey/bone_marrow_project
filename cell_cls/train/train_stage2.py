import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from PIL import Image
import timm

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from configs import stage2_config as cfg
from dataset import CLASSES, CLASS_TO_IDX, TRAIN_TRANSFORM, VAL_TRANSFORM

CROPS_DIR = cfg['crops_dir']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_all_samples():
    """Merge wsi_crops/train/ and wsi_crops/val/ into one list."""
    samples = []
    for split in ('train', 'val'):
        split_dir = os.path.join(CROPS_DIR, split)
        for cls in CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            label = CLASS_TO_IDX[cls]
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith('.jpg'):
                    samples.append((os.path.join(cls_dir, fname), label))
    return samples


class CropDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))
        if self.transform:
            img = self.transform(img)
        return img, label


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def main():
    set_seed(cfg['seed'])
    os.makedirs(cfg['output_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_samples = collect_all_samples()
    labels_only = [s[1] for s in all_samples]
    print(f'Total WSI crops: {len(all_samples)}')

    per_class = {}
    for _, lbl in all_samples:
        per_class[CLASSES[lbl]] = per_class.get(CLASSES[lbl], 0) + 1
    for cls, cnt in sorted(per_class.items(), key=lambda x: -x[1]):
        print(f'  {cls}: {cnt}')

    train_samples, val_samples = train_test_split(
        all_samples, test_size=cfg['val_ratio'],
        stratify=labels_only, random_state=cfg['seed']
    )
    print(f'\nTrain: {len(train_samples)}  Val: {len(val_samples)}')

    print('Loading Stage 1 checkpoint...')
    ckpt = torch.load(cfg['stage1_ckpt'], map_location=device)
    model = timm.create_model(cfg['model_name'], pretrained=False, num_classes=cfg['num_classes'])
    model.load_state_dict(ckpt['model_state'])

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)')

    model = model.to(device)

    train_ds = CropDataset(train_samples, transform=TRAIN_TRANSFORM)
    val_ds   = CropDataset(val_samples,   transform=VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=4, pin_memory=True)

    criterion = FocalLoss(gamma=cfg['focal_gamma'])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=1e-6
    )
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])

    best_f1, patience_counter = 0.0, 0
    history = []
    all_class_idx = list(range(cfg['num_classes']))

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)
            total_n += imgs.size(0)
        train_loss = total_loss / total_n

        model.eval()
        all_preds, all_labels = [], []
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                val_n += imgs.size(0)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= val_n
        val_f1 = f1_score(all_labels, all_preds, labels=all_class_idx,
                          average='macro', zero_division=0)
        scheduler.step()

        print(f'Epoch {epoch:02d}/{cfg["epochs"]}  '
              f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}')
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss, 'val_f1': val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_f1': val_f1, 'classes': CLASSES},
                       os.path.join(cfg['output_dir'], 'best.pt'))
            print(f'  -> saved best model (f1={best_f1:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f'Early stopping at epoch {epoch}')
                break

    print(f'\nDone. Best val F1 = {best_f1:.4f}')
    ckpt_best = torch.load(os.path.join(cfg['output_dir'], 'best.pt'), map_location=device)
    model.load_state_dict(ckpt_best['model_state'])
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds,
                                labels=all_class_idx, target_names=CLASSES, digits=3))

    per_class_f1 = f1_score(all_labels, all_preds, labels=all_class_idx,
                            average=None, zero_division=0)
    with open(os.path.join(cfg['output_dir'], 'eval_report.json'), 'w') as f:
        json.dump({
            'best_epoch': int(ckpt_best['epoch']),
            'best_val_f1': best_f1,
            'per_class_f1': {CLASSES[i]: float(per_class_f1[i]) for i in range(len(CLASSES))},
        }, f, indent=2)
    with open(os.path.join(cfg['output_dir'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()