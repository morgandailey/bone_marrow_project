import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
import timm
import json

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from configs import stage1_config as cfg
from dataset import BoneMarrowDataset, CLASSES, ALL_FOLDERS, MERGE_MAP, CLASS_TO_IDX
from dataset import TRAIN_TRANSFORM, VAL_TRANSFORM


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_all_samples(data_dir):
    labels = []
    for folder in ALL_FOLDERS:
        cls_dir = os.path.join(data_dir, folder)
        if not os.path.isdir(cls_dir):
            continue
        target_cls = MERGE_MAP.get(folder, folder)
        label = CLASS_TO_IDX[target_cls]
        for root, _, files in os.walk(cls_dir):
            for fname in files:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    labels.append(label)
    return labels


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def build_model(model_name, num_classes):
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_n = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        total_n += imgs.size(0)
    return total_loss / total_n


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, total_n = 0.0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        total_n += imgs.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / total_n
    macro_f1 = f1_score(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        average='macro', zero_division=0
    )
    return avg_loss, macro_f1


def main():
    set_seed(cfg['seed'])
    os.makedirs(cfg['output_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_labels = collect_all_samples(cfg['data_dir'])
    all_indices = list(range(len(all_labels)))

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=cfg['val_ratio'], random_state=cfg['seed']
    )
    train_idx, val_idx = next(splitter.split(all_indices, all_labels))

    train_ds = BoneMarrowDataset(cfg['data_dir'], train_idx, transform=TRAIN_TRANSFORM)
    val_ds   = BoneMarrowDataset(cfg['data_dir'], val_idx,   transform=VAL_TRANSFORM)

    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=True
    )

    criterion = FocalLoss(gamma=cfg['focal_gamma'])
    model = build_model(cfg['model_name'], cfg['num_classes']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=1e-6
    )
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])

    best_f1 = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_f1 = validate(model, val_loader, criterion, device, cfg['num_classes'])
        scheduler.step()

        print(f"Epoch {epoch:02d}/{cfg['epochs']}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}")

        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss, 'val_f1': val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_f1': val_f1,
                'classes': CLASSES,
            }, os.path.join(cfg['output_dir'], 'best.pt'))
            print(f"  -> saved best model (f1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

    with open(os.path.join(cfg['output_dir'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best val F1 = {best_f1:.4f}")


if __name__ == '__main__':
    main()