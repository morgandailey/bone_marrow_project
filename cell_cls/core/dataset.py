import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 樣本 < 100 的類別併入相近的大類
MERGE_MAP = {
    'ABE': 'EOS',  # abnormal eosinophil → eosinophil
    'LYI': 'LYT',  # immature lymphocyte → lymphocyte
    'KSC': 'OTH',  # smudge cell → other
    'FGC': 'OTH',  # faggott cell → other
}

# 17 類（合併後）
CLASSES = sorted([
    'ART', 'BAS', 'BLA', 'EBO', 'EOS', 'HAC', 'LYT', 'MMZ',
    'MON', 'MYB', 'NGB', 'NGS', 'NIF', 'OTH', 'PEB', 'PLM', 'PMO'
])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# 磁碟上原始的 21 個資料夾
ALL_FOLDERS = sorted([
    'ABE', 'ART', 'BAS', 'BLA', 'EBO', 'EOS', 'FGC', 'HAC', 'KSC',
    'LYI', 'LYT', 'MMZ', 'MON', 'MYB', 'NGB', 'NGS', 'NIF', 'OTH',
    'PEB', 'PLM', 'PMO'
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class BoneMarrowDataset(Dataset):
    def __init__(self, data_dir, split_indices, transform=None):
        self.transform = transform
        all_samples = []
        for folder in ALL_FOLDERS:
            cls_dir = os.path.join(data_dir, folder)
            if not os.path.isdir(cls_dir):
                continue
            target_cls = MERGE_MAP.get(folder, folder)
            label = CLASS_TO_IDX[target_cls]
            for root, _, files in os.walk(cls_dir):
                for fname in files:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_samples.append((os.path.join(root, fname), label))
        self.samples = [all_samples[i] for i in split_indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        while True:
            path, label = self.samples[idx]
            try:
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, label
            except Exception:
                idx = (idx + 1) % len(self.samples)