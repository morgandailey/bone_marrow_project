"""
產生兩張 PPT 用圖（16:9，字大，不擠）
  runs/ppt_f1_heatmap.png     — 三個模型 × 17 類的 F1 heatmap
  runs/ppt_data_count.png     — wsi_crops 各類別 train / val 數量
"""
import json, os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── 共用設定 ─────────────────────────────────────────────────────────────── #
CLASSES = sorted(['ART','BAS','BLA','EBO','EOS','HAC','LYT','MMZ',
                  'MON','MYB','NGB','NGS','NIF','OTH','PEB','PLM','PMO'])
HIGHLIGHT = {'MON','BAS','BLA','PMO'}   # 臨床重點

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  圖 1：F1 Heatmap                                                          #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
models = [
    ('Stage 1 v3\n(MLL, F1=0.807)',  'runs/stage1_v3/eval_report.json'),
    ('Stage 2 v2\n(WSI, F1=0.429)',  'runs/stage2_v2/eval_report.json'),
    ('Stage 2 v3\n(WSI, F1=0.480)', 'runs/stage2_v3/eval_report.json'),
]

matrix = []
for _, path in models:
    with open(path) as f:
        d = json.load(f)
    matrix.append([d['per_class_f1'].get(c, 0.0) for c in CLASSES])
mat = np.array(matrix)   # (3, 17)

fig, ax = plt.subplots(figsize=(20, 5.2))
fig.patch.set_facecolor('#F7F9FC')
ax.set_facecolor('#F7F9FC')

cmap = plt.cm.RdYlGn
im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')

# 數值標籤
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        txt_color = 'white' if v < 0.25 or v > 0.80 else '#222222'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                fontsize=10.5, fontweight='bold', color=txt_color)

# 軸標籤
ax.set_xticks(range(len(CLASSES)))
xlabels = [f'[{c}]' if c in HIGHLIGHT else c for c in CLASSES]
ax.set_xticklabels(xlabels, fontsize=11.5)
ax.set_yticks(range(len(models)))
ax.set_yticklabels([m[0] for m in models], fontsize=12)

# 臨床重點欄 highlight（框線）
for j, cls in enumerate(CLASSES):
    if cls in HIGHLIGHT:
        rect = plt.Rectangle((j - 0.5, -0.5), 1, mat.shape[0],
                              linewidth=2.5, edgecolor='#E74C3C',
                              facecolor='none', zorder=3)
        ax.add_patch(rect)

# colorbar
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.018, pad=0.02)
cbar.set_label('F1 Score', fontsize=11)
cbar.ax.tick_params(labelsize=10)

ax.set_title('Per-Class F1 Score — Model Comparison\n'
             '(stage1_v3 evaluated on MLL val  |  stage2_v2/v3 evaluated on WSI val)\n'
             '[ ] = Clinical priority classes',
             fontsize=13, fontweight='bold', pad=14)

ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

plt.tight_layout()
out1 = 'runs/ppt_f1_heatmap.png'
fig.savefig(out1, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Saved -> {out1}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  圖 2：Data Count（train / val 堆疊 + 80/20 split 後實際數量）              #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
MERGE_MAP  = {'ABE':'EOS','LYI':'LYT','KSC':'OTH','FGC':'OTH'}
ALL_FOLDERS = sorted(['ABE','ART','BAS','BLA','EBO','EOS','FGC','HAC','KSC',
                       'LYI','LYT','MMZ','MON','MYB','NGB','NGS','NIF','OTH',
                       'PEB','PLM','PMO'])
CROPS_DIR = 'wsi_crops'

disk = {'train': defaultdict(int), 'val': defaultdict(int)}
for split in ('train', 'val'):
    for folder in ALL_FOLDERS:
        d = os.path.join(CROPS_DIR, split, folder)
        if not os.path.isdir(d): continue
        target = MERGE_MAP.get(folder, folder)
        cnt = len([f for f in os.listdir(d) if f.lower().endswith('.jpg')])
        disk[split][target] += cnt

# 合計後 80/20 split（train_stage2.py 的行為）
totals  = {c: disk['train'][c] + disk['val'][c] for c in CLASSES}
tr_real = {c: int(totals[c] * 0.8) for c in CLASSES}
va_real = {c: totals[c] - tr_real[c]            for c in CLASSES}

# 依合計降冪排序
order = sorted(CLASSES, key=lambda c: -totals[c])
tr_vals = [tr_real[c] for c in order]
va_vals = [va_real[c] for c in order]
tots    = [totals[c]  for c in order]

x = np.arange(len(order))
w = 0.55

fig, ax = plt.subplots(figsize=(20, 6.5))
fig.patch.set_facecolor('#F7F9FC')
ax.set_facecolor('#F7F9FC')

b1 = ax.bar(x, tr_vals, width=w, color='#4C72B0', label='Train (~80%)', zorder=3)
b2 = ax.bar(x, va_vals, width=w, bottom=tr_vals,
            color='#DD8452', alpha=0.85, label='Val (~20%)', zorder=3)

# 數值標籤：total on top
for xi, (tv, tot) in enumerate(zip(tr_vals, tots)):
    ax.text(xi, tot + 8, str(tot), ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#2C3E50')

# 臨床重點標記
for xi, cls in enumerate(order):
    if cls in HIGHLIGHT:
        ax.get_xticklabels()   # force render
        ax.axvspan(xi - 0.45, xi + 0.45, alpha=0.12, color='#E74C3C', zorder=1)

ax.set_xticks(x)
xlbls = [f'[{c}]' if c in HIGHLIGHT else c for c in order]
ax.set_xticklabels(xlbls, fontsize=12)
ax.set_ylabel('Number of Images', fontsize=12)
ax.set_title('Training Data Distribution — wsi_crops\n'
             '(Total merged, then 80/20 stratified split by train_stage2.py)\n'
             '[ ] = Clinical priority classes',
             fontsize=13, fontweight='bold', pad=12)

ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.set_ylim(0, max(tots) * 1.15)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
out2 = 'runs/ppt_data_count.png'
fig.savefig(out2, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Saved -> {out2}')
