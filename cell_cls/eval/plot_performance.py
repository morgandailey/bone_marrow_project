"""
繪製各版本模型的 per-class F1 比較圖
輸出：cell_cls/runs/performance_comparison.png
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 資料 ─────────────────────────────────────────────────────────────────── #
models = {
    'Stage1 v3\n(MLL, F1=0.807)': {
        'path': 'runs/stage1_v3/eval_report.json',
        'color': '#4C72B0',
    },
    'Stage2 v2\n(WSI, F1=0.429)': {
        'path': 'runs/stage2_v2/eval_report.json',
        'color': '#DD8452',
    },
    'Stage2 v3\n(WSI, F1=0.480) *': {
        'path': 'runs/stage2_v3/eval_report.json',
        'color': '#55A868',
    },
}

data = {}
for label, info in models.items():
    with open(info['path']) as f:
        d = json.load(f)
    data[label] = {'f1': d['per_class_f1'], 'color': info['color']}

classes = sorted(data[list(data.keys())[0]]['f1'].keys())
n_cls   = len(classes)
n_mod   = len(models)

# 臨床重點類別標記
highlight = {'MON', 'BAS', 'BLA', 'PMO'}

# ── 圖表設定 ─────────────────────────────────────────────────────────────── #
fig, axes = plt.subplots(
    n_cls, 1,
    figsize=(13, n_cls * 1.15),
    constrained_layout=True
)
fig.suptitle('Per-Class F1 Score — Model Comparison\n'
             '(stage1_v3: MLL val  |  stage2_v2 / v3: WSI val)',
             fontsize=13, fontweight='bold', y=1.005)

bar_h  = 0.22
colors = [info['color'] for info in models.values()]
labels = list(models.keys())

for i, cls in enumerate(classes):
    ax = axes[i]
    vals = [data[lbl]['f1'].get(cls, 0.0) for lbl in labels]

    offsets = np.array([-bar_h, 0, bar_h])
    for j, (v, c) in enumerate(zip(vals, colors)):
        bar = ax.barh(offsets[j], v, height=bar_h * 0.85,
                      color=c, alpha=0.88, edgecolor='white', linewidth=0.5)
        if v > 0.02:
            ax.text(v + 0.01, offsets[j], f'{v:.3f}',
                    va='center', ha='left', fontsize=7.5, color='#333333')

    # 背景 highlight
    is_hl = cls in highlight
    ax.set_facecolor('#FFF8E7' if is_hl else '#FAFAFA')

    # 軸設定
    ax.set_xlim(0, 1.13)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='x', labelsize=7)
    ax.axvline(0.5, color='#BBBBBB', linewidth=0.6, linestyle='--')
    ax.axvline(0.8, color='#DDDDDD', linewidth=0.6, linestyle=':')

    # 類別標籤
    star = ' *' if is_hl else ''
    ax.set_ylabel(f'{cls}{star}', rotation=0, labelpad=52,
                  va='center', fontsize=9,
                  fontweight='bold' if is_hl else 'normal',
                  color='#B03A2E' if is_hl else '#2C3E50')

    # 只有最後一格顯示 x 軸 label
    if i < n_cls - 1:
        ax.set_xticklabels([])

axes[-1].set_xlabel('F1 Score', fontsize=9)

# legend
patches = [mpatches.Patch(color=c, label=lbl.replace('\n', ' '))
           for lbl, c in zip(labels, colors)]
fig.legend(handles=patches, loc='upper right',
           bbox_to_anchor=(1.0, 1.0), fontsize=8.5,
           framealpha=0.9, edgecolor='#CCCCCC')

# 說明
fig.text(0.01, -0.005,
         '* Clinical priority classes (MON/BAS/BLA/PMO)  |  '
         'dashed: F1=0.5  dotted: F1=0.8  |  * in legend = current best model',
         fontsize=7.5, color='#666666')

out = 'runs/performance_comparison.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
