import glob, os, re, csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = '/workspace/amy/runs'
EPOCH_RE = re.compile(
    r'epoch:\s*(\d+),.*?testloss:\s*([\d.]+)\s*mmae:\s*([\d.]+)'
    r'\s*fbeta:\s*([\d.]+)\s*iou:\s*([\d.]+)')

SERIES = ['#2a78d6', '#eb6834', '#1baf7a']      # validated categorical slots 1-3
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'

def parse(path):
    rows = []
    for line in open(path, errors='ignore'):
        m = EPOCH_RE.search(line)
        if m:
            e, loss, mmae, fbeta, iou = m.groups()
            rows.append(dict(epoch=int(e), loss=float(loss), mmae=float(mmae),
                             fbeta=float(fbeta), iou=float(iou)))
    return rows

runs = {}
for path in sorted(glob.glob(f'{RUNS}/*.log')):
    name = os.path.basename(path).replace('.log', '')
    rows = parse(path)
    if not rows:
        continue
    runs[name] = rows
    with open(f'{RUNS}/{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

print(f'{"run":<34}{"epochs":>7}{"best IoU":>10}{"@ep":>6}{"final":>8}')
for name, rows in runs.items():
    best = max(rows, key=lambda r: r['iou'])
    print(f'{name:<34}{len(rows):>7}{best["iou"]:>10.4f}{best["epoch"]:>6}{rows[-1]["iou"]:>8.4f}')

fig, ax = plt.subplots(figsize=(9, 5), dpi=150, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for i, (name, rows) in enumerate(runs.items()):
    c = SERIES[i % len(SERIES)]
    x = [r['epoch'] for r in rows]; y = [r['iou'] for r in rows]
    label = name.replace('-unet-tn3k', '').replace('-80ep', '')
    ax.plot(x, y, color=c, linewidth=2, label=label, solid_capstyle='round')
    ax.annotate(label, (x[-1], y[-1]), xytext=(6, 0), textcoords='offset points',
                color=c, fontsize=9, va='center')

ax.set_xlabel('Epoch', color=MUTED, fontsize=10)
ax.set_ylabel('Validation IoU', color=MUTED, fontsize=10)
ax.set_title('TN3K U-Net — validation IoU by epoch', color=INK, fontsize=13,
             loc='left', pad=14)
ax.grid(axis='y', color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9)
ax.legend(frameon=False, loc='lower right', fontsize=9, labelcolor=MUTED)
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
fig.subplots_adjust(right=0.86)
fig.savefig(f'{RUNS}/training_curves.png', bbox_inches='tight', facecolor=SURFACE)
print(f'\nfigure: {RUNS}/training_curves.png')
