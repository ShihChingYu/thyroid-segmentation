#!/usr/bin/env python
"""Check 08 - why does the repo's HD differ from a standard HD95?

Runs BOTH implementations over identical masks:
  - ours:  95th percentile over BOUNDARY pixels (standard definition)
  - repo:  visualization/metrics.py, imported directly, which takes the
           95th percentile over ALL foreground pixels

Run from the repo root."""
import sys, glob, os
import numpy as np, cv2, torch

sys.path.insert(0, 'visualization')
from metrics import HausdorffDistance
sys.path.insert(0, '/workspace/amy/harness')
from score import hd95 as hd_ours

FOLD = sys.argv[1] if len(sys.argv) > 1 else '0'
SZ   = 224
repo = HausdorffDistance()

def hd_repo(p, g):
    tp = torch.from_numpy(p.astype(np.float32))[None, None]
    tg = torch.from_numpy(g.astype(np.float32))[None, None]
    return float(repo.compute(tp, tg))

def to224(m):
    return cv2.resize(m.astype(np.uint8), (SZ, SZ),
                      interpolation=cv2.INTER_NEAREST).astype(bool)

rows = []
for gf in sorted(glob.glob('data/tn3k/test-mask/*')):
    stem = os.path.splitext(os.path.basename(gf))[0]
    pf = f'results/test-TN3K/unet/fold{FOLD}/{stem}.jpg'
    if not os.path.exists(pf):
        continue
    p = to224(cv2.imread(pf, 0) > 127)
    g = to224(cv2.imread(gf, 0) > 127)
    if g.sum() == 0:
        continue
    rows.append((stem, p.sum() == 0, hd_ours(p, g), hd_repo(p, g)))

empty   = [r for r in rows if r[1]]
both    = np.array([(r[2], r[3]) for r in rows if not r[1] and r[2] is not None])

print(f'\nfold {FOLD}, {len(rows)} cases at {SZ}x{SZ}  '
      f'({len(empty)} with an empty prediction)\n')
print(f'  ours (boundary pixels)    mean={both[:,0].mean():7.3f}  '
      f'median={np.median(both[:,0]):7.3f}')
print(f'  repo (all foreground)     mean={both[:,1].mean():7.3f}  '
      f'median={np.median(both[:,1]):7.3f}')
print(f'  ratio ours/repo           {both[:,0].mean()/both[:,1].mean():.3f}')
print(f'  repo <= ours in           {np.mean(both[:,1] <= both[:,0]):.1%} of cases')

k = np.argsort(-both[:, 0])[:5]
idx = [r for r in rows if not r[1] and r[2] is not None]
print('\n  worst 5 by our HD95:')
for j in k:
    print(f'    {idx[j][0]}   ours={both[j,0]:8.2f}   repo={both[j,1]:8.2f}')
