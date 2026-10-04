#!/usr/bin/env python
"""Check 09 - validate the scorer against predictions whose answer is known.
Run from the repo root."""
import sys, glob, os, tempfile
import numpy as np, cv2
sys.path.insert(0, '/workspace/amy/harness')
from score import dice, iou, hd95

SZ = 224
r = lambda m: cv2.resize(m.astype(np.uint8), (SZ, SZ),
                         interpolation=cv2.INTER_NEAREST).astype(bool)

gts = sorted(glob.glob('data/tn3k/test-mask/*'))
A, B = [], []
tmp = tempfile.mkdtemp()

for gf in gts:
    g = r(cv2.imread(gf, 0) > 127)
    if g.sum() == 0:
        continue
    A.append((dice(g, g), iou(g, g), hd95(g, g)))           # perfect
    z = np.zeros_like(g)
    B.append((dice(z, g), iou(z, g), hd95(z, g)))           # null

a = np.array(A, dtype=float)
print(f'\nn = {len(A)} cases at {SZ}x{SZ}\n')
print('A  PERFECT model (pred == ground truth)   expect 1.0, 1.0, 0.0')
print(f'   dice min={a[:,0].min():.6f}   iou min={a[:,1].min():.6f}   '
      f'hd95 max={np.nanmax(a[:,2]):.6f}')
print(f'   cases not exactly 1.0: {(a[:,0] != 1.0).sum()}')

b0 = [x[0] for x in B]; b2 = [x[2] for x in B]
print('\nB  NULL model (all zeros)                 expect 0.0, undefined')
print(f'   dice max={max(b0):.6f}   hd95 defined in {sum(x is not None for x in b2)} cases')

