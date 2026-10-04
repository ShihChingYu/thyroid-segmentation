#!/usr/bin/env python
"""Recover TG3K video grouping from image resolution, and test whether the
shipped train/val split respects video boundaries. Run from the repo root."""
import glob, json, os
import numpy as np
from PIL import Image

# A. video groups = contiguous runs of constant resolution
files, runs, prev = sorted(glob.glob('data/tg3k/thyroid-mask/*')), [], None
for f in files:
    s = np.array(Image.open(f).convert('L')).shape
    i = int(os.path.splitext(os.path.basename(f))[0])
    if s != prev: runs.append({'shape': s, 'a': i, 'b': i}); prev = s
    else:         runs[-1]['b'] = i
print(f'A. {len(runs)} constant-resolution runs for {len(files)} frames '
      f'(paper: 16 source videos)\n')

# B. the shipped split
d = json.load(open('data/tg3k/tg3k-trainval.json'))
train, val = set(d['train']), set(d['val'])
print(f'B. shipped split: train={len(train)} (0-{max(train)}) '
      f'val={len(val)} ({min(val)}-{max(val)})\n')

# C. does any video straddle the boundary?
print('C. video membership:')
leak = 0
for k, r in enumerate(runs, 1):
    idx = set(range(r['a'], r['b'] + 1))
    nt, nv = len(idx & train), len(idx & val)
    if nt and nv: leak += nv
    print(f'   video {k:>2}  {str(r["shape"]):<12} {r["a"]:>5}-{r["b"]:<5} '
          f'train={nt:<5} val={nv:<5}' + ('  <-- STRADDLES' if nt and nv else ''))
print(f'\n   leaked val frames: {leak}/{len(val)} = {leak/len(val):.1%}')
