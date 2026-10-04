#!/usr/bin/env python
"""Mask pixel values (is the ground truth binary?) and resolution structure.
Run from the repo root."""
import glob, os, collections
import numpy as np
from PIL import Image

def ddti_mask():
    root = next((p for p in ('data/DDTI', 'data/ddti') if os.path.isdir(p)), None)
    if not root:
        return None
    return next((f'{root}/{s}' for s in ('mask', 'masks', 'p_mask')
                 if os.path.isdir(f'{root}/{s}')), None)

sets = [('tn3k trainval', 'data/tn3k/trainval-mask'),
        ('tn3k test',     'data/tn3k/test-mask'),
        ('tg3k',          'data/tg3k/thyroid-mask'),
        ('ddti',          ddti_mask())]

for name, d in sets:
    if not d or not os.path.isdir(d):
        print(f'{name}: NOT FOUND\n'); continue
    files = sorted(f for f in glob.glob(f'{d}/*') if not f.endswith('.json'))
    vals, shapes, mid = set(), collections.Counter(), 0
    for f in files:
        a = np.array(Image.open(f).convert('L'))
        shapes[a.shape] += 1
        vals.update(np.unique(a).tolist())
        mid += int(((a > 50) & (a < 200)).sum() > 0)
    lo = sorted(v for v in vals if v < 128)[:5]
    hi = sorted(v for v in vals if v >= 128)[-5:]
    print(f'{name}: n={len(files)}  distinct_resolutions={len(shapes)}')
    print(f'   values  low={lo}  high={hi}')
    print(f'   masks with midtones (50-200): {mid}/{len(files)} '
          f'({mid/len(files):.0%})')
    print(f'   most common resolutions:')
    for s, c in shapes.most_common(5):
        print(f'      {s}  x{c}')
    print()

print('Binary ground truth would show values [0, 255] only. Values at 1-4 and')
print('251-254 are JPEG ringing; a binarization threshold (>127) is required')
print('and is recorded as a harness parameter. Midtones indicate genuinely soft')
print('boundaries, where the threshold choice changes the ground truth.')
print('Few distinct resolutions relative to file count indicates frames drawn')
print('from a small number of source videos (see check 03).')
