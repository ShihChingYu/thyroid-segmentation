#!/usr/bin/env python
"""Dataset inventory: image counts, empty masks, dimension ranges.
Run from the repo root."""
import glob, os
import numpy as np
from PIL import Image

def ddti_dirs():
    root = next((p for p in ('data/DDTI', 'data/ddti') if os.path.isdir(p)), None)
    if not root:
        return None
    img = next((f'{root}/{s}' for s in ('image', 'images', 'p_image')
                if os.path.isdir(f'{root}/{s}')), None)
    msk = next((f'{root}/{s}' for s in ('mask', 'masks', 'p_mask')
                if os.path.isdir(f'{root}/{s}')), None)
    return img, msk

sets = [('tn3k trainval', 'data/tn3k/trainval-image', 'data/tn3k/trainval-mask'),
        ('tn3k test',     'data/tn3k/test-image',     'data/tn3k/test-mask'),
        ('tg3k',          'data/tg3k/thyroid-image',  'data/tg3k/thyroid-mask')]
dd = ddti_dirs()
sets.append(('ddti', dd[0], dd[1]) if dd and all(dd) else ('ddti', None, None))

print(f'{"dataset":<15} {"images":>7} {"masks":>7} {"empty":>6}  dimensions')
print('-' * 72)
for name, idir, mdir in sets:
    if not mdir or not os.path.isdir(mdir):
        print(f'{name:<15} {"-":>7} {"-":>7} {"-":>6}  NOT FOUND'); continue
    imgs = glob.glob(f'{idir}/*') if idir else []
    mf = sorted(f for f in glob.glob(f'{mdir}/*') if not f.endswith('.json'))
    empty, hs, ws = 0, [], []
    for f in mf:
        a = np.array(Image.open(f).convert('L'))
        hs.append(a.shape[0]); ws.append(a.shape[1])
        if (a > 127).sum() == 0:
            empty += 1
    print(f'{name:<15} {len(imgs):>7} {len(mf):>7} {empty:>6}  '
          f'h {min(hs)}-{max(hs)}   w {min(ws)}-{max(ws)}')

print('\nempty mask = no pixel above 127. Zero empties means the Dice 0/0 case')
print('never arises on these datasets; the convention must still be fixed for')
print('the patient data, and is tested synthetically in check 07.')
