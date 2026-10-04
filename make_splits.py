#!/usr/bin/env python
"""Generate splits/v1.json — the frozen split definition.
Run from the repo root. Once written this file is never edited;
changes become v2, with a recorded reason."""
import glob, json, os, hashlib, datetime
import numpy as np
from PIL import Image

ROOT = 'data'
OUT  = '/workspace/amy/splits/v1.json'

def ids(d):
    return sorted(int(os.path.splitext(os.path.basename(f))[0])
                  for f in glob.glob(f'{d}/*') if not f.endswith('.json'))

def digest(d):
    names = sorted(os.path.basename(f) for f in glob.glob(f'{d}/*'))
    return hashlib.sha256('|'.join(names).encode()).hexdigest()[:16]

S = {'version': 'v1',
     'created': datetime.date.today().isoformat(),
     'author': 'a.shih@soundblademedical.com',
     'policy': 'Frozen. Do not edit. Supersede with v2 and record why.'}

# --- TN3K: adopt official split and folds unchanged ---
S['tn3k'] = {
    'source': 'official (repo + Gong et al., CBM 2022)',
    'rationale': ('Authors state images from one patient appear in only one '
                  'subset. Patient IDs are removed from the release, so this '
                  'cannot be verified or improved locally. Adopted unchanged '
                  'to preserve comparability with published results. Folds '
                  'verified as a clean partition (check 04); assignment is '
                  'index % 5, not grouped.'),
    'test':     ids(f'{ROOT}/tn3k/test-image'),
    'trainval': ids(f'{ROOT}/tn3k/trainval-image'),
    'folds':    {os.path.basename(f): json.load(open(f))
                 for f in sorted(glob.glob(f'{ROOT}/tn3k/*fold*.json'))},
    'digest':   {'test': digest(f'{ROOT}/tn3k/test-image'),
                 'trainval': digest(f'{ROOT}/tn3k/trainval-image')}}

# --- TG3K: regenerate on video boundaries ---
files, runs, prev = sorted(glob.glob(f'{ROOT}/tg3k/thyroid-mask/*')), [], None
for f in files:
    s = np.array(Image.open(f).convert('L')).shape
    i = int(os.path.splitext(os.path.basename(f))[0])
    if s != prev: runs.append([i, i]); prev = s
    else:         runs[-1][1] = i
val   = sorted(i for a, b in runs[-2:] for i in range(a, b + 1)) # video 14 and 15
train = sorted(set(range(len(files))) - set(val)) # video 1-13, so no overlap of video between train and val
S['tg3k'] = {
    'source': 'regenerated - do NOT use data/tg3k/tg3k-trainval.json',
    'rationale': ('Shipped split cuts at index 3226, inside the video spanning '
                  '3026-3269: 200 of its frames land in train and 44 in val, so '
                  '12% of validation has near-duplicates in training. Video IDs '
                  'are not released; groups recovered from contiguous runs of '
                  'constant image resolution (15 runs, 16 videos per the paper).'),
    'groups': runs,
    'train': train,
    'val': val,
    'val_groups': runs[-2:],
    'shipped_split_kept_for_comparison': 'data/tg3k/tg3k-trainval.json',
    'digest': digest(f'{ROOT}/tg3k/thyroid-image')}

# --- DDTI: evaluation only ---
dd = next((p for p in (f'{ROOT}/DDTI', f'{ROOT}/ddti') if os.path.isdir(p)), None)
img = None
if dd:
    img = next((p for p in (f'{dd}/image', f'{dd}/images', f'{dd}/p_image')
                if os.path.isdir(p)), None)
S['ddti'] = {
    'source': 'evaluation only - never trained on, so no split is required',
    'path_found': img,
    'test': ids(img) if img else [],
    'digest': digest(img) if img else None}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(S, open(OUT, 'w'), indent=1)
print(f'wrote {OUT}')
print(f"  tn3k  trainval={len(S['tn3k']['trainval'])} test={len(S['tn3k']['test'])} "
      f"folds={len(S['tn3k']['folds'])}")
print(f"  tg3k  train={len(S['tg3k']['train'])} val={len(S['tg3k']['val'])} "
      f"groups={len(S['tg3k']['groups'])}")
print(f"  ddti  test={len(S['ddti']['test'])}  path={img}")
