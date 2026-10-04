#!/usr/bin/env python
"""Do TN3K's five official fold files partition trainval cleanly?
Run from the repo root."""
import json, glob

folds = {int(f[-6]): json.load(open(f))
         for f in sorted(glob.glob('data/tn3k/tn3k-trainval-fold*.json'))}
print(f'{len(folds)} fold files, keys = {list(next(iter(folds.values())).keys())}\n')

allval = []
for k, d in sorted(folds.items()):
    tr, va = set(d['train']), set(d['val'])
    print(f'fold {k}: train={len(tr)} val={len(va)} total={len(tr | va)} '
          f'train_int_val={len(tr & va)}')
    allval += list(va)

print(f'\nunion of val sets : {len(set(allval))}  (expect 2879)')
print(f'images in 2 folds : {len(allval) - len(set(allval))}  (expect 0)')
print(f'fold 0 val head   : {sorted(folds[0]["val"])[:6]}  '
      f'(stride 5 => assignment is index %% 5, not grouped)')
