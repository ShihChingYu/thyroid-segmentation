#!/usr/bin/env python
"""Check 10 - near-duplicate scan across splits.

Three questions:
  Q1  Do any TN3K trainval images near-duplicate a TN3K test image?
      (the only available check on the authors' patient-level claim)
  Q2  Are consecutive TN3K indices more similar than random pairs?
      (if yes, patients are grouped in index order and the `index % 5`
       fold assignment splits patients across folds)
  Q3  Do TG3K frames near-duplicate TN3K test images?
      (whether the two datasets share patients - matters for W4)

Method: 63-bit perceptual hash (DCT) on the IMAGES, Hamming distance.
Run from the repo root."""
import glob, os, sys, csv
import numpy as np, cv2

THRESH = 5          # Hamming distance treated as a near-duplicate
OUT = '/workspace/amy/notes/10_duplicate_pairs.csv'


def phash(path, size=32, hs=8):
    a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if a is None:
        return None
    a = cv2.resize(a, (size, size)).astype(np.float32)
    d = cv2.dct(a)[:hs, :hs].flatten()[1:]     # drop the DC term
    return d > np.median(d)


def hashes(d):
    files = sorted(f for f in glob.glob(f'{d}/*')
                   if os.path.isfile(f) and '.ipynb_checkpoints' not in f)
    names, H = [], []
    for f in files:
        h = phash(f)
        if h is not None:
            names.append(os.path.splitext(os.path.basename(f))[0])
            H.append(h)
    return names, np.array(H)


def hamming(A, B):
    """Pairwise Hamming distance between two bool hash matrices."""
    a, b = A.astype(np.int16), B.astype(np.int16)
    return a @ (1 - b).T + (1 - a) @ b.T


print('hashing...', flush=True)
tv_n,  TV = hashes('data/tn3k/trainval-image')
te_n,  TE = hashes('data/tn3k/test-image')
tg_n,  TG = hashes('data/tg3k/thyroid-image')
print(f'  trainval {len(tv_n)}   test {len(te_n)}   tg3k {len(tg_n)}\n')

rows = []

# ---------- Q1: TN3K trainval vs TN3K test ----------
H = hamming(TV, TE)
mind = H.min(axis=0)
hits = np.argwhere(H <= THRESH)
print(f'Q1  TN3K trainval vs test   ({H.size:,} pairs)')
print(f'    min distance per test image: median={np.median(mind):.1f} '
      f'min={mind.min()} p5={np.percentile(mind,5):.1f}')
print(f'    pairs with distance <= {THRESH}: {len(hits)}')
order = np.argsort(H[hits[:,0], hits[:,1]])[:10] if len(hits) else []
for k in order:
    i, j = hits[k]
    print(f'      trainval/{tv_n[i]}  ~  test/{te_n[j]}   d={H[i,j]}')
    rows.append(['Q1', f'trainval/{tv_n[i]}', f'test/{te_n[j]}', int(H[i,j])])

# ---------- Q2: are neighbouring indices the same patient? ----------
adj = np.array([hamming(TV[i:i+1], TV[i+1:i+2])[0,0] for i in range(len(TV)-1)])
rng = np.random.default_rng(0)
ii = rng.integers(0, len(TV), 4000); jj = rng.integers(0, len(TV), 4000)
keep = ii != jj
rnd = np.array([hamming(TV[a:a+1], TV[b:b+1])[0,0]
                for a, b in zip(ii[keep][:2000], jj[keep][:2000])])
print(f'\nQ2  TN3K trainval, adjacent indices vs random pairs')
print(f'    adjacent (i, i+1): median={np.median(adj):.1f}  '
      f'frac <= {THRESH}: {np.mean(adj <= THRESH):.2%}')
print(f'    random pairs     : median={np.median(rnd):.1f}  '
      f'frac <= {THRESH}: {np.mean(rnd <= THRESH):.2%}')

# ---------- Q3: TG3K vs TN3K test ----------
H3 = hamming(TG, TE)
hits3 = np.argwhere(H3 <= THRESH)
print(f'\nQ3  TG3K vs TN3K test   ({H3.size:,} pairs)')
print(f'    min distance overall: {H3.min()}')
print(f'    pairs with distance <= {THRESH}: {len(hits3)}')
order3 = np.argsort(H3[hits3[:,0], hits3[:,1]])[:10] if len(hits3) else []
for k in order3:
    i, j = hits3[k]
    print(f'      tg3k/{tg_n[i]}  ~  test/{te_n[j]}   d={H3[i,j]}')
    rows.append(['Q3', f'tg3k/{tg_n[i]}', f'test/{te_n[j]}', int(H3[i,j])])

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['question','a','b','hamming']); w.writerows(rows)
print(f'\nwrote {OUT}  ({len(rows)} flagged pairs)')
