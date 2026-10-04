#!/usr/bin/env python
"""Score predicted masks against ground truth.

Every decision that affects a number is an explicit argument, not a
constant buried in a function.

--pred must name ONE fold's directory. Files prefixed 's' are probability
maps, not binary predictions, and are ignored automatically.

  python score.py --pred results/test-TN3K/unet/fold0 \
                  --gt data/tn3k/test-mask \
                  --split-key tn3k:test --at input --out scores.csv
"""
import argparse, csv, glob, json, os, sys, subprocess, datetime
import numpy as np
import cv2
from scipy import ndimage


# ---------- metrics ----------

def dice(p, g):
    s = p.sum() + g.sum()
    return None if s == 0 else 2.0 * np.logical_and(p, g).sum() / s


def iou(p, g):
    u = np.logical_or(p, g).sum()
    return None if u == 0 else np.logical_and(p, g).sum() / u


def surface(m):
    """Boundary pixels: foreground pixels with a background neighbour."""
    er = ndimage.binary_erosion(m, border_value=0)
    return np.argwhere(m & ~er)


def _mask_from(pts, shape):
    m = np.zeros(shape, dtype=bool)
    m[pts[:, 0], pts[:, 1]] = True
    return m


def hd95(p, g):
    """95th-percentile symmetric Hausdorff distance, in pixels.

    NOTE: pools both directions into one percentile. medpy uses
    max(p95(p->g), p95(g->p)), which is >= this value."""
    if p.sum() == 0 or g.sum() == 0:
        return None
    sp, sg = surface(p), surface(g)
    if len(sp) == 0 or len(sg) == 0:
        return None
    dg = ndimage.distance_transform_edt(~_mask_from(sg, g.shape))
    dp = ndimage.distance_transform_edt(~_mask_from(sp, p.shape))
    d_p_to_g = dg[sp[:, 0], sp[:, 1]]
    d_g_to_p = dp[sg[:, 0], sg[:, 1]]
    return float(np.percentile(np.concatenate([d_p_to_g, d_g_to_p]), 95))


# ---------- io ----------

def load(path, threshold):
    a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise IOError(f'cannot read {path}')
    return a > threshold


def match(pred_dir, gt_dir):
    """Pair prediction files with ground-truth files by filename stem.

    Refuses to continue if two predictions share a stem, which happens when
    --pred points at a directory holding several folds' outputs."""
    gt = {os.path.splitext(os.path.basename(f))[0]: f
          for f in glob.glob(f'{gt_dir}/*')
          if os.path.isfile(f) and not f.endswith('.json')}

    files = [f for f in sorted(glob.glob(f'{pred_dir}/**/*.png', recursive=True) +
                               glob.glob(f'{pred_dir}/**/*.jpg', recursive=True))
             if '.ipynb_checkpoints' not in f]

    pairs, seen = [], {}
    for pf in files:
        stem = os.path.splitext(os.path.basename(pf))[0]
        if stem in seen:
            sys.exit(f'duplicate prediction for "{stem}":\n'
                     f'  {seen[stem]}\n  {pf}\n'
                     f'point --pred at a single fold\'s directory')
        seen[stem] = pf
        if stem in gt:
            pairs.append((stem, pf, gt[stem]))

    unmatched = len(files) - len(pairs)
    if unmatched:
        print(f'note: {unmatched} prediction(s) had no ground truth in {gt_dir}')
    return pairs


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='directory of predicted masks')
    ap.add_argument('--gt',   required=True, help='directory of ground-truth masks')
    ap.add_argument('--out',  required=True, help='per-case CSV to write')
    ap.add_argument('--threshold', type=int, default=127,
                    help='binarization threshold (default 127)')
    ap.add_argument('--at', choices=['original', 'input'], default='original',
                    help='resolution to score at: ground-truth size, or model '
                         'input size (--input-size)')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--empty', choices=['exclude', 'one', 'zero'], default='exclude',
                    help='what Dice means when BOTH masks are empty')
    ap.add_argument('--split', default='/workspace/amy/splits/v1.json',
                    help='frozen split file (default: splits/v1.json)')
    ap.add_argument('--split-key', required=True,
                    help='which list to score against, e.g. tn3k:test')
    ap.add_argument('--class-name', default='nodule',
                    help='structure being scored; one row set per class')
    args = ap.parse_args()

    pairs = match(args.pred, args.gt)
    if not pairs:
        sys.exit(f'no prediction/ground-truth pairs found under {args.pred}')

    if args.split:
        ds, key = args.split_key.split(':')
        allow = {f'{i:04d}' for i in json.load(open(args.split))[ds][key]}
        before = len(pairs)
        pairs = [p for p in pairs if p[0] in allow]
        print(f'split filter {args.split_key}: {before} -> {len(pairs)} cases')
        if not pairs:
            sys.exit('the split filter removed every case.\n'
                     '  wrong --split-key, or --gt points at the wrong folder')

    rows = []
    for stem, pf, gf in pairs:
        p, g = load(pf, args.threshold), load(gf, args.threshold)

        if args.at == 'original':
            if p.shape != g.shape:
                p = cv2.resize(p.astype(np.uint8), (g.shape[1], g.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            sz = (args.input_size, args.input_size)
            p = cv2.resize(p.astype(np.uint8), sz,
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            g = cv2.resize(g.astype(np.uint8), sz,
                           interpolation=cv2.INTER_NEAREST).astype(bool)

        d, i, h = dice(p, g), iou(p, g), hd95(p, g)
        if d is None:                      # both masks empty
            if args.empty == 'one':
                d, i = 1.0, 1.0
            elif args.empty == 'zero':
                d, i = 0.0, 0.0
            # 'exclude' leaves them None -> dropped from the summary

        rows.append({'name': stem, 'class': args.class_name,
                     'dice': d, 'iou': i, 'hd95': h,
                     'gt_px': int(g.sum()), 'pred_px': int(p.sum()),
                     'inter_px': int(np.logical_and(p, g).sum()),
                     'pred_components': int(ndimage.label(p)[1])})

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f'\nscored {len(rows)} cases  class={args.class_name}')
    print(f'  threshold={args.threshold}  at={args.at}  empty={args.empty}')
    print(f'  pred={args.pred}\n  gt={args.gt}\n')

    for k in ('dice', 'iou', 'hd95'):
        v = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
        skipped = len(rows) - len(v)
        if len(v) == 0:
            print(f'  {k:<5} n=0      all undefined')
            continue
        sd = v.std(ddof=1) if len(v) > 1 else float('nan')
        print(f'  {k:<5} n={len(v):<5} mean={v.mean():.4f}  sd_across_cases={sd:.4f}'
              f'  median={np.median(v):.4f}'
              f'  iqr=[{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}]'
              + (f'   ({skipped} undefined)' if skipped else ''))

    I  = sum(r['inter_px'] for r in rows)
    PG = sum(r['pred_px'] + r['gt_px'] for r in rows)
    U  = sum(r['pred_px'] + r['gt_px'] - r['inter_px'] for r in rows)
    if PG:
        print(f'  aggregate  dice={2 * I / PG:.4f}  iou={I / U:.4f}'
              f'   (pooled over all pixels, not a per-case mean)')

    print(f'\nwrote {args.out}')

    # ---- provenance: every output carries the command that made it ----
    meta = args.out + '.meta'
    try:
        commit = subprocess.check_output(
            ['git', '-C', '/workspace/amy', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = 'no-git'
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(meta, 'w') as f:
        f.write(f'date    : {stamp}\n')
        f.write(f'cwd     : {os.getcwd()}\n')
        f.write(f'commit  : {commit}\n')
        f.write(f'command : python {" ".join(sys.argv)}\n')
        f.write(f'cases   : {len(rows)}\n')
        for key, val in sorted(vars(args).items()):
            f.write(f'  {key:<12}= {val}\n')
    print(f'wrote {meta}')


if __name__ == '__main__':
    main()
