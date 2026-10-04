#!/bin/bash
# Score run-008 predictions with our own harness, all five folds,
# at BOTH scoring resolutions.
#
#   at=input    224x224, the grid the reference implementation uses.
#               This is the comparable number.
#   at=original the ground-truth image size. Dice is unchanged; HD95 is
#               ~2.3x larger, because distance is counted in pixels and
#               the pixels are smaller. Kept to document that difference.
#
# --pred must name ONE fold: results/ holds all five, same filenames in each.
cd /workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation

for at in input original; do
  for f in 0 1 2 3 4; do
    echo "===== fold $f  at=$at  $(date -u) ====="
    python /workspace/amy/harness/score.py \
      --pred results/test-TN3K/unet/fold$f \
      --gt   data/tn3k/test-mask \
      --split-key tn3k:test \
      --at   $at \
      --out  /workspace/amy/notes/06_score_run008_fold${f}_at-${at}.csv
  done
done
