## 01 — Dataset inventory
**Question:** Are all three datasets complete, are any masks empty, and what
size are the images?

**Command:** `python checks/01_dataset_inventory.py | tee notes/01_*.log`
**Date:** 2026-10-01

**Result:**

| dataset | images | masks | empty | height | width |
|---|---|---|---|---|---|
| TN3K trainval | 2,879 | 2,879 | 0 | 174–826 | 184–1463 |
| TN3K test | 614 | 614 | 0 | 261–772 | 255–1399 |
| TG3K | 3,585 | 3,585 | 0 | 174–238 | 214–325 |
| DDTI | — | — | — | **not found** |

Image and mask counts match exactly in all three — no orphans. TN3K totals
3,493, matching Gong et al. Test dimensions sit inside the trainval range,
so the test set is not a different acquisition regime.

**Implication:** No empty masks, so the Dice 0/0 case never arises here. The
convention must still be fixed for the patient data and is tested
synthetically in check 07.

**Outstanding:** DDTI not found at `data/DDTI` or `data/ddti` with any of the
expected subdirectory names. Must be resolved before any cross-dataset
evaluation — `dataloaders/ddti.py` expects `./data/DDTI/{image,mask}`.

## 02 — Mask values and resolution structure
**Question:** Is the ground truth cleanly binary, and what does the
resolution distribution reveal about how each dataset was acquired?

**Command:** `python checks/02_mask_values_and_shapes.py | tee notes/02_*.log`
**Date:** 2026-10-01

**Result:**

| dataset | n | distinct resolutions | midtones (50–200) |
|---|---|---|---|
| TN3K trainval | 2,879 | 2,443 | 0 (0%) |
| TN3K test | 614 | 555 | 0 (0%) |
| TG3K | 3,585 | **15** | 3,341 (93%) |

All three show values only at 0–4 and 251–255 — JPEG ringing around a binary
mask, not genuine grey levels.

**TN3K** is effectively binary: no pixel between 50 and 200 in any mask, so
any threshold from 5 to 250 gives an identical result. Nearly every image has
its own resolution, but widths recur across otherwise-unique sizes — all five
most-common trainval sizes share width 411; test shows 417, 466 and 400 —
suggesting a small number of acquisition devices, consistent with the three
the authors report.

**TG3K** has only 15 distinct resolutions for 3,585 files, with group sizes of
329, 316, 315, 272, 261… — the signature of frames drawn from a small number
of videos (16 per the paper). This is what makes video grouping recoverable;
see check 03. 93% of its masks carry genuinely soft boundaries, so the
threshold choice slightly alters the gland ground truth.

**Implication:** Binarization threshold fixed at **>127** and recorded as an
explicit harness parameter rather than a constant inside a function. TN3K is
insensitive to the choice; TG3K is mildly sensitive and the value must be
stated whenever gland numbers are reported.

## 03 — TG3K video grouping and shipped-split integrity
**Question:** Do TG3K's frames group into videos, and does the shipped
train/val split respect those groups?

**Command:** `python checks/03_tg3k_video_groups.py | tee notes/03_*.log`
**Date:** 2026-10-01

**Result:** 3,585 frames form 15 contiguous runs of constant resolution
(paper reports 16 source videos; two adjacent videos likely share a
resolution, which is conservative for splitting). Videos 1-13 fall entirely
in train and video 15 entirely in val, but video 14 (3026-3269) straddles:
200 frames in train, 44 in val. 44/359 = 12.3% of the gland validation set
comes from a video that also contributes to training.

**Not yet measured:** whether those frames are near-duplicates in pixel
terms. Same-video membership is established; visual similarity is not.

**Implication:** Shipped split not used. splits/v1.json holds out videos
14-15 whole. Original retained so the two can be compared.

## 04 — TN3K fold integrity
**Question:** Do the five official fold files partition trainval cleanly,
and was the assignment grouped or arbitrary?

**Command:** `python checks/04_tn3k_fold_integrity.py | tee notes/04_*.log`
**Date:** 2026-10-01

**Result:** Pass. Five fold files, each a dict of train/val index lists.
train ∩ val = 0 in all five; each fold totals 2,879; the five validation
sets union to exactly 2,879 with no image validated twice. Sizes 2,303/576
match Gong et al.; fold 4 is 2,304/575, the remainder.

**Note:** fold 0's val list begins [0, 5, 10, 15, 20] — assignment is
index % 5, not grouped. Patient-safe only if the index order is already
patient-randomized. Untested; added to the duplicate scan (check 06) as an
adjacent-pair similarity test.

**Implication:** Folds adopted unchanged in splits/v1.json. Any fold-level
leakage would affect checkpoint selection only, not the reported test
metric, which uses the separate patient-disjoint test directory.

## 05 — Frozen split definition
**Command:** `python make_splits.py | tee notes/05_make_splits.log`
**Date:** 2026-10-01
**Output:** `splits/v1.json`

**Contents:**
- TN3K — official split and five folds copied verbatim; trainval 2,879,
  test 614. Adopted rather than rebuilt because patient IDs are removed
  from the release (see check 04).
- TG3K — regenerated. Validation = videos 14-15 whole (indices 3026-3584,
  559 frames); training = videos 1-13 (3,026 frames). No video appears on
  both sides. The shipped split is retained at data/tg3k/tg3k-trainval.json
  so the two can be compared (see check 03).
- DDTI — evaluation only; no split required.

Each dataset carries a SHA-256 fingerprint of its file list so data drift
is detectable later.

**Policy:** frozen. From here every script reads splits from this file and
never lists a directory. Changes become v2 with a recorded reason.

**Known limitation:** TG3K validation has an effective sample size of 2
(two videos, two patients), not 559. Usable for checkpoint selection; not
reportable as a gland measurement. A defensible gland result would need
grouped k-fold across all 15 videos — not done.

**Outstanding:** dataloaders/tg3k.py hardcodes tg3k-trainval.json and must
be pointed at v1.json before the first W4 multi-task run.

## 09 — Scorer validation against dummy models
**Question:** Does score.py return the correct value when the answer is
known in advance?

**Command:** `python checks/09_dummy_validation.py | tee notes/09_*.log`
**Date:** 2026-10-04

**Result:** 614 TN3K test masks at 224x224.
  A - prediction == ground truth: dice min 1.000000, iou min 1.000000,
      hd95 max 0.000000; 0 cases deviating from exactly 1.0.
  B - all-zero prediction: dice max 0.000000; hd95 undefined in all 614.

**Implication:** The scorer is correct at both ends of the range, through
the full path (read, binarize at 127, resize to 224 nearest-neighbour,
match, filter, compute).
