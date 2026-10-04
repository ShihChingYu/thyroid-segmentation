cat > /workspace/amy/CHECKS.md << 'MDEOF'
# Checks and results log

## Status — Sprint 1 Week 3

| # | Task | Status |
|---|---|---|
| 1 | Split generator, commit `splits/v1.json` | done |
| 2 | `predict.py` | deferred to W4 — `eval.py` covers TN3K; needed for DDTI and Sound Blade data |
| 3 | `score.py` | done |
| 4 | `report.py` | cut — premature tooling; CSVs plus this file serve the purpose |
| 5 | Dice and HD95 per case | done |
| 6 | Dice and HD95 per class | deferred to W4 — `--class-name` column exists; needs a multi-task model |
| 7 | Empty-mask convention | done — `--empty {exclude,one,zero}`, default `exclude`, validated in check 09 |
| 8 | Validate with ground-truth and all-zero dummy models | done — check 09 |
| 9 | Re-score week 2's model | done — check 06 |
| 10 | Automated duplicate scan across splits | **outstanding** |
| 11 | Configure the N-class head | deferred to W4 — prerequisite for multi-task, not a W3 deliverable |

---

## 01 — Dataset inventory
**Question:** Are the three datasets complete, are any masks empty, and what
size are the images?

**Command:** `python checks/01_dataset_inventory.py | tee notes/01_*.log`

**Result:**

| dataset | images | masks | empty | height | width |
|---|---|---|---|---|---|
| TN3K trainval | 2,879 | 2,879 | 0 | 174–826 | 184–1463 |
| TN3K test | 614 | 614 | 0 | 261–772 | 255–1399 |
| TG3K | 3,585 | 3,585 | 0 | 174–238 | 214–325 |
| DDTI | — | — | — | not found |

Image and mask counts match exactly in all three. TN3K totals
3,493, matching Gong et al. Test dimensions fall inside the trainval range.

**Implication:** No empty masks, so the Dice 0/0 case never arises on these
datasets. 

---

## 02 — Mask values and resolution structure
**Question:** Is the ground truth cleanly binary, and what does the
resolution distribution reveal about acquisition?

**Command:** `python checks/02_mask_values_and_shapes.py | tee notes/02_*.log`

**Result:**

| dataset | n | distinct resolutions | masks with midtones (50–200) |
|---|---|---|---|
| TN3K trainval | 2,879 | 2,443 | 0 (0%) |
| TN3K test | 614 | 555 | 0 (0%) |
| TG3K | 3,585 | **15** | 3,341 (93%) |

All values fall at 0–4 and 251–255 — JPEG ringing around a binary mask, not
genuine grey levels.

TN3K is effectively binary: no pixel between 50 and 200 in any mask, so any
threshold from 5 to 250 gives an identical result. Nearly every image has its
own resolution, but widths recur across otherwise-unique sizes — all five
most-common trainval sizes share width 411; test shows 417, 466 and 400 —
suggesting a small number of acquisition devices, consistent with the three
the authors report.

TG3K has only 15 distinct resolutions for 3,585 files, with group sizes 329,
316, 315, 272, 261… — the signature of frames drawn from a small number of
videos. 93% of its masks carry genuinely soft boundaries.

**Implication:** Binarization threshold fixed at **>127** and recorded as an
explicit harness parameter. TN3K is insensitive to the choice; TG3K is mildly
sensitive and the value must be stated whenever gland numbers are reported.

---

## 03 — TG3K video grouping and shipped-split integrity
**Question:** Do TG3K's frames group into videos, and does the shipped
train/val split respect those groups?

**Command:** `python checks/03_tg3k_video_groups.py | tee notes/03_*.log`

**Result:** 3,585 frames form 15 contiguous runs of constant resolution.
The paper reports 16 source videos; two adjacent videos likely share a
resolution, which is conservative for splitting since merging two videos can
never split one.

The shipped `tg3k-trainval.json` cuts at index 3226. Videos 1–13 fall entirely
in train and video 15 entirely in val, but **video 14 (indices 3026–3269)
straddles the cut: 200 frames in train, 44 in val.** That is 44/359 = **12.3%**
of the gland validation set drawn from a video that also contributes to
training.

The pattern — thirteen clean videos either side, one straddle — suggests an
intended 90/10 tail cut whose boundary was never checked against video
boundaries.

**Not measured:** whether those 44 frames are near-duplicates in pixel terms.
Same-video membership is established; visual similarity is not.

**Implication:** Shipped split not used. `splits/v1.json` holds out videos
14–15 whole. The original is retained so the two can be compared, and so the
cost of the leak can be measured in W4.

---

## 04 — TN3K fold integrity
**Question:** Do the five official fold files partition trainval cleanly, and
was the assignment grouped or arbitrary?

**Command:** `python checks/04_tn3k_fold_integrity.py | tee notes/04_*.log`

**Result:** Pass. Five fold files, each a dict of `train`/`val` index lists.
`train ∩ val = 0` in all five; each fold totals 2,879; the five validation
sets union to exactly 2,879 with no image validated twice. Sizes 2,303/576
match Gong et al.; fold 4 is 2,304/575, the remainder.

**Note:** fold 0's val list begins [0, 5, 10, 15, 20] — assignment is
`index % 5`, not grouped. Patient-safe only if the index order is already
patient-randomized, which is untested and added to the duplicate scan.

**Implication:** Folds adopted unchanged in `splits/v1.json`. Any fold-level
leakage would affect checkpoint selection only, not the reported test metric,
which uses the separate, patient-disjoint test directory.

---

## 05 — Frozen split definition

**Command:** `python make_splits.py | tee notes/05_make_splits.log`
**Output:** `splits/v1.json`

- **TN3K** — official split and five folds copied verbatim; trainval 2,879,
  test 614. Adopted rather than rebuilt because patient identifiers are
  removed from the release ("the personal identities of all images have been
  removed and cannot be reconstructed"), so any split built here would group
  by image rather than patient and would break comparability with published
  results.
- **TG3K** — regenerated. Validation = videos 14–15 whole (indices 3026–3584,
  559 frames); training = videos 1–13 (3,026 frames). No video appears on
  both sides. See check 03.
- **DDTI** — evaluation only; no split required. Path unresolved, see check 01.


**Policy:** frozen. Every script reads splits from this file and never lists a
directory. Changes become `v2` with a recorded reason.

**Known limitation:** TG3K validation has an effective sample size of **2**
(two videos, two patients), not 559. Usable for checkpoint selection; not
reportable as a gland measurement. A defensible gland result would need
grouped k-fold across all 15 videos — not done.

**Outstanding:** `dataloaders/tg3k.py` hardcodes `tg3k-trainval.json` and must
be pointed at `v1.json` before the first W4 multi-task run, or the leaky split
loads silently.

---

## 06 — Re-scoring run-008 with our own harness
**Question:** What does week 2's model score when measured by code we control?

**Command:** `bash harness/run_score_run008.sh | tee notes/06_score_run008_all.log`
**Date:** 2026-10-04
**Output:** 10 CSVs (5 folds × 2 scoring resolutions) with `.meta` files.

**Result** — five folds, threshold 127, empties excluded:

| Metric | `score.py` @224 | `score.py` @original | `eval.py` | Published |
|---|---|---|---|---|
| Dice, per-case mean | 0.7812 ± 0.0068 | 0.7811 ± 0.0068 | 0.7627 ± 0.0070 | 0.7643 ± 0.0067 *(F1 col)* |
| Dice, per-case median | 0.8661 | 0.8667 | — | — |
| Dice, aggregate (pooled) | 0.8449 | 0.8471 | — | — |
| Dice, derived 2·IoU/(1+IoU) | 0.8119 | — | — | 0.7951 ± 0.0131 *(Dice col)* |
| IoU, per-case mean | 0.6834 ± 0.0066 | 0.6832 ± 0.0065 | 0.6571 ± 0.0065 | 0.6599 ± 0.0066 |
| IoU, aggregate | 0.7314 | 0.7334 | — | — |
| HD95, standard (boundary) | 25.53 ± 1.66 | 57.95 | — | — |
| HD, repo definition | 18.54 ± 1.30 | — | 19.98 ± 1.22 | 18.44 ± 0.75 |

± is the standard deviation across the five folds, not across cases.

**Four findings:**

1. **Dice and IoU are resolution-invariant, measured.** 0.7812 at 224 against
   0.7811 at original — a difference of 0.0001 across 3,070 image pairs.
2. **HD95 is not.** 2.27× between the two grids on identical predictions.
   Reported at 224×224 to match the reference implementation. TN3K ships no
   pixel spacing (JPEGs), so millimetre reporting is unavailable; Sound Blade
   DICOM data carries real spacing and should be reported in mm.
3. **Per-case Dice is bimodal.** Median 0.866, IQR [0.73, 0.93], mean 0.781.
   The mean is dragged down by a tail including 6–13 completely empty
   predictions per fold. Report the median alongside the mean.
4. **Aggregate Dice (0.845) far exceeds the per-case mean (0.781)** — the
   signature of small nodules failing while large ones dominate the pooled
   total. The paper's 0.7951 is not a pooled figure; it equals
   2 × 0.6599 / 1.6599 exactly, i.e. it is derived from their IoU column.

**Also:** 49.3% of fold-0 predictions contain more than one connected
component (median 1, max 27). Worst HD95 cases split into two groups — good
Dice ruined by distant specks (0425: Dice 0.762, HD95 471, 11 components),
and total misses (0279: Dice 0.000). A relative-size component filter would
address the first group and correctly leave the second alone.

---

## 08 — HD95 definition: ours vs the reference implementation
**Question:** Why is our HD95 (25.53) higher than `eval.py`'s (19.98) on
identical prediction files?

**Command:** `for f in 0 1 2 3 4; do python checks/08_hd_definition.py $f; done | tee notes/08_hd_definition_all.log`

**Result:** The repo's `visualization/metrics.py` does not compute Hausdorff
distance:

```python
indexes = np.nonzero(x)              # ALL foreground pixels, not the boundary
distances = edt(np.logical_not(y))   # distance to nearest FOREGROUND of y
return np.percentile(distances[indexes], 95)
```

Hausdorff distance is defined over surface points; this includes the entire
interior, and every pixel inside the overlap contributes zero. Running their
code on our masks at 224×224, five folds:

| | mean | median |
|---|---|---|
| Ours (boundary pixels) | 25.53 ± 1.66 | 13.51 |
| Repo definition | 18.54 ± 1.30 | 6.17 |

Ratio is stable across folds: 1.366, 1.419, 1.352, 1.382, 1.369.
Their value is ≤ ours in 98.7–98.8% of cases.

The two **agree** on total failures (fold 0, case 0354: 147.8 vs 144.3),
because dilution only helps where there is overlap.

Empty predictions explain the residual: `eval.py` scores them by planting a
pixel at `pred[0][0][0][0]` rather than excluding them, worth about 127 px per
case and accounting for 18.54 → 19.98. The published 18.44 matches our 18.54
to 0.10, so the published figure evidently excludes failures too.

**Implication:** HD figures published from this codebase are not comparable to
standard HD95 implementations and understate ordinary boundary error by
roughly 2× at the median. We keep the standard boundary-based definition and
record the translation. Both metrics reproduce the published result once
definitions are aligned.

---

## 09 — Scorer validation against dummy models
**Question:** Does `score.py` return the correct value when the answer is
known in advance?

**Command:** `python checks/09_dummy_validation.py | tee notes/09_*.log`

**Result:** 614 TN3K test masks at 224×224.

- **A — prediction == ground truth:** dice min 1.000000, iou min 1.000000,
  hd95 max 0.000000; **0 cases** deviating from exactly 1.0.
- **B — all-zero prediction:** dice max 0.000000; hd95 undefined in all 614.

**Implication:** The scorer is correct at both ends of the range, through the
full path — read, binarize at 127, resize to 224 nearest-neighbour, match,
split-filter, compute. The ~1.8-point Dice difference against `eval.py` on
identical prediction files is therefore attributable to the reference
implementation, not to this harness. Test A also exercises `surface()` on all
614 real shapes, so the high HD95 on fold 3 case 0245 is not a
boundary-extraction fault.

---

## Open items

| Item | Why it matters |
|---|---|
| Per-case Dice runs +1.8 points above `eval.py` on identical masks | Our side is validated (check 09); the cause is in `eval.py`'s data loading. Candidates: ground-truth resize interpolation, batch-level aggregation. A curiosity, not a blocker. |
| DDTI not located | Blocks cross-dataset evaluation, which is the preview of device shift. |
| Duplicate scan not run | The only available check on TN3K's patient-level claim, on whether `index % 5` folds are patient-safe, and on TN3K ↔ TG3K patient overlap. |
| `dataloaders/tg3k.py` hardcodes the leaky split | Silently loads the wrong split in W4 unless changed. |
| Fold 3 case 0245: HD95 136 at Dice 0.94 | Likely a small component buried inside the gland — correct Hausdorff behaviour, and an argument for filtering before measuring. Worth one look. |
MDEOF