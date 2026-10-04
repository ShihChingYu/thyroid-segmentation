#!/bin/bash
# run-008: U-Net on TN3K, 5-fold, TRFE+ paper recipe
# lr 1e-2 (paper baselr), 50 epochs, bs16, 224px, Dice, SGD+poly decay
cd /workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation
mkdir -p /workspace/amy/runs

for f in 0 1 2 3 4; do
  echo "===== fold $f  $(date) =====" | tee -a /workspace/amy/runs/run-008-eval.log

  python train.py -model_name unet -dataset TN3K -fold $f -gpu 0 \
    -lr 1e-2 -batch_size 16 -nepochs 50 -input_size 224 -criterion Dice \
    2>&1 | tee /workspace/amy/runs/run-008-unet-fold$f.log

  cp run/unet/fold$f/unet_best.pth /workspace/amy/runs/run-008-unet-fold$f.pth

  python eval.py -model_name unet \
    -load_path /workspace/amy/runs/run-008-unet-fold$f.pth \
    -test_dataset TN3K -test_fold test -fold $f -save_dir ./results -gpu 0 \
    2>&1 | tee -a /workspace/amy/runs/run-008-eval.log
done
echo "===== done $(date) =====" | tee -a /workspace/amy/runs/run-008-eval.log
