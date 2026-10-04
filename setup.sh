#!/bin/bash
# Environment setup for a fresh pod.
# MUST be sourced, not executed:   source /workspace/amy/setup.sh
# Packages live on the network volume, so they survive pod redeploys.

VENV=/workspace/amy/venv
REPO=/workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation
PKGS="opencv-python-headless scipy scikit-image scikit-learn matplotlib \
      tqdm pillow gdown tensorboard tensorboardX ml_collections einops timm medpy"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: source this script, do not run it:"
  echo "    source ${BASH_SOURCE[0]}"
  exit 1
fi

if [ ! -d "$VENV" ]; then
  echo "creating venv at $VENV (first time on this volume)"
  python -m venv "$VENV" --system-site-packages
fi

source "$VENV/bin/activate"

python -c "import cv2, sklearn, ml_collections, medpy" 2>/dev/null || {
  echo "installing packages into $VENV ..."
  python -m pip install -q $PKGS
}

python - <<'PY'
import torch, cv2, scipy, sklearn, ml_collections, medpy
print(f'env    : torch {torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'         cv2 {cv2.__version__}  scipy {scipy.__version__}')
PY

echo "python : $(which python)"
echo "data   : $(ls $REPO/data/tn3k/test-image 2>/dev/null | wc -l) TN3K test images"
echo "splits : $([ -f /workspace/amy/splits/v1.json ] && echo 'v1.json present' || echo 'MISSING - run make_splits.py')"
