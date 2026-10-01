#!/usr/bin/env bash
set -e

ENV=ours_idjc320
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG="configs/mask2former/my_mask2former_swin-b_LGFE_CDCC-20k_IDJC320-512x512.py"
CKPT="checkpoint/best_mIoU_iter_3000.pth"

activate_env () {
    eval "$(conda shell.bash hook)"
    conda activate "$ENV"
    export PYTHONPATH="$ROOT:$ROOT/mmdetection:${PYTHONPATH:-}"
    cd "$ROOT"
}

case "${1:-}" in

setup)
    echo "===== Creating reproduction environment ====="

    if ! command -v conda >/dev/null 2>&1; then
        echo "[ERROR] Conda was not found. Please use a Linux GPU environment with Conda/Miniconda installed."
        exit 1
    fi

    eval "$(conda shell.bash hook)"

    if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
        conda create -n "$ENV" python=3.10 -y
    fi

    conda activate "$ENV"

    python -m pip install --upgrade pip setuptools wheel

    echo "===== Installing PyTorch 2.1.2 + CUDA 12.1 ====="
    pip install \
      torch==2.1.2 \
      torchvision==0.16.2 \
      torchaudio==2.1.2 \
      --index-url https://download.pytorch.org/whl/cu121

    echo "===== Installing pinned dependencies ====="
    pip install numpy==1.26.4
    pip install openmim==0.3.9
    pip install mmengine==0.10.7

    echo "===== Installing MMCV 2.1.0 ====="
    mim install "mmcv==2.1.0"

    echo "===== Installing MMSegmentation dependency ====="
    pip install "mmsegmentation==1.2.2"

    echo "===== Pinning NumPy / OpenCV and installing additional project dependencies ====="
    pip install --force-reinstall "numpy==1.26.4" "opencv-python==4.10.0.84"
    pip install ftfy regex

    echo "===== Installing the bundled MMDetection package ====="
    pip install -v --no-build-isolation "$ROOT/mmdetection"

    echo
    echo "Environment setup completed. Next step:"
    echo "bash reproduce_ubuntu_linux.sh check"
    ;;

check)
    activate_env

    echo "===== Automatic environment and project checks ====="

    python - <<'PY'
from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mmdetection"))

import torch
import numpy
import mmcv
import mmengine
import mmseg
import mmdet

print("[PASS] Python       :", sys.version.split()[0])
print("[PASS] PyTorch      :", torch.__version__)
print("[PASS] CUDA Runtime :", torch.version.cuda)
print("[PASS] NumPy        :", numpy.__version__)
print("[PASS] MMCV         :", mmcv.__version__)
print("[PASS] MMEngine     :", mmengine.__version__)
print("[PASS] MMSeg        :", mmseg.__version__)
print("[PASS] MMDet        :", mmdet.__version__)

assert torch.cuda.is_available(), "CUDA/GPU is unavailable"
print("[PASS] GPU          :", torch.cuda.get_device_name(0))

from mmcv.ops import MultiScaleDeformableAttention
print("[PASS] MMCV CUDA Ops")

from mmseg.models.necks.lgfe_neck import LGFENeck
print("[PASS] LGFE")

from mmseg.models.decode_heads.mask2former_head_cdcc import Mask2FormerHeadCDCC
print("[PASS] CDCC")

from mmengine.config import Config
cfg = Config.fromfile(
    "configs/mask2former/"
    "my_mask2former_swin-b_LGFE_CDCC-20k_IDJC320-512x512.py"
)

assert cfg.model.neck.type == "LGFENeck"
assert cfg.model.decode_head.type == "Mask2FormerHeadCDCC"

for p in [
    "data/my_IDJC320_voc/JPEGImages",
    "data/my_IDJC320_voc/SegmentationClass",
    "data/my_IDJC320_voc/train.txt",
    "data/my_IDJC320_voc/val.txt",
    "data/my_IDJC320_voc/test.txt",
    "checkpoint/best_mIoU_iter_3000.pth",
    "pretrained/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth",
]:
    assert Path(p).exists(), f"Missing required file: {p}"

print("[PASS] Dataset, weights, and configuration")
print()
print("========== All checks passed ==========")
PY

    echo
    echo "Next step: bash reproduce_ubuntu_linux.sh test"
    ;;

test)
    activate_env

    echo "===== Evaluating the final Ours model ====="
    echo "Reference target: mIoU=73.92  mAcc=82.39  aAcc=91.27"

    python tools/test.py \
      "$CONFIG" \
      "$CKPT" \
      --work-dir work_dirs/final_test \
      --cfg-options test_dataloader.dataset.ann_file=val.txt
    ;;

smoke)
    activate_env

    echo "===== Training-path smoke test: 20 iterations ====="
    rm -rf work_dirs/smoke_test

    python tools/train.py \
      "$CONFIG" \
      --work-dir work_dirs/smoke_test \
      --cfg-options \
      train_cfg.max_iters=20 \
      train_cfg.val_interval=20 \
      default_hooks.checkpoint.interval=20
    ;;

train)
    activate_env

    echo "===== Full Ours retraining: 20,000 iterations ====="

    python tools/train.py \
      "$CONFIG" \
      --work-dir work_dirs/ours_retrain_full
    ;;

*)
    echo "Usage:"
    echo
    echo "  bash reproduce_ubuntu_linux.sh setup   # Step 1: create the environment"
    echo "  bash reproduce_ubuntu_linux.sh check   # Step 2: check the environment and project"
    echo "  bash reproduce_ubuntu_linux.sh test    # Step 3: reproduce the 73.92% mIoU validation result"
    echo "  bash reproduce_ubuntu_linux.sh smoke   # Step 4: run a 20-iteration training smoke test"
    echo "  bash reproduce_ubuntu_linux.sh train   # Full 20,000-iteration retraining"
    exit 1
    ;;
esac
