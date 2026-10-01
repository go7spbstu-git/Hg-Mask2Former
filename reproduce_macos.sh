#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-check}"
ENV_PATH="${2:-$HOME/ours_idjc320_mac_env}"

ROOT="$(cd "$(dirname "$0")" && pwd)"

CONFIG="$ROOT/configs/mask2former/my_mask2former_swin-b_LGFE_CDCC-20k_IDJC320-512x512.py"
CKPT="$ROOT/checkpoint/best_mIoU_iter_3000.pth"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$ROOT/mmdetection:${PYTHONPATH:-}"

PY="$ENV_PATH/bin/python"

require_macos() {
    if [ "$(uname -s)" != "Darwin" ]; then
        echo "ERROR: This script can only run on macOS."
        exit 1
    fi
}

require_env() {
    if [ ! -x "$PY" ]; then
        echo "ERROR: Environment not found: $ENV_PATH"
        echo "Please run first: bash reproduce_macos.sh setup"
        exit 1
    fi
}

require_package_files() {
    local failed=0

    for p in \
        "$CONFIG" \
        "$CKPT" \
        "$ROOT/mmseg" \
        "$ROOT/mmdetection"
    do
        if [ ! -e "$p" ]; then
            echo "ERROR: Missing required reproduction-package file: $p"
            failed=1
        fi
    done

    if [ "$failed" -ne 0 ]; then
        exit 1
    fi
}

check_native_arch() {
    local host_arch
    local py_arch

    host_arch="$(uname -m)"
    py_arch="$("$PY" -c 'import platform; print(platform.machine())')"

    echo "[INFO] Host architecture   : $host_arch"
    echo "[INFO] Python architecture : $py_arch"

    if [ "$host_arch" != "$py_arch" ]; then
        echo "ERROR: The Python/Conda architecture does not match the native macOS architecture."
        echo "On Apple Silicon, use a native arm64 Miniconda/Conda environment. Rosetta/x86_64 reproduction is not recommended."
        exit 1
    fi
}

case "$ACTION" in

setup)
    require_macos
    require_package_files

    echo "=================================================="
    echo " Ours IDJC320 - macOS Environment Setup"
    echo "=================================================="
    echo "macOS version : $(sw_vers -productVersion)"
    echo "Architecture  : $(uname -m)"
    echo "Environment   : $ENV_PATH"
    echo

    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: Conda / Miniconda was not detected."
        exit 1
    fi

    if ! command -v git >/dev/null 2>&1; then
        echo "ERROR: git was not detected."
        exit 1
    fi

    if ! xcode-select -p >/dev/null 2>&1; then
        echo "ERROR: Xcode Command Line Tools are not installed."
        echo "Please run first: xcode-select --install"
        exit 1
    fi

    if ! xcrun --find clang++ >/dev/null 2>&1; then
        echo "ERROR: Apple Clang C++ compiler was not detected."
        exit 1
    fi

    if [ ! -x "$PY" ]; then
        echo "===== Creating Python 3.10 environment ====="
        conda create -p "$ENV_PATH" python=3.10 pip -y
    else
        echo "===== Using existing environment ====="
    fi

    check_native_arch

    echo
    echo "===== Pinning Python build tools ====="
    "$PY" -m pip install \
        "pip==24.0" \
        "setuptools==69.5.1" \
        "wheel==0.43.0"

    echo
    echo "===== Installing PyTorch 2.1.2 (macOS) ====="
    "$PY" -m pip install \
        "torch==2.1.2" \
        "torchvision==0.16.2" \
        "torchaudio==2.1.2"

    echo
    echo "===== Installing base compatibility dependencies ====="
    "$PY" -m pip install \
        "numpy==1.26.4" \
        "mmengine==0.10.7" \
        "opencv-python==4.10.0.84" \
        ninja \
        psutil \
        ftfy \
        regex

    echo
    echo "===== Removing potentially conflicting MMCV packages ====="
    "$PY" -m pip uninstall -y \
        mmcv \
        mmcv-lite \
        mmcv-full >/dev/null 2>&1 || true

    echo
    echo "===== Fetching MMCV 2.1.0 ====="

    MMCV_SRC="$ENV_PATH/src/mmcv-2.1.0"
    mkdir -p "$ENV_PATH/src"

    if [ ! -d "$MMCV_SRC/.git" ]; then
        rm -rf "$MMCV_SRC"

        git clone \
            --branch v2.1.0 \
            --depth 1 \
            https://github.com/open-mmlab/mmcv.git \
            "$MMCV_SRC"
    fi

    MMCV_TAG="$(git -C "$MMCV_SRC" describe --tags --exact-match 2>/dev/null || true)"

    if [ "$MMCV_TAG" != "v2.1.0" ]; then
        echo "ERROR: MMCV source tag is not v2.1.0."
        exit 1
    fi

    echo
    echo "===== Building MMCV 2.1.0 macOS Ops ====="

    export CC="$(xcrun --find clang)"
    export CXX="$(xcrun --find clang++)"
    export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
    export MAX_JOBS="${MAX_JOBS:-4}"

    (
        cd "$MMCV_SRC"

        MMCV_WITH_OPS=1 \
        "$PY" -m pip install \
            -v \
            -e . \
            --no-build-isolation
    )

    echo
    echo "===== Verifying MMCV immediately ====="

    "$PY" - <<'PY'
import mmcv
import mmcv._ext
from mmcv.ops import MultiScaleDeformableAttention

assert mmcv.__version__ == "2.1.0", mmcv.__version__

print("[PASS] MMCV version :", mmcv.__version__)
print("[PASS] MMCV _ext    :", mmcv._ext.__file__)
print("[PASS] MMCV Ops     : MultiScaleDeformableAttention")
PY

    echo
    echo "===== Installing MMSegmentation 1.2.2 ====="

    "$PY" -m pip install \
        "mmsegmentation==1.2.2"

    echo
    echo "===== Installing bundled MMDetection ====="

    "$PY" -m pip install \
        -v \
        --no-build-isolation \
        "$ROOT/mmdetection"

    echo
    echo "===== Re-pinning key compatibility versions ====="

    "$PY" -m pip install \
        "numpy==1.26.4" \
        "opencv-python==4.10.0.84" \
        "mmengine==0.10.7"

    echo
    echo "===== pip dependency check ====="
    "$PY" -m pip check

    echo
    echo "=================================================="
    echo " macOS setup completed"
    echo "=================================================="
    echo
    echo "Continue with the full CPU reproduction check:"
    echo
    echo "bash reproduce_macos.sh check \"$ENV_PATH\""
    echo
    ;;

check)
    require_macos
    require_env
    require_package_files

    cd "$ROOT"

    echo "=================================================="
    echo " Ours IDJC320 - macOS Full Reproduction Check"
    echo "=================================================="
    echo

    check_native_arch

    "$PY" - <<PY
import os
import sys
import platform

ROOT = os.path.realpath(r"$ROOT")
CONFIG = os.path.realpath(r"$CONFIG")
CKPT = os.path.realpath(r"$CKPT")

print()
print("===== 1. Environment =====")

assert sys.version_info[:2] == (3, 10), sys.version
print("[PASS] Python       :", platform.python_version())

import torch
import torchvision
import torchaudio
import numpy
import cv2
import mmcv
import mmcv._ext
import mmengine
import mmseg
import mmdet

assert torch.__version__ == "2.1.2", torch.__version__
assert torchvision.__version__ == "0.16.2", torchvision.__version__
assert torchaudio.__version__ == "2.1.2", torchaudio.__version__
assert numpy.__version__ == "1.26.4", numpy.__version__
assert mmcv.__version__ == "2.1.0", mmcv.__version__
assert mmengine.__version__ == "0.10.7", mmengine.__version__
assert mmseg.__version__ == "1.2.2", mmseg.__version__
assert mmdet.__version__ == "3.3.0", mmdet.__version__

print("[PASS] PyTorch      :", torch.__version__)
print("[PASS] TorchVision  :", torchvision.__version__)
print("[PASS] TorchAudio   :", torchaudio.__version__)
print("[PASS] NumPy        :", numpy.__version__)
print("[PASS] OpenCV       :", cv2.__version__)
print("[PASS] MMCV         :", mmcv.__version__)
print("[PASS] MMEngine     :", mmengine.__version__)
print("[PASS] MMSeg        :", mmseg.__version__)
print("[PASS] MMDet        :", mmdet.__version__)
print("[INFO] MPS built    :", torch.backends.mps.is_built())
print("[INFO] MPS available:", torch.backends.mps.is_available())

print()
print("===== 2. Local Ours source =====")

mmseg_path = os.path.realpath(mmseg.__file__)
mmdet_path = os.path.realpath(mmdet.__file__)

print("[INFO] MMSEG PATH:", mmseg_path)
print("[INFO] MMDET PATH:", mmdet_path)

assert mmseg_path.startswith(os.path.join(ROOT, "mmseg")), \
    "Loaded mmseg is not the bundled package source"
assert mmdet_path.startswith(os.path.join(ROOT, "mmdetection")), \
    "Loaded mmdetection is not the bundled package source"

from mmseg.models.necks.lgfe_neck import LGFENeck
from mmseg.models.decode_heads.mask2former_head_cdcc import Mask2FormerHeadCDCC

print("[PASS] Local MMSEG source")
print("[PASS] Local MMDET source")
print("[PASS] LGFE")
print("[PASS] CDCC")

print()
print("===== 3. MMCV Ops =====")

from mmcv.ops import MultiScaleDeformableAttention
from mmcv.ops.multi_scale_deform_attn import \
    multi_scale_deformable_attn_pytorch

assert os.path.isfile(mmcv._ext.__file__)

print("[PASS] mmcv._ext :", mmcv._ext.__file__)
print("[PASS] MultiScaleDeformableAttention import")

print()
print("===== 4. Final Config =====")

from mmengine.config import Config

cfg = Config.fromfile(CONFIG)

assert cfg.model.neck.type == "LGFENeck"
assert cfg.model.decode_head.type == "Mask2FormerHeadCDCC"
assert cfg.model.decode_head.num_classes == 15

print("[PASS] Config loaded")
print("[PASS] Neck        : LGFENeck")
print("[PASS] Decode head : Mask2FormerHeadCDCC")
print("[PASS] Classes     : 15")

print()
print("===== 5. Final Checkpoint + CPU Model Build =====")

from mmseg.apis import init_model

model = init_model(
    CONFIG,
    CKPT,
    device="cpu"
)

model.eval()

print("[PASS] Final checkpoint loaded")
print("[PASS] Ours model built on CPU")

print()
print("===== 6. Real CPU Forward / Inference =====")

import numpy as np
from mmseg.apis import inference_model

# Synthetic image avoids dependence on an external dataset while still
# executing the complete segmentation inference path.
image = np.zeros((512, 512, 3), dtype=np.uint8)

torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

with torch.inference_mode():
    result = inference_model(model, image)

assert hasattr(result, "pred_sem_seg")
pred = result.pred_sem_seg.data

assert pred.numel() > 0
assert int(pred.min()) >= 0
assert int(pred.max()) < 15

print("[PASS] Real CPU inference")
print("[PASS] Prediction shape :", tuple(pred.shape))
print("[PASS] Prediction range :", int(pred.min()), "-", int(pred.max()))

print()
print("==================================================")
print(" macOS Ours CPU reproduction: PASS")
print("==================================================")
print()
print("Notes:")
print("- The model was successfully constructed")
print("- The final checkpoint was successfully loaded")
print("- LGFE / CDCC were successfully loaded")
print("- One complete CPU segmentation forward pass was completed")
print("- MPS status is informational only and is not a pass condition for this reproduction workflow")
PY

    echo
    echo "=================================================="
    echo " macOS checks passed"
    echo "=================================================="
    ;;

*)
    echo "Usage:"
    echo
    echo "  bash reproduce_macos.sh setup [environment_path]"
    echo "  bash reproduce_macos.sh check [environment_path]"
    echo
    exit 1
    ;;

esac
