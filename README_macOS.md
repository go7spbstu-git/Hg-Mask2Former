# Ours IDJC320 — macOS Reproduction Guide

## 1. Scope

This guide is used to verify the cross-platform executability of the final Ours IDJC320 model on macOS.

The documented reference device for this open-source release is **Apple M5 (Apple Silicon, arm64)**.

Because the macOS compute backend differs from Ubuntu / Windows CUDA environments, the standard macOS reproduction path uses **CPU**. The script reports MPS status automatically, but MPS availability is not required for a successful reproduction check.

## 2. Reference environment

- macOS
- Apple M5 (Apple Silicon, arm64)
- Native-architecture Conda / Miniconda
- Python 3.10
- Xcode Command Line Tools
- Git
- Network connection

Core package versions:

- PyTorch 2.1.2
- TorchVision 0.16.2
- TorchAudio 2.1.2
- NumPy 1.26.4
- OpenCV 4.10.0.84
- MMEngine 0.10.7
- MMCV 2.1.0
- MMSegmentation 1.2.2
- MMDetection 3.3.0

MMCV 2.1.0 is compiled locally with Apple Clang to provide the required macOS CPU Ops.

## 3. File locations

Keep this file and `reproduce_macos.sh` in the package root:

```text
Ours_IDJC320_Final_V2_20260920/
├── reproduce_ubuntu_linux.sh
├── reproduce.sh
├── reproduce_macos.sh
├── README_macOS.md
├── configs/
├── checkpoint/
├── mmseg/
└── mmdetection/
```

Do not move `reproduce_macos.sh` out of the package root.

## 4. Initial setup

Open Terminal and enter the package root:

```bash
cd /path/to/Ours_IDJC320_Final_V2_20260920
```

Grant execute permission:

```bash
chmod +x reproduce_macos.sh
```

Install the environment:

```bash
bash reproduce_macos.sh setup
```

Default environment path:

`~/ours_idjc320_mac_env`

You may also specify a custom path:

```bash
bash reproduce_macos.sh setup /your/custom/env/path
```

If Xcode Command Line Tools are not installed, run:

```bash
xcode-select --install
```

## 5. Full reproduction check

After setup completes, run:

```bash
bash reproduce_macos.sh check
```

If a custom environment path is used:

```bash
bash reproduce_macos.sh check /your/custom/env/path
```

The script checks:

1. macOS and Python architecture;
2. Python / PyTorch / OpenMMLab versions;
3. MMCV `_ext` and `MultiScaleDeformableAttention`;
4. bundled MMSeg / MMDetection sources;
5. LGFE and CDCC;
6. the final configuration;
7. `best_mIoU_iter_3000.pth`;
8. complete CPU model construction;
9. one real 512×512 segmentation inference pass;
10. `pred_sem_seg` output and class range.

## 6. Success criteria

A successful run ends with:

```text
macOS Ours CPU reproduction: PASS
```

and:

```text
macOS checks passed
```

This indicates that the final model has passed the standard CPU cross-platform reproduction check on macOS.

## 7. MPS note

The script reports:

```text
MPS built
MPS available
```

These values are recorded only as information about the Apple GPU / PyTorch state.

The standard reproduction path does not depend on MPS. Even if MPS is unavailable, the macOS reproduction check is considered successful as long as the full CPU reproduction check passes.

## 8. Apple M5 / Apple Silicon note

The documented macOS reference device is **Apple M5 (arm64)**. Use a native arm64 Conda / Miniconda installation.

The script compares:

```text
Host architecture
Python architecture
```

and stops if they differ, preventing compatibility issues caused by running an x86_64 Python environment through Rosetta on Apple Silicon.

## 9. If MMCV compilation fails

Do not replace MMCV with `mmcv-lite`.

First verify that the following commands return valid paths:

```bash
xcode-select -p
xcrun --find clang
xcrun --find clang++
```

If compilation still fails, retain the complete terminal error log for diagnosis.

## 10. Platform note

Ubuntu / NVIDIA CUDA is the primary full-training reproduction environment.

Windows / NVIDIA CUDA is supported by the Windows reproduction workflow.

macOS uses the same final configuration, final checkpoint, LGFE, CDCC, and project source code, with CPU as the standard cross-platform verification backend.
