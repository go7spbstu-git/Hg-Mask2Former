# Ours IDJC320 — Windows Reproduction Guide

## 1. Platform scope

This guide applies to native Windows environments.

The documented reference hardware for this open-source release is:

- Windows 10 Pro
- NVIDIA GeForce RTX 4090 24GB
- Python 3.10.21
- setuptools 81.0.0
- PyTorch 2.1.2+cu121
- CUDA Runtime 12.1
- NumPy 1.26.4
- OpenCV 4.10.0.84
- MMCV 2.1.0
- MMEngine 0.10.7
- MMSegmentation 1.2.2
- MMDetection 3.3.0

The Windows workflow checks the following components:

- PyTorch CUDA availability
- MMCV CUDA Ops
- bundled MMSeg / MMDetection sources
- LGFE
- CDCC
- final configuration parsing
- final Ours model construction
- `best_mIoU_iter_3000.pth` loading

The formal validation workflow uses the 48-image `val.txt` split. Full smoke testing and full retraining require more GPU memory than checkpoint loading or basic environment checks; the documented reference GPU is an NVIDIA GeForce RTX 4090 24GB.

## 2. Windows extraction note

The Linux-built package contains Linux symbolic links under:

`mmseg/.mim/`

A standard Windows `tar` extraction may report messages such as:

```text
Can't create ... mmseg\.mim\...
```

This does not indicate corruption of the model, dataset, or checkpoint.

When extracting in Windows PowerShell, exclude `.mim`:

```powershell
tar -xf Ours_IDJC320_Final_Repro_CrossPlatform_WindowsFixed_20260927.tar `
-C "TARGET_DIRECTORY" `
--exclude="*/.mim/*" `
--exclude="*/.mim"
```

The `.mim` directory is not required for running the final model in this reproduction package.

## 3. One-command environment setup

Use **Anaconda PowerShell Prompt**.

From the project root, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproduce_windows.ps1 setup
```

The script creates a Python 3.10 environment and installs:

- setuptools 81.0.0
- PyTorch 2.1.2+cu121
- TorchVision 0.16.2+cu121
- TorchAudio 2.1.2+cu121
- MMEngine 0.10.7
- MMCV 2.1.0
- MMSegmentation 1.2.2
- MMDetection 3.3.0
- NumPy 1.26.4
- OpenCV 4.10.0.84
- ftfy
- regex

The Windows setup explicitly pins `setuptools==81.0.0` for compatibility with project components that still rely on `pkg_resources`.

A `pkg_resources` deprecation `UserWarning` is not a reproduction failure if the subsequent `pkg_resources` check reports PASS.

To specify a custom Conda environment path:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproduce_windows.ps1 `
setup `
-EnvPath "E:\your_path\ours_win_env"
```

## 4. Automatic Windows checks

After setup, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproduce_windows.ps1 check
```

With a custom environment path:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproduce_windows.ps1 `
check `
-EnvPath "E:\your_path\ours_win_env"
```

A successful run ends with:

```text
========== All Windows checks passed ==========
```

The script checks:

- Python
- PyTorch
- CUDA Runtime
- NVIDIA GPU
- NumPy
- OpenCV
- MMCV
- MMEngine
- MMSeg
- MMDet
- MMCV CUDA Ops
- Local MMSeg/MMDet
- LGFE
- CDCC
- Final Config
- Final Checkpoint

## 5. Formal model evaluation

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproduce_windows.ps1 test
```

The script loads:

`checkpoint\best_mIoU_iter_3000.pth`

and evaluates the formal `val.txt` split.

Reference results:

- aAcc = 91.27%
- mIoU = 73.92%
- mAcc = 82.39%

The formal validation split contains 48 images.

Small last-decimal differences can occur across GPUs or operating systems because of numerical implementation details and rounding.

## 6. 20-iteration smoke test

Before full training, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproduce_windows.ps1 smoke
```

This step trains for only 20 iterations and checks:

- ADE20K pretrained checkpoint loading
- dataset loading
- LGFE
- CDCC
- backpropagation
- checkpoint saving
- validation

The metric after 20 iterations is not a formal experimental result.

## 7. Full retraining

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproduce_windows.ps1 train
```

Main training settings:

- max_iters = 20000
- val_interval = 1000
- batch_size = 2
- crop_size = 512×512
- optimizer = AdamW
- learning rate = 0.0001
- CDCC loss weight = 0.1
- CDCC temperature = 0.1
- CDCC points per class = 32

Output directory:

`work_dirs\ours_retrain_full_windows`

## 8. Expected messages during training

You may see:

```text
No pre-trained weights for SwinTransformer, training start from scratch
```

The program then loads the complete ADE20K Mask2Former pretrained checkpoint from:

`pretrained\mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth`

Therefore, the complete model is not trained from random initialization.

You may also see:

```text
size mismatch for decode_head.cls_embed.weight
size mismatch for decode_head.cls_embed.bias
```

This is expected because the ADE20K model uses 150 classes whereas this project uses 15 classes, so the classification head must be reinitialized.

You may also see:

```text
missing keys in source state_dict: neck.blocks...
```

This is also expected. LGFE is introduced in this study and therefore has no corresponding parameters in the official ADE20K checkpoint.

## 9. Core files

Final configuration:

`configs\mask2former\my_mask2former_swin-b_LGFE_CDCC-20k_IDJC320-512x512.py`

Final checkpoint:

`checkpoint\best_mIoU_iter_3000.pth`

LGFE:

`mmseg\models\necks\lgfe_neck.py`

CDCC:

`mmseg\models\decode_heads\mask2former_head_cdcc.py`

Dataset:

`data\my_IDJC320_voc`

## 10. Recommended execution order

Run the following actions in order:

`setup → check → test → smoke → train`

`check` requires less GPU memory than training. The documented Windows reference hardware for the open-source release is **NVIDIA GeForce RTX 4090 24GB**.

## 11. Windows verification scope

The final Windows workflow verifies:

- environment setup;
- CUDA availability;
- MMCV CUDA Ops;
- LGFE;
- CDCC;
- final configuration;
- final model construction;
- final checkpoint loading;
- automatic Windows checks;
- GPU inference path;
- formal evaluation on all 48 `val.txt` images.

Formal reference results:

- aAcc = 91.27%
- mIoU = 73.92%
- mAcc = 82.39%

Full retraining can show small variation in the best iteration and last decimal digits because of stochastic training behavior and hardware-dependent numerical effects.
