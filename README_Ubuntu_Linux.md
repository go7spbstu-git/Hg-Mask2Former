# Ours IDJC320 — Ubuntu / Linux Reproduction Guide

The final model in this package is:

**Mask2Former-Swin-B + LGFE + CDCC**

This reproducibility package has completed a clean-start verification workflow on a fresh Linux GPU server, including environment creation, environment checks, final-model evaluation, and a 20-iteration training smoke test.

Formal reference results:

- aAcc = 91.27%
- mIoU = 73.92%
- mAcc = 82.39%

Final checkpoint:

`checkpoint/best_mIoU_iter_3000.pth`

## 1. Reference server environment

Documented reference environment:

- Ubuntu 22.04
- NVIDIA GeForce RTX 4090 24GB
- Python 3.10
- CUDA 12.1
- Conda / Miniconda

Pinned reproduction environment:

- PyTorch 2.1.2+cu121
- TorchVision 0.16.2+cu121
- TorchAudio 2.1.2+cu121
- NumPy 1.26.4
- OpenCV 4.10.0.84
- MMCV 2.1.0
- MMEngine 0.10.7
- MMSegmentation 1.2.2
- MMDetection 3.3.0

Note: the CUDA version reported by `nvidia-smi` may be newer than 12.1. It indicates the maximum CUDA version supported by the installed NVIDIA driver and does not need to be exactly 12.1.

## 2. Enter the project directory

After extracting the package, enter the project root:

```bash
cd /root/autodl-tmp/Ours_IDJC320_Final_V2_20260920
```

Run all subsequent commands from this directory.

## 3. Step 1 — Create the environment from scratch

Run:

```bash
bash reproduce_ubuntu_linux.sh setup
```

The script creates the Conda environment:

`ours_idjc320`

and installs the pinned dependencies required by this project.

If setup finishes normally, continue to the next step. If a real error is reported, stop and inspect the error before changing any dependency versions.

## 4. Step 2 — Check the environment and project

Run:

```bash
bash reproduce_ubuntu_linux.sh check
```

A successful check should report PASS entries for:

- Python
- PyTorch
- CUDA Runtime
- NumPy
- MMCV
- MMEngine
- MMSeg
- MMDet
- GPU
- MMCV CUDA Ops
- LGFE
- CDCC
- Dataset, weights, and configuration

The final line should be:

```text
========== All checks passed ==========
```

## 5. Step 3 — Evaluate the final model

Run:

```bash
bash reproduce_ubuntu_linux.sh test
```

This step does not retrain the model. It loads:

`checkpoint/best_mIoU_iter_3000.pth`

and evaluates it on the formal `val.txt` split.

Expected reference results:

```text
aAcc: 91.2700
mIoU: 73.9200
mAcc: 82.3900
```

The validation split contains 48 images.

The paper result mIoU = 73.92% corresponds to `val.txt`. `test.txt` is an independent test split and is not expected to reproduce the same value.

## 6. Step 4 — Run the 20-iteration training smoke test

Before attempting a full 20,000-iteration retraining, run:

```bash
bash reproduce_ubuntu_linux.sh smoke
```

This smoke test runs only 20 iterations and checks that:

- the ADE20K pretrained weights load correctly;
- the IDJC320 training data can be read;
- LGFE participates correctly in training;
- CDCC participates correctly in training;
- loss backpropagation works;
- checkpoints can be saved;
- validation can run.

The mIoU obtained after 20 iterations is not a formal experimental result.

## 7. Expected messages during the smoke test

At the beginning of training, you may see:

```text
No pre-trained weights for SwinTransformer, training start from scratch
```

This is followed by:

```text
Loads checkpoint by local backend from path:
pretrained/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth
```

This is expected. Separate online backbone weight loading is disabled, and the complete model is then initialized from the packaged ADE20K Mask2Former pretrained checkpoint.

You may also see:

```text
size mismatch for decode_head.cls_embed.weight
size mismatch for decode_head.cls_embed.bias
```

This is also expected. The ADE20K pretrained model has 150 classes, whereas this project uses 15 classes, so the classification head must be reinitialized.

You may also see:

```text
missing keys in source state_dict: neck.blocks...
```

This is expected as well. LGFE is introduced by this study and does not exist in the official ADE20K pretrained checkpoint, so its parameters are learned from scratch.

These messages do not indicate a training failure.

## 8. Full retraining

Only proceed to full retraining after all of the following have passed:

1. `bash reproduce_ubuntu_linux.sh check`
2. `bash reproduce_ubuntu_linux.sh test`
3. `bash reproduce_ubuntu_linux.sh smoke`

Run full training with:

```bash
bash reproduce_ubuntu_linux.sh train
```

Formal training settings:

- max_iters = 20000
- val_interval = 1000
- batch_size = 2
- crop_size = 512×512
- optimizer = AdamW
- learning rate = 0.0001
- CDCC loss weight = 0.1
- CDCC temperature = 0.1
- CDCC points per class = 32

Training outputs are saved to:

`work_dirs/ours_retrain_full/`

Because full retraining includes random data augmentation, random sampling, and GPU-dependent numerical effects, the best iteration and the last decimal digits may vary slightly.

## 9. Core files

Final configuration:

`configs/mask2former/my_mask2former_swin-b_LGFE_CDCC-20k_IDJC320-512x512.py`

Final checkpoint:

`checkpoint/best_mIoU_iter_3000.pth`

ADE20K initialization checkpoint:

`pretrained/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth`

LGFE:

`mmseg/models/necks/lgfe_neck.py`

CDCC:

`mmseg/models/decode_heads/mask2former_head_cdcc.py`

Dataset:

`data/my_IDJC320_voc/`

Data splits:

- `train.txt`
- `val.txt`
- `test.txt`

## 10. Shortest reproduction workflow

Run the following commands in order:

```bash
bash reproduce_ubuntu_linux.sh setup
bash reproduce_ubuntu_linux.sh check
bash reproduce_ubuntu_linux.sh test
bash reproduce_ubuntu_linux.sh smoke
```

After all four steps pass, run full retraining only if required:

```bash
bash reproduce_ubuntu_linux.sh train
```

## 11. Final verification record

The final reproducibility workflow covers:

- clean environment creation;
- environment and project checks;
- 48/48-image formal validation;
- aAcc = 91.27%;
- mIoU = 73.92%;
- mAcc = 82.39%;
- successful 20-iteration smoke test;
- successful checkpoint saving;
- successful validation execution.

For this open-source release, the documented Ubuntu / Linux reference hardware is **NVIDIA GeForce RTX 4090 24GB**.

Follow the steps above in order and avoid changing the pinned dependency versions unless a new environment requires a documented compatibility update.
