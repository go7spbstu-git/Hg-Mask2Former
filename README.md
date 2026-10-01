# Ours IDJC320 Cross-Platform Reproducibility Package

This package contains the final reproducibility workflow for the proposed model:

**Mask2Former-Swin-B + LGFE + CDCC**

Formal reference results on the 48-image `val.txt` split:

- aAcc = 91.27%
- mIoU = 73.92%
- mAcc = 82.39%

Final checkpoint:

`checkpoint/best_mIoU_iter_3000.pth`

## Platform-specific documentation

| Platform | Reference hardware | Documentation | Reproduction script |
|---|---|---|---|
| Ubuntu / Linux | NVIDIA GeForce RTX 4090 24GB | `README_Ubuntu_Linux.md` | `reproduce_ubuntu_linux.sh` |
| Windows | NVIDIA GeForce RTX 4090 24GB | `README_Windows.md` | `reproduce_windows.ps1` |
| macOS | Apple M5 (Apple Silicon, arm64) | `README_macOS.md` | `reproduce_macos.sh` |

For backward compatibility, `reproduce.sh` is retained as a thin wrapper that forwards all arguments to `reproduce_ubuntu_linux.sh`.

## Recommended execution order

For Ubuntu / Linux and Windows, use the following sequence:

1. `setup`
2. `check`
3. `test`
4. `smoke`
5. `train` (only when full retraining is required)

For macOS, the standard cross-platform verification path is CPU-based and uses:

1. `setup`
2. `check`

The macOS script reports MPS availability for reference, but MPS is not required for the standard verification path.

## Important notes

- The formal mIoU of 73.92% corresponds to `val.txt`.
- `test.txt` is an independent split and is not expected to reproduce the same metric values.
- Full retraining includes random augmentation, random sampling, and device-dependent numerical effects; the best iteration and the last decimal digits may vary slightly.
- Do not change the pinned dependency versions unless necessary for a new environment.
