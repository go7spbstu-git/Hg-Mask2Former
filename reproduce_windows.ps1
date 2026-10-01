param(
    [Parameter(Position=0)]
    [ValidateSet("setup","check","test","smoke","train")]
    [string]$Action = "check",

    [string]$EnvPath = ""
)

$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

if ([string]::IsNullOrWhiteSpace($EnvPath)) {
    $EnvPath = Join-Path (Split-Path $ROOT -Parent) "ours_idjc320_win_env"
}

$PY = Join-Path $EnvPath "python.exe"

$CONFIG = "configs\mask2former\my_mask2former_swin-b_LGFE_CDCC-20k_IDJC320-512x512.py"
$CKPT   = "checkpoint\best_mIoU_iter_3000.pth"

$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = "$ROOT;$ROOT\mmdetection"

function Require-Env {
    if (-not (Test-Path $PY)) {
        throw "Windows environment not found: $EnvPath`nPlease run setup first."
    }
}

function Run-Python {
    param([string]$Code)
    & $PY -c $Code
    if ($LASTEXITCODE -ne 0) {
        throw "Python check failed."
    }
}

switch ($Action) {

    "setup" {
        Write-Host "===== Windows environment setup ====="
        Write-Host "Environment path: $EnvPath"

        if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
            throw "Conda was not detected. Please run this script from Anaconda PowerShell Prompt."
        }

        $CACHE = Join-Path (Split-Path $EnvPath -Parent) "ours_idjc320_win_cache"
        $CONDA_CACHE = Join-Path $CACHE "conda_pkgs"
        $PIP_CACHE = Join-Path $CACHE "pip_cache"
        $TMP_CACHE = Join-Path $CACHE "tmp"

        New-Item -ItemType Directory -Force -Path $CONDA_CACHE | Out-Null
        New-Item -ItemType Directory -Force -Path $PIP_CACHE | Out-Null
        New-Item -ItemType Directory -Force -Path $TMP_CACHE | Out-Null

        $env:CONDA_PKGS_DIRS = $CONDA_CACHE
        $env:PIP_CACHE_DIR = $PIP_CACHE
        $env:TEMP = $TMP_CACHE
        $env:TMP = $TMP_CACHE

        if (-not (Test-Path $PY)) {
            conda create -p $EnvPath python=3.10 -y
            if ($LASTEXITCODE -ne 0) { throw "Conda environment creation failed." }
        }
        else {
            Write-Host "The environment already exists. Continuing with dependency installation/correction."
        }

        Write-Host "===== Pinning Windows Python packaging tools ====="

        & $PY -m pip install --force-reinstall `
            "setuptools==81.0.0"

        if ($LASTEXITCODE -ne 0) {
            throw "Failed to pin setuptools."
        }

        & $PY -c "import setuptools,pkg_resources; assert setuptools.__version__=='81.0.0', setuptools.__version__; print('[PASS] setuptools   :',setuptools.__version__); print('[PASS] pkg_resources')"

        if ($LASTEXITCODE -ne 0) {
            throw "pkg_resources check failed."
        }

        Write-Host "===== Pre-pinning NumPy / OpenCV ====="

        & $PY -m pip install `
            "numpy==1.26.4" `
            "opencv-python==4.10.0.84"

        if ($LASTEXITCODE -ne 0) {
            throw "Failed to pin NumPy / OpenCV."
        }

        & $PY -m pip install `
            torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 `
            --index-url https://download.pytorch.org/whl/cu121
        if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }

        & $PY -m pip install "mmengine==0.10.7"
        if ($LASTEXITCODE -ne 0) { throw "MMEngine installation failed." }

        & $PY -m pip install "opencv-python==4.10.0.84"

        & $PY -m pip install `
            --only-binary=mmcv `
            "mmcv==2.1.0" `
            -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
        if ($LASTEXITCODE -ne 0) { throw "MMCV installation failed." }

        & $PY -m pip install `
            "mmsegmentation==1.2.2" `
            "mmdet==3.3.0" `
            ftfy regex

        if ($LASTEXITCODE -ne 0) { throw "MMSeg/MMDet installation failed." }

        Write-Host "===== Pinning compatibility versions ====="

        & $PY -m pip install --force-reinstall `
            "setuptools==81.0.0" `
            "numpy==1.26.4" `
            "opencv-python==4.10.0.84" `
            "typing-extensions==4.15.0"

        if ($LASTEXITCODE -ne 0) { throw "Failed to pin compatibility versions." }

        & $PY -m pip check
        if ($LASTEXITCODE -ne 0) { throw "pip check detected dependency conflicts." }

        Write-Host ""
        Write-Host "========== Windows setup completed =========="
        Write-Host "Next step:"
        Write-Host ".\reproduce_windows.ps1 check -EnvPath `"$EnvPath`""
    }

    "check" {
        Require-Env

        Write-Host "===== Automatic Windows Ours checks ====="

        Run-Python "import sys; print('[PASS] Python       :',sys.version.split()[0])"

        Run-Python "import setuptools,pkg_resources; assert setuptools.__version__=='81.0.0', setuptools.__version__; print('[PASS] setuptools   :',setuptools.__version__); print('[PASS] pkg_resources')"

        Run-Python "import torch; print('[PASS] PyTorch      :',torch.__version__); print('[PASS] CUDA Runtime :',torch.version.cuda); assert torch.cuda.is_available(); print('[PASS] GPU          :',torch.cuda.get_device_name(0))"

        Run-Python "import numpy,cv2,mmcv,mmengine,mmseg,mmdet; print('[PASS] NumPy        :',numpy.__version__); print('[PASS] OpenCV       :',cv2.__version__); print('[PASS] MMCV         :',mmcv.__version__); print('[PASS] MMEngine     :',mmengine.__version__); print('[PASS] MMSeg        :',mmseg.__version__); print('[PASS] MMDet        :',mmdet.__version__)"

        Run-Python "import mmcv._ext; from mmcv.ops import MultiScaleDeformableAttention; print('[PASS] MMCV CUDA Ops')"

        Run-Python "import mmseg,mmdet; assert mmseg.__file__.lower().startswith(r'$ROOT'.lower()); assert mmdet.__file__.lower().startswith(r'$ROOT'.lower()); print('[PASS] Local MMSeg/MMDet')"

        Run-Python "from mmseg.models.necks.lgfe_neck import LGFENeck; print('[PASS] LGFE'); from mmseg.models.decode_heads.mask2former_head_cdcc import Mask2FormerHeadCDCC; print('[PASS] CDCC')"

        Run-Python "from mmengine.config import Config; c=Config.fromfile(r'$CONFIG'); assert c.model.neck.type=='LGFENeck'; assert c.model.decode_head.type=='Mask2FormerHeadCDCC'; assert c.model.decode_head.num_classes==15; print('[PASS] Final Config')"

        Run-Python "from mmengine.config import Config; from mmengine.runner import load_checkpoint; from mmseg.utils import register_all_modules; from mmseg.registry import MODELS; import mmseg.models.necks.lgfe_neck; import mmseg.models.decode_heads.mask2former_head_cdcc; register_all_modules(init_default_scope=True); c=Config.fromfile(r'$CONFIG'); m=MODELS.build(c.model); n=sum(p.numel() for p in m.parameters()); assert n==108483276, n; load_checkpoint(m,r'$CKPT',map_location='cpu'); print('[PASS] Final Params     :',format(n,',')); print('[PASS] Final Checkpoint')"

        Write-Host ""
        Write-Host "========== All Windows checks passed =========="
    }

    "test" {
        Require-Env
        Write-Host "===== Formal Ours validation ====="
        Write-Host "Reference hardware: NVIDIA GeForce RTX 4090 24GB. Formal validation uses all 48 images in val.txt."
        Write-Host "Reference target: mIoU=73.92  mAcc=82.39  aAcc=91.27"

        & $PY tools\test.py $CONFIG $CKPT `
            --work-dir work_dirs\final_test_windows `
            --cfg-options `
            test_dataloader.dataset.ann_file=val.txt `
            test_dataloader.num_workers=0 `
            test_dataloader.persistent_workers=False

        if ($LASTEXITCODE -ne 0) { throw "Windows test failed." }
    }

    "smoke" {
        Require-Env
        Write-Host "===== Windows 20-iteration smoke test ====="
        Write-Host "Recommended GPU memory: approximately 24GB on an NVIDIA GPU."

        & $PY tools\train.py $CONFIG `
            --work-dir work_dirs\smoke_test_windows `
            --cfg-options `
            train_cfg.max_iters=20 `
            train_cfg.val_interval=20 `
            default_hooks.checkpoint.interval=20

        if ($LASTEXITCODE -ne 0) { throw "Windows smoke test failed." }
    }

    "train" {
        Require-Env
        Write-Host "===== Full Windows training ====="
        Write-Host "Recommended GPU memory: approximately 24GB on an NVIDIA GPU."

        & $PY tools\train.py $CONFIG `
            --work-dir work_dirs\ours_retrain_full_windows

        if ($LASTEXITCODE -ne 0) { throw "Windows training failed." }
    }
}
