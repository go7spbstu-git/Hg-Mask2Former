# Mask2Former-Swin-B + LGFE first structural-improvement experiment.
# Everything else inherits from the completed 512x512 baseline.

_base_ = './my_mask2former_swin-b-in22k-384x384-pre_8xb2-20k_IDJC320-512x512.py'

custom_imports = dict(
    imports=['mmseg.models.necks.lgfe_neck'],
    allow_failed_imports=False
)

model = dict(
    neck=dict(
        type='LGFENeck',
        in_channels=[128, 256, 512, 1024],
        reduction=16,
    )
)

work_dir = './work_dirs/mask2former_swin-b_LGFE_IDJC320-20k-512x512'
