_base_ = (
    './my_mask2former_swin-b_LGFE-20k_'
    'IDJC320-512x512.py'
)

# IMPORTANT:
# The LGFE base config already needs LGFENeck.
# This combined config must explicitly register BOTH custom modules.
custom_imports = dict(
    imports=[
        'mmseg.models.necks.lgfe_neck',
        'mmseg.models.decode_heads.mask2former_head_cdcc',
    ],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(init_cfg=None),
    decode_head=dict(
        type='Mask2FormerHeadCDCC',

        # Keep EXACTLY the same CDCC settings
        # as the CDCC-only ablation.
        cdcc_loss_weight=0.1,
        cdcc_temperature=0.1,
        cdcc_points_per_class=32,
    )
)

work_dir = (
    './work_dirs/'
    'mask2former_swin-b_LGFE_CDCC_'
    'IDJC320-20k-512x512'
)
