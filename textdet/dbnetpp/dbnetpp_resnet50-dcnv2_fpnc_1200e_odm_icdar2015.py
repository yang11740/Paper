_base_ = [
    'dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py'
]

custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)
load_from = None


model = dict(
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        deep_stem=True,
        avg_down=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./odm_mmocr.pth',
            prefix='backbone.'  # 关键点：告诉框架加载时削掉 'backbone.' 前缀去匹配
        )
    )
)

data_root = 'data/icdar2015/'

# --- 重新引用 base 里的 pipeline ---
train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                data_root=data_root,
                ann_file='textdet_train.json'
            )
        ],
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                data_root=data_root,
                ann_file='textdet_test.json',
                test_mode=True
            )
        ],
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

optim_wrapper = dict(
    optimizer=dict(lr=0.007 / 8)
)

work_dir = 'work_dirs/dbnetpp_odm_finetune_ic15'