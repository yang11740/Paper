_base_ = [
    'fcenet_resnet50_fpn_1500e_icdar2015.py'
]

# 1. 权重配置：使用 ImageNet 预训练权重作为对照
load_from = None
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='torchvision://resnet50' # 标准 ImageNet 权重
        )
    )
)

# 2. 数据集路径配置 (必须与 ODM 版本完全一致)
data_root = 'data/icdar2015/'

train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=8,       # 必须与 ODM 版本一致
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='OCRDataset',
        data_root=data_root,
        ann_file='textdet_train.json',
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='OCRDataset',
        data_root=data_root,
        ann_file='textdet_test.json',
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

# 3. 优化器调整 (必须与 ODM 版本一致)
optim_wrapper = dict(
    optimizer=dict(lr=1e-3 / 2)
)

# 4. 训练结果保存目录
work_dir = 'work_dirs/fcenet_baseline_finetune_ic15'