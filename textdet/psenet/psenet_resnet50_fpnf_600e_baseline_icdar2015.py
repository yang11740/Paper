_base_ = [
    'psenet_resnet50_fpnf_600e_icdar2015.py'
]

# 1. 权重配置：Baseline 通常加载官方的 ImageNet 预训练权重或 SynthText 预训练权重
# 论文中的 Baseline 是指没有经过 ODM 进一步对齐任务的模型。
# 这里我们可以选择加载 MMOCR 官方在 SynthText 上练好的 PSENet 权重作为起点
load_from = None

# 2. 显式指定 Backbone 加载标准的 ImageNet 预训练权重
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='torchvision://resnet50' # 自动从 torchvision 下载标准 ResNet50
        )
    )
)

# 3. 数据集路径配置 (必须与 ODM 版本完全一致)
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

# 4. 优化器调整 (必须与 ODM 版本一致)
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4 / 4)
)

# 5. 训练结果保存目录
work_dir = 'work_dirs/psenet_baseline_finetune_ic15'