_base_ = [
    'psenet_resnet50_fpnf_600e_icdar2015.py'
]

# 1. 禁用冲突权重
load_from = None

# 2. 模型配置：加载 oCLIP 的 Backbone 权重
model = dict(
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        deep_stem=False,   # oCLIP 使用标准 ResNet
        init_cfg=dict(
            type='Pretrained',
            # 填写你刚才下载的文件的绝对路径
            checkpoint='./psenet_resnet50-oclip_fpnf_600e_icdar2015_20221101_131357-2bdca389.pth',
            # 关键：这个文件是完整模型，我们只需要 backbone 部分
            prefix='backbone.' 
        )
    )
)

# 3. 硬件与数据适配（必须与你的 ODM 实验完全一致，确保公平）
data_root = 'data/icdar2015/'
train_dataloader = dict(
    batch_size=8,        # 保持与 ODM 实验一致
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='OCRDataset',
        data_root=data_root,
        ann_file='textdet_train.json',
        pipeline=_base_.train_pipeline
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
        pipeline=_base_.test_pipeline
    )
)

test_dataloader = val_dataloader

# 4. 优化器调整（保持一致）
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4 / 4)
)

# 5. 保存目录
work_dir = 'work_dirs/psenet_oclip_pretrain_ic15'