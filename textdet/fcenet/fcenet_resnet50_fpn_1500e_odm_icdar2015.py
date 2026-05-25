_base_ = [
    'fcenet_resnet50_fpn_1500e_icdar2015.py'
]

# 显式导入 mmdet，用于调用支持 Deep Stem 的 ResNet 结构
custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)

# 1. 禁用官方预训练权重下载
load_from = None

# 2. 修改模型配置：将 Backbone 替换为与 ODM 权重匹配的 Deep Stem 结构
model = dict(
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        deep_stem=True,      # 【关键】匹配 ODM 权重的 Deep Stem 结构
        avg_down=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./odm_mmocr.pth', # 填写你服务器上的实际路径
            prefix='backbone.'  # 确保与你之前转换好的 Key 匹配
        )
    )
)

# 3. 数据集与路径配置
data_root = 'data/icdar2015/'

# 引用 base 里的 pipeline
train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=8,           # V100 16G 建议设为 4 或 8
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

# 4. 优化器调整
# 原始 base 配置 lr=1e-3 是针对 batch_size=8 的
# 如果你改为 batch_size=4，建议 lr 改为 5e-4
optim_wrapper = dict(
    optimizer=dict(lr=1e-3 / 2)
)

# 5. 训练结果保存目录
work_dir = 'work_dirs/fcenet_odm_finetune_ic15'