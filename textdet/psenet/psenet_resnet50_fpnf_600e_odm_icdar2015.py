_base_ = [
    'psenet_resnet50_fpnf_600e_icdar2015.py'
]

# 1. 显式导入 mmdet，用于调用支持 Deep Stem 的 ResNet 结构
custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)

# 2. 禁用官方预训练权重，确保只加载 ODM 权重
load_from = None

# 3. 修改模型结构以匹配 ODM 预训练权重
model = dict(
    backbone=dict(
        type='mmdet.ResNet', # 使用 mmdet 的 ResNet 实现
        depth=50,
        deep_stem=True,      # 【关键】ODM 采用的是 Deep Stem 结构
        avg_down=True,
        init_cfg=dict(
            type='Pretrained',
            # 填写你上传到服务器后的权重路径
            checkpoint='./odm_mmocr.pth',
            prefix='backbone.'
        )
    ),
    # 保持 PSENet 处理 ICDAR2015 的四边形输出设置
    det_head=dict(postprocessor=dict(text_repr_type='quad'))
)

# 4. 数据集路径与加载配置
data_root = 'data/icdar2015/'

# 引用 base 里的 pipeline (数据增强流水线)
train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=8,       # 如果是 V100 16G，可以设为 4 或 8
    num_workers=4,       # Linux 服务器建议设为 4 或 8
    persistent_workers=True,
    dataset=dict(
        type='OCRDataset',
        data_root=data_root,
        ann_file='textdet_train.json', # 指向生成的 JSON 标注
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

# 5. 优化器调整
# PSENet 默认使用 Adam，lr 为 1e-4。
# 由于 Batch Size 从原始的 64 降低到了 4，建议微调学习率
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4 / 4)
)

# 6. 训练结果保存目录
work_dir = 'work_dirs/psenet_odm_finetune_ic15'