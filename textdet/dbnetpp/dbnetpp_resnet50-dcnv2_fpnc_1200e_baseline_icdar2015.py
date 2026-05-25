_base_ = [
    'dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py'
]

# 1. 权重配置：这里使用 MMOCR 官方提供的预训练权重作为 Baseline
# 论文中的 Baseline 通常是先在 SynthText 预训练，再在 IC15 微调
# 这里的 load_from 指向的就是官方在 SynthText 上练好的 DBNet++
load_from = 'https://download.openmmlab.com/mmocr/textdet/dbnetpp/tmp_1.0_pretrain/dbnetpp_r50dcnv2_fpnc_100k_iter_synthtext-20220502-352fec8a.pth'

# 2. 模型结构：Baseline 使用标准的 ResNet (非 Deep Stem)
# 注意：官方权重是为标准 ResNet 训练的，所以这里不需要写 model=dict(backbone=...)
# 它会直接继承 _base_ 里的标准结构

# 3. 数据集路径配置（必须与 ODM 版完全一致）
data_root = 'data/icdar2015/'

train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline

# 4. 训练环境配置（必须与 ODM 版完全一致，确保公平对比）
train_dataloader = dict(
    batch_size=8,            # 同样设为 2
    num_workers=4,            # 同样设为 1
    persistent_workers=False, # 同样设为 False
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

# 5. 优化器与学习率（必须与 ODM 版一致）
# 由于 Batch Size 从 16 缩减到 2，学习率必须同样缩小 8 倍
optim_wrapper = dict(
    optimizer=dict(lr=0.007 / 8)
)

# 6. 实验记录目录
work_dir = 'work_dirs/dbnetpp_baseline_finetune_ic15'