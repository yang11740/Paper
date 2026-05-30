# 继承基础 ODM 预训练配置
_base_ = [
    '../textdet/_base_/default_runtime.py'
]

# 显式导入 mmdet 用于 Deep Stem ResNet
custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)

# --- 模型配置 ---
_dictionary_path = 'data/chinese_dict.txt'
with open(_dictionary_path, 'r', encoding='utf-8') as f:
    _chars = f.read().splitlines()
del f  # 必须删除文件对象包装器，否则 MMEngine 无法序列化配置
vocab_size = len(_chars) + 4  # +4 用于 [PAD], [BOS], [EOS], [UNK]

model = dict(
    type='ODM',
    # 1. 图像编码器 (必须开启 Deep Stem 以适配你的应用改进)
    image_encoder=dict(
        type='mmdet.ResNet',
        depth=50,
        deep_stem=True,
        avg_down=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    # 2. 文本编码器 (关键修改：vocab_size 适配中文)
    text_encoder=dict(
        type='TransformerEncoder',
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        max_seq_len=25
    ),
    # 3. 解码器 (重建二值图)
    decoder=dict(
        type='ODMDecoder',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        scale_factor=4
    ),
    # 损失函数权重
    loss_weights=dict(loss_seg=1.0, loss_ocr=1.0, loss_bc=0.5)
)

# --- 数据流水线 (Pipeline) ---
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(type='LoadImageFromFile', color_type='grayscale', key='gt_mask'), # 加载合成的印刷体二值图
    dict(type='Resize', scale=(512, 512)),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    dict(type='PackTextDetInputs', meta_keys=('img_path', 'ori_shape', 'img_shape', 'text'))
]

# --- 数据集加载 ---
data_root = 'data/synthetic_chinese_handwriting/' # 指向你的 5万张合成数据

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='OCRDataset', # 或者使用项目定义的专用 ODMDataset
        data_root=data_root,
        ann_file='labels.json', # 合成数据时生成的标注文件
        pipeline=train_pipeline
    )
)

# --- 优化器与调度 ---
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05)
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None
visualizer = None

# 日志与保存
work_dir = 'work_dirs/odm_pretrain_chinese_handwriting'