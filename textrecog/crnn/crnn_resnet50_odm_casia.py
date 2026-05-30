# 继承 MMOCR 1.x 的基础运行配置
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_base.py', # 手写体识别建议使用 Adam 优化器
]

# 1. 显式导入 mmdet 模块，用于调用支持 Deep Stem 的 ResNet
custom_imports = dict(
    allow_failed_imports=False, 
    imports=[
        'mmdet.models', 
        # 'mmocr.visualizers' # <--- 必须加上这一行
    ])

# --- 字典配置 ---
# 指向你从 CASIA JSON 中提取的包含标点的字典
dictionary = dict(
    type='Dictionary',
    dict_file='E:/ODM/data/casia/casia_dict.txt', 
    with_start=False,
    with_end=False,
    same_start_end=False,
    with_padding=True,
    with_unknown=True)

# --- 模型配置 ---
model = dict(
    type='CRNN',
    # 数据预处理器：设置归一化参数，建议与预训练时的 Normalize 保持一致
    data_preprocessor=dict(
        type='mmocr.models.textrecog.data_preprocessors.data_preprocessor.TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ), # CRNN 通常只需要最后一层的特征
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        # 【核心改进点】必须开启 Deep Stem 以适配你的 ODM 预训练权重
        deep_stem=True,
        avg_down=True,
        init_cfg=dict(
            type='Pretrained',
            # 指向你提取出的中文手写体预训练 Backbone
            checkpoint='E:/ODM/odm_chinese_handwriting_v1d.pth'),),
            
    
    # Encoder：将 CNN 特征图转换为序列特征（使用 channel reduction 以匹配 decoder 输入）
    encoder=dict(
        type='ChannelReductionEncoder',
        in_channels=2048,    # ResNet50 Stage4 的输出通道
        out_channels=256),
    
    # 解码头：CRNN Decoder（使用 CTC loss）
    decoder=dict(
        type='CRNNDecoder',
        in_channels=256,
        dictionary=dictionary,
        rnn_flag=True,
        module_loss=dict(type='CTCModuleLoss', zero_infinity=True),
        postprocessor=dict(type='CTCPostProcessor')),
)

# --- 数据流水线 (Pipeline) ---
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(type='LoadOCRAnnotations', with_text=True),
    # 针对手写行数据进行 Resize，保持长宽比或统一缩放到 (高32, 宽不限/固定)
    # 这里采用标准 CRNN 设置：高度 32，宽度按需缩放（最大 448）
    dict(
        type='RescaleToHeight',
        height=32,
        min_width=32,
        max_width=256,
        width_divisor=16),
    dict(type='PackTextRecogInputs', meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(
        type='RescaleToHeight',
        height=32,
        min_width=32,
        max_width=256,
        width_divisor=16),
    # 载入标注用于测试评估
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='PackTextRecogInputs', meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

# --- 数据加载器 ---
data_root = 'E:/ODM/data/casia/'

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RecogTextDataset',
        data_root=data_root,
        ann_file='textrecog_train.jsonl',
        data_prefix=dict(img_path='imgs/train/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RecogTextDataset',
        data_root=data_root,
        ann_file='textrecog_test.jsonl',
        data_prefix=dict(img_path='imgs/test/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# --- 评估器 ---
val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[dict(type='WordMetric', mode=['exact', 'ignore_case_symbol'])],
    dataset_prefixes=['casia'])

test_evaluator = val_evaluator

# --- 训练计划 ---
# 手写体识别通常需要比场景文本更多的 Epoch 才能收敛
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=2)

# 学习率配置：Adam 初始 1e-4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=0.0001))

# 日志与保存路径
work_dir = 'work_dirs/crnn_resnet50_odm_chinese_casia'

visualizer = dict(
    type='mmengine.Visualizer', # 使用最基础的，不依赖 mmocr 的特殊类
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')]
)