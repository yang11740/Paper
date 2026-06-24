# ODM & Improved Chinese-HTR 复现与改进项目指南

本项目包含两个核心部分：
1. **原论文复现**：复现 CVPR 2024 论文《ODM: A Text-Image Further Alignment Pre-training Approach for Scene Text Detection and Spotting》。
2. **改进应用实验**：将 ODM 架构应用于**中文手写文本行（HTR）**领域，利用古典文学语料（《红楼梦》）进行预训练改进，并在 CASIA-HWDB 数据集上进行验证。

---

## 环境准备

### 基础环境
*   **OS**: Ubuntu 22.04 (推荐) 或 Windows 11
*   **GPU**: NVIDIA RTX 3060 (6GB) 或 Tesla V100 (16GB)
*   **Python**: 3.8+
*   **PyTorch**: 1.10.2+
*   **CUDA**: 11.3+

### 安装步骤
```bash
# 1. 克隆代码库
git clone https://github.com/PriNing/ODM.git
cd ODM

# 2. 安装 MMOCR 框架 (用于下游微调)
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
pip install -e .

# 3. 安装原生代码依赖 (用于预训练)
pip install lpips pandas opencv-python
```

---

## 第一阶段：原论文 ODM 复现 (场景文本检测)

### A. 下载预训练权重
从论文官方渠道下载 `epoch_100.pt` (SynthText 预训练权重)。

### B. 权重转换 (Native -> MMOCR)
运行转换脚本，将原生代码权重对齐至 MMOCR 框架：
```bash
python tools/convert2mmocr.py --src epoch_100.pt --dst odm_mmocr.pth
```

### C. 下游微调 (以 DBNet++ 为例)
在 `mmocr` 目录下运行，复现 ICDAR 2015 检测指标：
```bash
python tools/train.py configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_odm_icdar2015.py
```

---

## 第二阶段：改进应用实验 (中文手写体识别)

### A. 数据合成
1.  **语料处理**：运行 `extract_casia_dict.py` 提取 “红楼梦.txt” 中的去重汉字，生成 `chinese_dict.txt`。
2.  **数据合成**：使用手写字体库渲染语料，生成 5 万张图片对。
    *   **Input**: 手写样式图。
    *   **GT**: 标准宋体二值掩码图。
3.  **CSV 生成**：运行脚本生成 `pretrain_chinese.csv`，包含 `filepath`, `title`, `rotate_box` 字段。

### B. 中文 ODM 预训练
**核心改进**：修复了原代码中 LPIPS 的路径硬编码 Bug，激活了 CRAFT 引导的感知损失。
1.  修改 `RN50_Seg_Clip.json`，将 `vocab_size` 设为 **9265**。
2.  启动预训练：
```bash
python main.py \
    --train-data data/pretrain_chinese.csv \
    --char-dict-pth data/chinese_handwriting_char_dict.pkl \
    --use-LPIPS --use-OCR-LPIPS \
    --model RN50 --epochs 300 --name odm_chinese_pretrain
```

### C. 结构化权重提取
针对 **ResNet-V1d (Deep Stem)** 架构进行精准重映射：
```bash
python extract_backbone.py --src work_dirs/odm_chinese_pretrain/epoch_30.pt --dst odm_chinese_v1d.pth
```

### D. CASIA-HWDB 2.2 微调
1.  **数据解析**：运行 `convert_dgrl.py` 将原始 `.dgrl` 文件解析为行图片及 `.jsonl` 标注。
2.  **启动对比实验**：
```bash
# 实验 C: 改进后的中文 ODM (本项目)
python tools/train.py configs/textrecog/crnn/crnn_resnet50_odm_casia.py

# 实验 A/B: Baseline 与 原版 ODM 对照组
python tools/train.py configs/textrecog/crnn/crnn_resnet50_baseline_casia.py
```

---

## 结论
通过将 ODM 的“去样式化”思想与中文古典文学语料相结合，并修复关键的感知损失函数，我们成功地将轻量级（25M）模型的手写识别能力提升到了准生产水平，其在特定领域的表现优于参数量大千倍的通用多模态大模型（Qwen2-VL）。

---

### 注意事项
*   **显存管理**：本地 3060 运行微调时建议 `batch_size` 设为 16 或更小。
*   **虚拟内存**：Windows 用户如遇 `WinError 1455`，请将虚拟内存手动调至 40GB+。
*   **路径**：所有配置文件中的路径需根据实际存放位置修改。
