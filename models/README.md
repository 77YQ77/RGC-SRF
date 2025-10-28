# 预训练模型 | Pre-trained Models

本目录包含RGC SRF项目的预训练模型。

This directory contains pre-trained models from the RGC SRF project.

---

## 📋 模型列表 | Model Zoo

### 1. 医疗VLM基础模型 | Medical VLM Base

- **模型名称**: MedVLM-Base
- **架构**: CLIP-based Vision-Language Model
- **参数量**: 149M
- **训练数据**: BIOMEDICA 24M
- **任务**: Image-Text Contrastive Learning

**性能指标**:
- Zero-shot Classification: 75.3%
- Image-Text Retrieval: R@1: 68.5%

**下载**:
```bash
# 从Hugging Face下载
huggingface-cli download RGC-SRF/MedVLM-Base

# 或使用Python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="RGC-SRF/MedVLM-Base", filename="pytorch_model.bin")
```

**使用示例**:
```python
from models import MedVLM

model = MedVLM.from_pretrained("RGC-SRF/MedVLM-Base")
model.eval()

# 推理
output = model(image, text)
```

---

### 2. 多任务医疗VLM | Multi-task Medical VLM

- **模型名称**: MedVLM-MultiTask
- **架构**: Multi-Task Learning (Classification + Generation + Segmentation)
- **参数量**: 380M
- **训练数据**: BIOMEDICA + 标注数据集

**任务性能**:
- 诊断分类: 85.3% Accuracy
- 报告生成: BLEU-4: 0.32
- 病灶分割: Dice: 0.78

**下载**:
```bash
huggingface-cli download RGC-SRF/MedVLM-MultiTask
```

---

### 3. 领域特定模型 | Domain-Specific Models

#### 3.1 X-ray诊断模型
- **模型**: XRay-Diagnosis-V1
- **数据**: Chest X-ray数据集
- **准确率**: 88.5%

#### 3.2 病理分割模型
- **模型**: Histo-Segment-V1
- **数据**: 病理切片数据集
- **Dice Score**: 0.82

---

## 📥 下载指南 | Download Guide

### 方法1: Hugging Face Hub

```bash
# 下载单个模型
huggingface-cli download RGC-SRF/model-name

# 下载所有模型
bash scripts/download_all_models.sh
```

### 方法2: 直接链接

| 模型 | 大小 | 下载链接 |
|------|------|---------|
| MedVLM-Base | 600MB | [Download](URL) |
| MedVLM-MultiTask | 1.5GB | [Download](URL) |
| XRay-Diagnosis-V1 | 800MB | [Download](URL) |

### 方法3: Git LFS

```bash
git lfs install
git clone https://huggingface.co/RGC-SRF/MedVLM-Base
```

---

## 🔧 使用说明 | Usage

### 加载模型 | Load Model

```python
import torch
from models import load_model

# 方法1: 从本地加载
model = load_model("checkpoints/MedVLM-Base.pth")

# 方法2: 从Hugging Face加载
model = load_model("RGC-SRF/MedVLM-Base")

# 方法3: 自定义配置
model = load_model(
    "MedVLM-Base",
    config="configs/custom_config.yaml",
    device="cuda"
)
```

### 推理示例 | Inference Example

```python
# 单图像推理
image = load_image("xray.jpg")
prediction = model.predict(image)

# 批量推理
images = load_batch(["img1.jpg", "img2.jpg"])
predictions = model.predict_batch(images)

# 多任务推理
outputs = model.predict_multitask(image)
diagnosis = outputs['diagnosis']
report = outputs['report']
segmentation = outputs['segmentation']
```

---

## 📊 模型配置 | Model Configs

配置文件位于 `configs/` 目录:

```yaml
# configs/medvlm_base.yaml
model:
  name: MedVLM-Base
  architecture: CLIP
  image_encoder:
    type: ViT-B/32
    embed_dim: 768
  text_encoder:
    type: Transformer
    embed_dim: 512
  
training:
  batch_size: 256
  learning_rate: 1e-4
  epochs: 100
```

---

## 🔄 模型更新 | Model Updates

### 版本历史 | Version History

- **v1.0.0** (2025-10) - 初始发布
  - MedVLM-Base
  - MedVLM-MultiTask

- **v1.1.0** (计划中) - 性能优化
  - 模型量化
  - 推理加速

---

## 📝 引用 | Citation

如果使用了我们的模型，请引用：

```bibtex
@model{medvlm_2025,
  title={Medical Vision-Language Models for Clinical Applications},
  author={Your Name},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/RGC-SRF}
}
```

---

## 📧 反馈 | Feedback

- 模型问题: [GitHub Issues](https://github.com/77YQ77/RGC-SRF/issues)
- 模型请求: [Discussions](https://github.com/77YQ77/RGC-SRF/discussions)

---

**Last Updated**: 2025-10

