# 数据集说明 | Dataset Documentation

本目录包含项目使用的数据集信息和元数据。

This directory contains information and metadata about datasets used in the project.

---

## 📊 数据集概览 | Dataset Overview

### 主要数据集 | Primary Datasets

1. **BIOMEDICA**
   - 规模: 24M 图像-文本对
   - 来源: PubMed Central
   - 模态: X-ray, CT, MRI, Histopathology
   - 许可: 分Commercial和Non-commercial

2. **自建数据集**
   - *(在此添加您的数据集)*

---

## 📁 数据集结构 | Dataset Structure

```
datasets/
├── metadata/              # 元数据文件
│   ├── biomedica_stats.json
│   └── dataset_info.yaml
│
├── splits/               # 数据划分
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
│
└── annotations/          # 标注文件
    ├── diagnoses.json
    └── segmentations/
```

---

## 🔐 数据访问 | Data Access

### BIOMEDICA数据集

**访问方式1: Hybrid Access（推荐）**
```bash
cd ../data_access/hybrid_access
python download_biomedica.py
```

**访问方式2: 直接下载**
```bash
huggingface-cli download BIOMEDICA/biomedica_commercial_webdataset
```

**访问方式3: 流式访问**
```python
from datasets import load_dataset

dataset = load_dataset(
    "BIOMEDICA/biomedica_commercial_webdataset",
    streaming=True
)
```


