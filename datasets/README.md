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

---

## 📋 数据统计 | Statistics

### BIOMEDICA统计

| 子集 | 样本数 | 大小 | 用途 |
|------|--------|------|------|
| Commercial | 8M | 9TB | 可商用 |
| Non-commercial | 16M | 18TB | 仅研究 |
| Total | 24M | 27TB | - |

### 模态分布

| 模态 | 数量 | 占比 |
|------|------|------|
| X-ray | 8.5M | 35% |
| CT | 6.2M | 26% |
| MRI | 4.8M | 20% |
| Histopathology | 4.5M | 19% |

---

## 🔨 数据处理 | Data Processing

### 预处理脚本

```bash
# 数据清洗
python scripts/clean_data.py --input raw/ --output processed/

# 数据增强
python scripts/augment_data.py --config augment_config.yaml

# 生成统计信息
python scripts/generate_stats.py --dataset biomedica
```

### 数据格式转换

```bash
# WebDataset转Parquet
python scripts/convert_webdataset_to_parquet.py

# Parquet转JSON
python scripts/convert_parquet_to_json.py
```

---

## 📝 数据集引用 | Dataset Citation

### BIOMEDICA

```bibtex
@article{biomedica2024,
  title={BIOMEDICA: A Large-Scale Biomedical Image-Text Dataset},
  author={...},
  journal={...},
  year={2024}
}
```

---

## 🔒 数据使用协议 | Data Usage Agreement

### 许可证

- **Commercial子集**: 可用于商业目的
- **Non-commercial子集**: 仅限学术研究

### 使用条款

1. 引用原始数据集论文
2. 不得重新分发原始数据
3. 遵守数据隐私保护规定

---

## 📞 数据相关问题 | Data Issues

遇到数据问题？
- 检查 [数据访问指南](../data_access/README.md)
- 提交 [Issue](https://github.com/77YQ77/RGC-SRF/issues)
- 查看 [FAQ](../docs/faq.md)

---

**Last Updated**: 2025-10

