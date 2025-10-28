# RGC SRF Research Repository

**Research Grant Council (RGC) Senior Research Fellowship Project**

研究资助局 (RGC) 高级研究奖学金项目

---

## 🎯 项目目标 | Project Goal

本项目旨在开发一个用于医疗领域的 **Vision-Language Model (VLM)**，能够理解医疗影像并生成相应的诊断描述，推动人工智能在医疗诊断中的应用。

This project aims to develop a **Vision-Language Model (VLM)** for medical applications that can understand medical images and generate corresponding diagnostic descriptions, advancing AI applications in medical diagnosis.

## 📦 仓库内容 | Repository Contents

本仓库用于共享 RGC SRF 项目的研究成果，包括：

This repository shares research outputs from the RGC SRF project, including:

- 📄 **论文** (Papers) - 学术论文、预印本和演讲材料
- 💻 **代码** (Code) - 研究代码、训练和推理脚本
- 📊 **数据** (Data) - BIOMEDICA 数据集访问工具
- 🧠 **模型** (Models) - 预训练的医疗 VLM 模型
- 📚 **文档** (Docs) - 使用指南和 API 文档

## 📁 目录结构 | Directory Structure

```
RGC SRF/
├── papers/           # 📄 论文和出版物
│   ├── VLM/         # 视觉-语言模型相关
│   ├── LLM/         # 大语言模型相关
│   └── Dataset/     # 数据集相关
│
├── code/             # 💻 研究代码
│   ├── training/    # 训练脚本
│   ├── inference/   # 推理代码
│   └── evaluation/  # 评估工具
│
├── models/           # 🧠 预训练模型
│   ├── checkpoints/ # 模型检查点
│   └── configs/     # 模型配置
│
├── datasets/         # 📊 数据集
│   ├── metadata/    # 数据集元数据
│   └── splits/      # 数据划分
│
├── data_access/      # 🔐 数据获取工具
│   └── hybrid_access/  # BIOMEDICA 混合访问
│
└── docs/             # 📚 文档
    ├── tutorials/   # 使用教程
    ├── api/         # API 文档
    └── guides/      # 指南
```

## 🚀 快速开始 | Quick Start

### 1. 环境设置

```bash
# 克隆仓库
git clone https://github.com/77YQ77/RGC-SRF.git
cd RGC-SRF

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据访问

```bash
cd data_access/hybrid_access
python demo_hybrid_training.py --help
```

详细说明见 [data_access/hybrid_access/README.md](data_access/hybrid_access/README.md)

### 3. 模型训练

```bash
cd code/training
python train.py --config configs/vlm_config.yaml
```

## 📚 文档 | Documentation

- [数据获取指南](data_access/hybrid_access/README.md)
- [数据集说明](datasets/README.md)
- [模型说明](models/README.md)
- [论文列表](papers/README.md)

## 📧 联系方式 | Contact

- **GitHub**: [@77YQ77](https://github.com/77YQ77)
- **Issues**: [提交问题](https://github.com/77YQ77/RGC-SRF/issues)

## 📝 引用 | Citation

```bibtex
@misc{rgc_srf_2025,
  title={RGC SRF Research Project},
  author={RGC SRF Team},
  year={2025},
  publisher={GitHub},
  url={https://github.com/77YQ77/RGC-SRF}
}
```

## 📜 许可证 | License

[MIT License](LICENSE)

---

**Made with ❤️ for Medical AI Research Community**

