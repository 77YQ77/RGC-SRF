# 数据获取工具 | Data Access Tools

本目录包含用于获取研究数据集的工具和脚本。

This directory contains tools and scripts for accessing research datasets.

---

## 📋 目录 | Contents

- [Hybrid Access方式](#hybrid-access方式)
- [数据集列表](#数据集列表)
- [使用指南](#使用指南)
- [故障排除](#故障排除)

---

## 🔐 Hybrid Access方式

Hybrid Access结合了本地缓存和云端流式访问，适合处理大规模数据集。

### 特点 | Features

- ✅ **混合模式**: 本地缓存 + 云端流式
- ✅ **节省空间**: 无需下载完整数据集
- ✅ **高效训练**: 智能缓存常用数据
- ✅ **断点续传**: 网络中断自动恢复

### 快速开始 | Quick Start

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置HuggingFace Token
export HUGGING_FACE_TOKEN="your_token_here"

# 或使用命令行登录
huggingface-cli login

# 3. 运行hybrid access脚本
cd hybrid_access
python download_with_hybrid_access.py --config config.json
```

---

## 📊 数据集列表 | Available Datasets

### 1. BIOMEDICA数据集

- **规模**: 24M图像-文本对
- **大小**: 27TB
- **模态**: X-ray, CT, MRI, Histopathology
- **访问**: Hybrid Access推荐

```bash
python hybrid_access/download_biomedica.py \
    --subset commercial \
    --local-ratio 0.3 \
    --cloud-ratio 0.7
```

### 2. 其他数据集

*(在此添加其他数据集)*

---

## 📖 使用指南 | Usage Guide

### 配置文件 | Configuration

编辑 `hybrid_access/config.json`:

```json
{
  "dataset_name": "BIOMEDICA",
  "repo_id": "BIOMEDICA/biomedica_commercial_webdataset",
  "local_cache_dir": "./cache",
  "local_ratio": 0.3,
  "cloud_ratio": 0.7,
  "batch_size": 32,
  "num_workers": 4
}
```

### 基本用法 | Basic Usage

```python
from hybrid_access import HybridDataLoader

# 创建hybrid数据加载器
loader = HybridDataLoader(
    repo_id="BIOMEDICA/biomedica_commercial_webdataset",
    local_ratio=0.3,
    cloud_ratio=0.7
)

# 训练循环
for batch in loader:
    images, texts = batch
    # 训练代码...
```

### 高级用法 | Advanced Usage

```python
# 自定义缓存策略
loader = HybridDataLoader(
    repo_id="BIOMEDICA/biomedica_commercial_webdataset",
    cache_strategy="lru",  # 最近最少使用
    max_cache_size="10GB",
    prefetch=True
)
```

---

## 🔧 故障排除 | Troubleshooting

### 问题1: Token认证失败

**症状**: `401 Unauthorized`

**解决**:
```bash
# 重新登录
huggingface-cli logout
huggingface-cli login

# 或设置环境变量
export HUGGING_FACE_TOKEN="your_new_token"
```

### 问题2: 下载速度慢

**解决**:
```bash
# 使用代理（如在中国大陆）
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"

# 或使用镜像
export HF_ENDPOINT="https://hf-mirror.com"
```

### 问题3: 磁盘空间不足

**解决**:
- 减小`local_ratio`参数
- 清理缓存: `python cleanup_cache.py`
- 使用纯流式模式: `--local-ratio 0 --cloud-ratio 1`

---

## 📧 支持 | Support

遇到问题？
- 查看 [FAQ](../docs/faq.md)
- 提交 [Issue](https://github.com/77YQ77/RGC-SRF/issues)
- 联系项目维护者

---

**Updated**: 2025-10

