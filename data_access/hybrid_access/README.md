# BIOMEDICA 混合训练工具

本目录包含 BIOMEDICA 数据集的混合访问训练脚本，支持本地数据与云端流式数据的无缝结合。

## 📋 功能特点

### 核心功能
- ✅ **混合数据加载**: 本地数据优先 + Hugging Face 云端流式补充
- ✅ **零下载训练**: 云端数据通过流式方式读取，无需完整下载
- ✅ **自动资源管理**: 智能切换本地和云端数据源
- ✅ **CLIP 模型训练**: 内置 CLIP 视觉-语言模型训练
- ✅ **TensorBoard 日志**: 自动记录训练指标

### 优势
1. **节省存储空间**: 云端数据不下载，节省 TB 级空间
2. **高效训练**: 本地快速数据 + 云端流式补充，平衡速度与成本
3. **灵活配置**: 支持自定义训练样本数、批次大小等参数
4. **自动管理**: Token 自动验证，数据源自动切换

## 📁 文件说明

- **demo_hybrid_training.py**: 混合训练主脚本，包含完整的数据加载和训练逻辑

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision
pip install transformers
pip install huggingface_hub
pip install webdataset
pip install pillow tqdm tensorboard
```

### 2. 配置 HuggingFace Token

```bash
# 方法1: 环境变量
export HF_TOKEN=hf_your_token_here

# 方法2: 命令行参数
python demo_hybrid_training.py --hf-token hf_your_token
```

### 3. 同意数据集使用条款

访问并同意 BIOMEDICA 数据集的使用条款:
https://huggingface.co/datasets/BIOMEDICA/biomedica_webdataset_24M

### 4. 运行训练

```bash
# 基础训练（默认配置）
python demo_hybrid_training.py

# 自定义训练
python demo_hybrid_training.py \
    --total-samples 100000 \
    --epochs 10 \
    --batch-size 256 \
    --local-path ./dataset
```

## 🔧 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-samples` | 50000 | 总训练样本数 |
| `--epochs` | 10 | 训练轮数 |
| `--batch-size` | 256 | 批次大小 |
| `--local-path` | `./dataset` | 本地数据目录 |
| `--hf-repo` | `BIOMEDICA/biomedica_webdataset_24M` | HuggingFace 仓库 |
| `--hf-token` | None | HuggingFace Token |

## 📊 工作原理

### 数据流

```
开始训练
    ↓
加载本地数据 (tar文件)
    ↓
本地数据不足？
    ├─ 是 → 连接 HuggingFace 云端流式读取
    │         ↓
    │      自动混合数据
    └─ 否 → 仅使用本地数据
            ↓
        训练 CLIP 模型
            ↓
        保存最佳模型
```

### 数据分配示例

假设：
- 本地数据: 10,653 个样本
- 目标训练: 50,000 个样本

分配结果：
- 本地数据: 10,653 (21.3%)
- 云端流式: 39,347 (78.7%)
- **节省存储**: ~19 GB

## 📖 使用示例

### 示例 1: 小规模训练（本地为主）

```bash
python demo_hybrid_training.py \
    --total-samples 15000 \
    --epochs 5
```

### 示例 2: 大规模训练（大量云端数据）

```bash
python demo_hybrid_training.py \
    --total-samples 100000 \
    --epochs 20 \
    --batch-size 512
```

### 示例 3: 仅使用本地数据

如果本地数据充足，脚本会自动优先使用本地数据：

```bash
# 如果本地有足够数据，自动100%使用本地
python demo_hybrid_training.py
```

## 📈 监控训练

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir=./logs

# 访问: http://localhost:6006
```

### 日志输出

训练过程会实时显示：
- 每个 batch 的损失值
- 本地/云端数据使用统计
- Epoch 总结信息

## 📁 输出文件

### 模型检查点

```
models/
└── best_model_e[N].pt  # 最佳模型（按损失值）
```

### TensorBoard 日志

```
logs/
└── run_YYYY-MM-DD_HH-MM-SS/
    └── events.out.tfevents.*
```

## ⚙️ 配置说明

### 代理设置

如果在中国大陆，需要配置代理（脚本已内置）：

```python
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
```

### 本地数据格式

本地数据应为 tar 文件：

```
./dataset/
├── 00000.tar
├── 00001.tar
├── 00002.tar
└── ...
```

## 🔍 故障排除

### 问题 1: Token 验证失败

**错误**: `401 Unauthorized`

**解决方案**:
1. 检查 Token 是否正确: `echo $HF_TOKEN`
2. 确保已同意数据集使用条款
3. 重新获取 Token: https://huggingface.co/settings/tokens

### 问题 2: 连接超时

**错误**: `Connection timeout` 或 `Read timeout`

**解决方案**:
1. 检查网络连接
2. 确认代理设置正确
3. 尝试使用 HuggingFace 镜像站

### 问题 3: 本地数据未找到

**警告**: `本地tar文件: 0 个`

**解决方案**:
- 检查 `--local-path` 路径是否正确
- 确保 tar 文件存在
- 脚本会自动切换为纯云端流式模式

## 📝 注意事项

1. **在 Terminal 运行**: 确保在 Terminal.app 中运行，而不是 IDE 终端
2. **数据目录**: 本地数据应包含 `.tar` 文件
3. **GPU 推荐**: 建议使用 GPU 训练以提高速度
4. **存储空间**: 云端流式数据不占用本地存储
5. **网络稳定**: 云端模式需要稳定的网络连接

## 📚 相关资源

- [BIOMEDICA 数据集](https://huggingface.co/datasets/BIOMEDICA/biomedica_webdataset_24M)
- [HuggingFace Hub 文档](https://huggingface.co/docs/hub)
- [CLIP 模型](https://github.com/openai/CLIP)
- [项目主页](../README.md)

## 📧 获取帮助

遇到问题？
- 查看 [故障排除](#故障排除) 部分
- 检查 [Issues](https://github.com/77YQ77/RGC-SRF/issues)
- 联系项目维护者

---

**最后更新**: 2025-10
