#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIOMEDICA混合训练 - 最终完整版
本地数据 + Hugging Face流式数据

特点:
1. 纯Python实现（不需要CLI命令）
2. 本地数据优先（您的10,653个样本）
3. 云端流式补充（从Hugging Face实时读取，不下载）
4. Token自动管理（环境变量或参数）

使用方法:
    export HF_TOKEN=hf_your_token_here
    python demo_hybrid_training.py --total-samples 50000 --epochs 10

必须在Terminal.app中运行！
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
import io

# 设置代理（必须！）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

print("[代理] 代理设置: http://127.0.0.1:7890")

# 检查依赖
print("[检查] 检查依赖...")

try:
    from huggingface_hub import hf_hub_url, login, HfApi, list_repo_files
    print("[OK] huggingface_hub")
except ImportError:
    print("[错误] 缺少 huggingface_hub")
    print("安装: pip install huggingface_hub")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    import torchvision.transforms as transforms
    print("[OK] torch")
except ImportError:
    print("[错误] 缺少 torch")
    print("安装: pip install torch torchvision")
    sys.exit(1)

try:
    import webdataset as wds
    print("[OK] webdataset")
except ImportError:
    print("[错误] 缺少 webdataset")
    print("安装: pip install webdataset")
    sys.exit(1)

try:
    from PIL import Image
    print("[OK] PIL")
except ImportError:
    print("[错误] 缺少 pillow")
    print("安装: pip install pillow")
    sys.exit(1)

try:
    from transformers import CLIPModel, CLIPProcessor
    print("[OK] transformers")
except ImportError:
    print("[错误] 缺少 transformers")
    print("安装: pip install transformers")
    sys.exit(1)

try:
    from tqdm import tqdm
    print("[OK] tqdm")
except ImportError:
    print("[警告]  建议安装 tqdm: pip install tqdm")
    # tqdm不是必须的，可以继续
    tqdm = lambda x, **kwargs: x

print("\n[OK] 所有依赖检查通过！\n")


class HybridDataLoader:
    """混合数据加载器 - 本地 + HF流式"""
    
    def __init__(self, local_path, hf_repo, total_samples, batch_size, hf_token):
        self.local_path = Path(local_path)
        self.hf_repo = hf_repo
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.hf_token = hf_token
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 收集本地tar文件
        print(f"[目录] 扫描本地tar文件: {local_path}")
        self.local_tar_files = self._collect_local_tar_files()
        print(f"[OK] 本地tar文件: {len(self.local_tar_files)} 个")
        
        # 估算样本数
        self.local_count = len(self.local_tar_files) * 5000
        print(f"   估计样本: ~{self.local_count:,}")
        
        self._print_plan()
    
    def _collect_local_tar_files(self):
        """收集本地tar文件"""
        tar_files = []
        if not self.local_path.exists():
            return tar_files
        
        for tar_file in self.local_path.glob("*.tar"):
            tar_files.append(str(tar_file))
        
        return sorted(tar_files)
    
    def _print_plan(self):
        """打印训练计划"""
        print("\n" + "=" * 70)
        print("  [数据] 混合训练计划")
        print("=" * 70)
        print(f"\n[训练] 目标: {self.total_samples:,} 个样本")
        
        if self.local_count >= self.total_samples:
            print(f"\n[OK] 本地数据充足")
            print(f"   使用: {self.total_samples:,} / {self.local_count:,} (100%本地)")
        else:
            cloud_needed = self.total_samples - self.local_count
            local_pct = (self.local_count / self.total_samples) * 100
            cloud_pct = 100 - local_pct
            
            print(f"\n[Epoch] 数据分配:")
            print(f"   [OK] 本地数据: {self.local_count:,} ({local_pct:.1f}%)")
            print(f"   [云端]  HF流式: {cloud_needed:,} ({cloud_pct:.1f}%)")
            print(f"\n[存储] 节省存储: ~{cloud_needed * 0.5 / 1024:.1f} GB")
            print(f"   (云端数据不下载，直接流式读取)")
        
        print("=" * 70 + "\n")
    
    def _create_hf_stream_manual(self):
        """手动streaming - 使用Python requests逐个读取tar"""
        print(f"[云端]  连接Hugging Face streaming...")
        
        try:
            import requests
            import tarfile
            
            # 获取tar文件列表
            files = list_repo_files(
                repo_id=self.hf_repo,
                repo_type="dataset",
                token=self.hf_token
            )
            tar_files = sorted([f for f in files if f.endswith('.tar')])
            print(f"   找到 {len(tar_files)} 个tar文件")
            
            # 只使用前3个
            max_shards = 3
            tar_files = tar_files[:max_shards]
            print(f"   [提示] 使用前 {len(tar_files)} 个文件\n")
            
            # 构建URL列表
            self.hf_tar_urls = []
            for f in tar_files:
                url = hf_hub_url(
                    repo_id=self.hf_repo,
                    filename=f,
                    repo_type="dataset"
                )
                self.hf_tar_urls.append(url)
            
            print(f"   [OK] 准备 {len(self.hf_tar_urls)} 个tar文件\n")
            return True
            
        except Exception as e:
            print(f"   [错误] 失败: {e}")
            return False
    
    def _stream_from_tar_url(self, url):
        """从单个tar URL streaming读取样本"""
        import requests
        import tarfile
        
        try:
            # 使用代理下载
            proxies = {
                'http': 'http://127.0.0.1:7890',
                'https': 'http://127.0.0.1:7890'
            }
            
            headers = {'Authorization': f'Bearer {self.hf_token}'}
            
            print(f"   [下载] Streaming: {url.split('/')[-1]}")
            
            response = requests.get(url, headers=headers, proxies=proxies, stream=True, timeout=60)
            response.raise_for_status()
            
            # 用tarfile读取streaming内容
            tar = tarfile.open(fileobj=response.raw, mode='r|*')
            
            for member in tar:
                if member.isfile():
                    # 只处理jpg文件
                    if member.name.endswith('.jpg'):
                        # 提取对应的txt和json
                        base_name = member.name[:-4]
                        
                        try:
                            # 读取jpg
                            jpg_file = tar.extractfile(member)
                            if jpg_file:
                                img = Image.open(jpg_file)
                                
                                # 查找txt（简化：假设在同一个tar中）
                                txt_content = ""
                                
                                yield {
                                    'image': img,
                                    'text': txt_content,
                                    'metadata': {}
                                }
                        except:
                            continue
            
            tar.close()
            
        except Exception as e:
            print(f"   [警告]  Tar streaming错误: {e}")
    
    def _create_local_stream(self):
        """创建本地tar文件流"""
        if not self.local_tar_files:
            return None
        
        print(f"[目录] 创建本地tar流...")
        print(f"   本地tar: {len(self.local_tar_files)} 个")
        
        try:
            # 使用官方tutorial的DataPipeline
            dataset = wds.DataPipeline(
                wds.SimpleShardList(self.local_tar_files),
                wds.tarfile_to_samples(),
                wds.decode("pil"),
                wds.to_tuple("jpg", "txt", "json"),
            )
            
            print(f"   [OK] 本地tar Pipeline创建完成\n")
            return dataset
            
        except Exception as e:
            print(f"   [错误] 本地tar流失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def __iter__(self):
        """迭代器 - 混合模式"""
        sample_count = 0
        local_used = 0
        cloud_used = 0
        
        # 阶段1: 本地tar数据
        print(f"[数据] 阶段1: 加载本地tar数据\n")
        
        local_stream = self._create_local_stream()
        
        if local_stream:
            for i, sample in enumerate(local_stream):
                if sample_count >= self.total_samples:
                    break
                
                try:
                    # tuple解包
                    img, caption, metadata = sample
                    
                    if not isinstance(img, Image.Image):
                        if isinstance(img, bytes):
                            img = Image.open(io.BytesIO(img))
                        else:
                            continue
                    
                    img = img.convert('RGB')
                    img_tensor = self.transform(img)
                    
                    if isinstance(caption, bytes):
                        caption = caption.decode('utf-8', errors='ignore')
                    
                    yield {
                        "image": img_tensor,
                        "text": str(caption),
                        "source": "local"
                    }
                    
                    sample_count += 1
                    local_used += 1
                    
                    if (i + 1) % 1000 == 0:
                        print(f"   本地tar: {local_used:,}")
                        
                except Exception as e:
                    if local_used < 5:
                        print(f"   [警告]  本地样本错误: {e}")
                    continue
        
        print(f"[OK] 本地tar数据: {local_used:,}\n")
        
        # 阶段2: HF流式数据
        if sample_count < self.total_samples:
            cloud_needed = self.total_samples - sample_count
            print(f"[数据] 阶段2: HF流式数据")
            print(f"   还需要: {cloud_needed:,}\n")
            
            # 准备HF tar URLs
            if self._create_hf_stream_manual():
                print(f"   [处理] 开始streaming读取...\n")
                
                # 逐个tar文件streaming
                for tar_url in self.hf_tar_urls:
                    if sample_count >= self.total_samples:
                        break
                    
                    # 从这个tar URL streaming读取
                    for sample in self._stream_from_tar_url(tar_url):
                        if sample_count >= self.total_samples:
                            break
                        
                        try:
                            img = sample['image']
                            
                            if not isinstance(img, Image.Image):
                                continue
                            
                            img = img.convert('RGB')
                            img_tensor = self.transform(img)
                            
                            yield {
                                "image": img_tensor,
                                "text": sample.get('text', ''),
                                "source": "cloud"
                            }
                            
                            sample_count += 1
                            cloud_used += 1
                            
                            if cloud_used % 100 == 0:
                                print(f"   云端: {cloud_used:,}/{cloud_needed:,}")
                        
                        except Exception as e:
                            if cloud_used < 5:
                                print(f"   [警告]  样本错误: {e}")
                            continue
                
                print(f"[OK] 云端streaming: {cloud_used:,}\n")
        
        print(f"[OK] 总计: {sample_count:,}")
        print(f"   本地: {local_used:,}")
        print(f"   云端: {cloud_used:,}\n")
    
    def get_batches(self):
        """批次迭代器"""
        batch = []
        
        for sample in self:
            batch.append(sample)
            
            if len(batch) >= self.batch_size:
                yield {
                    "image": torch.stack([s["image"] for s in batch]),
                    "text": [s["text"] for s in batch],
                    "source": [s["source"] for s in batch]
                }
                batch = []
        
        if batch:
            yield {
                "image": torch.stack([s["image"] for s in batch]),
                "text": [s["text"] for s in batch],
                "source": [s["source"] for s in batch]
            }


def train_hybrid(args, hf_token):
    """混合训练主函数 - 标准CLIP对比学习"""
    
    print("\n" + "=" * 70)
    print("  [开始] 开始混合训练（标准CLIP对比学习）")
    print("=" * 70)
    
    # 设置模型
    print("\n[数据] 设置CLIP模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        writer = SummaryWriter('./logs')
        print(f"[OK] 模型设置完成 (设备: {device})")
        print(f"   Vision Encoder: ViT-B/32")
        print(f"   Text Encoder: Transformer")
        print(f"   参数量: ~149M")
    except Exception as e:
        print(f"[错误] 模型设置失败: {e}")
        return False
    
    # 创建数据加载器
    print(f"\n[数据] 创建混合数据加载器...")
    try:
        dataloader = HybridDataLoader(
            local_path=args.local_path,
            hf_repo=args.hf_repo,
            total_samples=args.total_samples,
            batch_size=args.batch_size,
            hf_token=hf_token
        )
    except Exception as e:
        print(f"[错误] 数据加载器创建失败: {e}")
        return False
    
    # 训练循环
    print(f"\n[训练] 开始训练 ({args.epochs} epochs)")
    print("=" * 70 + "\n")
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"[Epoch] Epoch {epoch + 1}/{args.epochs}")
        
        model.train()
        epoch_loss = 0
        batch_count = 0
        epoch_local = 0
        epoch_cloud = 0
        
        try:
            for batch_idx, batch in enumerate(dataloader.get_batches()):
                # 统计来源
                for src in batch['source']:
                    if src == 'local':
                        epoch_local += 1
                    else:
                        epoch_cloud += 1
                
                # 准备数据
                images = batch['image']  # tensor已经处理好
                texts = batch['text']    # list of strings
                
                # 转换为PIL图像（processor需要）
                from torchvision.transforms import ToPILImage
                to_pil = ToPILImage()
                pil_images = [to_pil(img) for img in images]
                
                # 使用CLIP processor处理
                inputs = processor(
                    text=texts,
                    images=pil_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(device)
                
                # 前向传播 - 标准CLIP
                outputs = model(**inputs)
                
                # CLIP对比学习损失
                logits_per_image = outputs.logits_per_image  # [batch, batch]
                logits_per_text = outputs.logits_per_text    # [batch, batch]
                
                # 标签：对角线为正样本
                labels = torch.arange(len(pil_images)).to(device)
                
                # 双向对比损失
                loss_i2t = torch.nn.functional.cross_entropy(logits_per_image, labels)
                loss_t2i = torch.nn.functional.cross_entropy(logits_per_text, labels)
                
                # 总损失
                loss = (loss_i2t + loss_t2i) / 2
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # 进度显示
                if batch_idx % 10 == 0:
                    avg_loss = epoch_loss / batch_count
                    print(f"   Batch {batch_idx}: Loss={loss.item():.4f}, "
                          f"I2T={loss_i2t.item():.4f}, T2I={loss_t2i.item():.4f}, "
                          f"Local={epoch_local}, Cloud={epoch_cloud}")
                
                # 不保存中间检查点（节省空间）
                # 只在每个epoch结束时保存最佳模型
        
        except StopIteration:
            pass
        except Exception as e:
            print(f"[警告]  Epoch训练出错: {e}")
            continue
        
        # Epoch总结
        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"\n[数据] Epoch {epoch+1} 完成:")
        print(f"   损失: {avg_loss:.4f}")
        print(f"   本地: {epoch_local:,}")
        print(f"   云端: {epoch_cloud:,}\n")
        
        # TensorBoard记录
        writer.add_scalar('Loss/Total', avg_loss, epoch)
        writer.add_scalar('Samples/Local', epoch_local, epoch)
        writer.add_scalar('Samples/Cloud', epoch_cloud, epoch)
        writer.add_scalar('Samples/Total', epoch_local + epoch_cloud, epoch)
        
        # 保存最佳模型（只保留最好的一个）
        if avg_loss < best_loss:
            best_loss = avg_loss
            Path("./models").mkdir(exist_ok=True)
            
            # 删除旧模型
            import glob
            for old_model in glob.glob("./models/best_model_*.pt"):
                try:
                    os.remove(old_model)
                except:
                    pass
            
            # 保存新的最佳模型
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }, f"./models/best_model_e{epoch}.pt")
            print(f"   [保存] 保存最佳模型 (Loss: {avg_loss:.4f})\n")
    
    writer.close()
    
    print("=" * 70)
    print("  [完成] 训练完成！")
    print("=" * 70)
    print(f"[最佳] 最佳损失: {best_loss:.4f}")
    print(f"[目录] 模型: ./models/")
    print(f"[Epoch] 日志: tensorboard --logdir=./logs")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='BIOMEDICA混合训练')
    parser.add_argument('--total-samples', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--local-path', type=str, default='./dataset')
    parser.add_argument('--hf-repo', type=str, default='BIOMEDICA/biomedica_webdataset_24M')
    parser.add_argument('--hf-token', type=str, default=None)
    
    args = parser.parse_args()
    
    # 获取token
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    
    if not hf_token:
        print("\n[错误] 未找到Hugging Face token")
        print("\n使用方法:")
        print("  export HF_TOKEN=hf_your_token")
        print("  python 完整混合训练_最终版.py")
        print("\n或:")
        print("  python 完整混合训练_最终版.py --hf-token hf_your_token")
        sys.exit(1)
    
    # 登录验证
    print(f"\n🔐 登录Hugging Face...")
    print(f"Token: {hf_token[:15]}...{hf_token[-10:]}")
    
    try:
        login(token=hf_token, add_to_git_credential=False)
        
        api = HfApi()
        user = api.whoami(token=hf_token)
        print(f"[OK] 登录成功: {user.get('name', 'User')}")
        
        # 验证数据集访问
        print(f"\n[检查] 验证数据集访问...")
        files = list_repo_files(args.hf_repo, repo_type="dataset", token=hf_token)
        print(f"[OK] 可访问: {args.hf_repo}")
        print(f"   文件数: {len(files)}")
        
    except Exception as e:
        print(f"[错误] 登录/验证失败: {e}")
        print(f"\n请确保:")
        print(f"1. Token有效")
        print(f"2. 已同意条款: https://huggingface.co/datasets/{args.hf_repo}")
        print(f"3. 在Terminal.app运行（不是Cursor）")
        sys.exit(1)
    
    # 开始训练
    success = train_hybrid(args, hf_token)
    
    if success:
        print("\n[OK] 成功完成！")
        sys.exit(0)
    else:
        print("\n[错误] 训练失败")
        sys.exit(1)


if __name__ == "__main__":
    print("\n" + "!" * 70)
    print("  [警告]  重要提示")
    print("!" * 70)
    print("\n必须在Terminal.app中运行（不是Cursor终端）！")
    print("\n必须先:")
    print("  1. 同意条款: https://huggingface.co/datasets/BIOMEDICA/biomedica_webdataset_24M")
    print("  2. 设置Token: export HF_TOKEN=hf_your_token_here")
    print("\n" + "!" * 70 + "\n")
    
    main()

