#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实Hugging Face数据下载器
必须在系统Terminal中运行，不能在Cursor中运行

使用方法:
    # 下载10个文件测试
    python download_real_hf_data.py --max-files 10
    
    # 下载50个文件
    python download_real_hf_data.py --max-files 50
    
    # 下载全部
    python download_real_hf_data.py
"""

from huggingface_hub import hf_hub_download, list_repo_files, HfApi
import os
import sys
from pathlib import Path
import argparse


def test_connection():
    """测试Hugging Face连接"""
    print("\n🔍 测试Hugging Face连接...")
    try:
        api = HfApi()
        api.whoami(token=None)
        print("✅ 连接成功\n")
        return True
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("\n🔧 解决方案:")
        print("1. 确保在系统Terminal运行（不是Cursor）")
        print("2. 配置代理: export HTTPS_PROXY=http://proxy:port")
        print("3. 使用镜像: export HF_ENDPOINT=https://hf-mirror.com")
        print("4. 登录: huggingface-cli login")
        return False


def download_biomedica_data(
    subset="commercial",
    max_files=None,
    output_dir="./biomedica_real_data"
):
    """
    下载真实的BioMedICA数据
    
    Args:
        subset: 数据集子集 (commercial, noncommercial, other)
        max_files: 最多下载文件数 (None=全部)
        output_dir: 输出目录
    """
    
    repo_id = f"BIOMEDICA/biomedica_webdataset_{subset}"
    
    print("=" * 70)
    print("  🚀 真实Hugging Face数据下载器")
    print("=" * 70)
    print(f"\n📦 仓库: {repo_id}")
    print(f"📁 目标: {output_dir}")
    print(f"📊 限制: {max_files if max_files else '全部'} 个文件")
    print("\n" + "-" * 70)
    
    try:
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 获取文件列表
        print("\n📋 获取文件列表...")
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        tar_files = sorted([f for f in files if f.endswith('.tar')])
        
        if not tar_files:
            print(f"❌ 未找到tar文件在仓库 {repo_id}")
            return []
        
        print(f"✅ 找到 {len(tar_files)} 个tar文件")
        
        # 确定下载数量
        download_count = len(tar_files) if max_files is None else min(max_files, len(tar_files))
        
        print(f"\n📥 开始下载 {download_count} 个文件...")
        print("   (支持断点续传，可以随时中断)\n")
        
        # 下载文件
        downloaded = []
        failed = []
        
        for i, tar_file in enumerate(tar_files[:download_count]):
            print(f"[{i+1}/{download_count}] {tar_file}")
            
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=tar_file,
                    repo_type="dataset",
                    local_dir=output_dir,
                    resume_download=True
                )
                
                # 检查文件大小
                size_mb = os.path.getsize(local_path) / (1024 * 1024)
                print(f"    ✅ 完成 ({size_mb:.1f} MB)")
                print(f"    📁 {local_path}\n")
                
                downloaded.append(local_path)
                
            except KeyboardInterrupt:
                print("\n⚠️  用户中断下载")
                print(f"已下载: {len(downloaded)} 个文件")
                print("可以重新运行命令继续下载（支持断点续传）")
                break
                
            except Exception as e:
                print(f"    ⚠️  失败: {e}\n")
                failed.append(tar_file)
                continue
        
        # 显示结果
        print("\n" + "=" * 70)
        print("  📊 下载完成")
        print("=" * 70)
        print(f"✅ 成功: {len(downloaded)} 个文件")
        if failed:
            print(f"⚠️  失败: {len(failed)} 个文件")
        
        print(f"\n📁 数据位置: {os.path.abspath(output_dir)}")
        
        # 统计总大小
        total_size = sum(os.path.getsize(f) for f in downloaded)
        total_gb = total_size / (1024 ** 3)
        print(f"💾 总大小: {total_gb:.2f} GB")
        
        print("\n" + "=" * 70)
        
        if downloaded:
            print("\n✅ 真实数据下载成功！不是模拟数据！")
            print("\n🚀 下一步:")
            print(f"\npython improved_cloud_training.py \\")
            print(f"    --local-path {output_dir} \\")
            print(f"    --epochs 10 \\")
            print(f"    --batch-size 32")
        
        return downloaded
        
    except Exception as e:
        print(f"\n❌ 下载过程出错: {e}")
        print("\n🔧 排查步骤:")
        print("1. 确认在系统Terminal运行: which python")
        print("2. 测试网络: curl -I https://huggingface.co")
        print("3. 查看错误信息并配置网络")
        return []


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='下载真实的BioMedICA数据（必须在系统Terminal运行）'
    )
    parser.add_argument(
        '--subset', 
        default='commercial',
        choices=['commercial', 'noncommercial', 'other'],
        help='数据集子集'
    )
    parser.add_argument(
        '--max-files', 
        type=int, 
        default=10,
        help='最多下载文件数 (默认:10, 0=全部)'
    )
    parser.add_argument(
        '--output-dir', 
        default='./biomedica_real_data',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 如果max_files=0，下载全部
    max_files = None if args.max_files == 0 else args.max_files
    
    # 警告信息
    print("\n" + "!" * 70)
    print("  ⚠️  重要提示")
    print("!" * 70)
    print("\n此脚本必须在系统Terminal中运行，不能在Cursor中运行！")
    print("\n如何运行:")
    print("  1. 打开Terminal.app (Command+Space → terminal)")
    print("  2. cd /Users/yaohu/Desktop/advanced_vlm_training")
    print("  3. conda activate dataset_dwonload")
    print("  4. python download_real_hf_data.py --max-files 10")
    print("\n" + "!" * 70)
    
    # 测试连接
    if not test_connection():
        print("\n❌ 无法继续，请先解决网络连接问题")
        sys.exit(1)
    
    # 确认下载
    if max_files and max_files > 20:
        print(f"\n⚠️  准备下载 {max_files} 个文件，可能需要较长时间")
        response = input("确认继续? (y/N): ")
        if response.lower() != 'y':
            print("取消下载")
            sys.exit(0)
    
    # 执行下载
    files = download_biomedica_data(
        subset=args.subset,
        max_files=max_files,
        output_dir=args.output_dir
    )
    
    if files:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()



