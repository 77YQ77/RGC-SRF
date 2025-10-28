#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版BioMedICA数据集下载器
仿照biomedica-etl项目，支持完整的ETL流程

功能特性:
- 支持WebDataset和Parquet两种格式
- 并行下载和序列化
- 数据预处理和过滤
- 支持Hugging Face Hub和直接下载
- 完整的日志记录和错误处理

使用方法:
    python enhanced_biomedica_downloader.py --format webdataset --subset commercial
    python enhanced_biomedica_downloader.py --format parquet --subset all
    python enhanced_biomedica_downloader.py --list-available
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Hugging Face相关导入
try:
    from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files, HfApi
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("警告: huggingface_hub未安装，部分功能可能不可用")

# WebDataset相关导入
try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False
    print("警告: webdataset未安装，WebDataset功能不可用")

# Parquet相关导入
try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("警告: pandas/pyarrow未安装，Parquet功能不可用")


class EnhancedBioMedICADownloader:
    """增强版BioMedICA数据集下载器"""
    
    # 可用的数据集配置
    AVAILABLE_DATASETS = {
        "commercial": {
            "webdataset": "BIOMEDICA/biomedica_webdataset_commercial",
            "parquet": "BIOMEDICA/biomedica_parquet_commercial"
        },
        "noncommercial": {
            "webdataset": "BIOMEDICA/biomedica_webdataset_noncommercial", 
            "parquet": "BIOMEDICA/biomedica_parquet_noncommercial"
        },
        "other": {
            "webdataset": "BIOMEDICA/biomedica_webdataset_other",
            "parquet": "BIOMEDICA/biomedica_parquet_other"
        },
        "24m": {
            "webdataset": "BIOMEDICA/biomedica_webdataset_24M"
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化下载器"""
        self.config = self._load_config(config_file)
        self._setup_logging()
        self.hf_api = HfApi() if HF_AVAILABLE else None
        
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "output_directory": "./biomedica_data",
            "max_retries": 5,
            "retry_delay": 15,
            "resume_download": True,
            "parallel_downloads": 4,
            "chunk_size": 1024 * 1024,  # 1MB
            "timeout": 300,
            "verify_checksum": True,
            "create_symlinks": False,
            "preprocessing": {
                "extract_images": True,
                "extract_captions": True,
                "filter_quality": True,
                "min_image_size": 224,
                "max_image_size": 2048
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"警告: 无法加载配置文件 {config_file}: {e}")
                print("使用默认配置")
        
        return default_config
    
    def _setup_logging(self):
        """设置日志记录"""
        log_dir = Path(self.config['output_directory']) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'enhanced_download.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def list_available_datasets(self) -> Dict[str, Any]:
        """列出所有可用的数据集"""
        print("🔍 可用的BioMedICA数据集:")
        print("=" * 60)
        
        for subset, formats in self.AVAILABLE_DATASETS.items():
            print(f"\n📊 {subset.upper()} 子集:")
            for format_type, repo_id in formats.items():
                print(f"  • {format_type}: {repo_id}")
        
        print(f"\n💡 使用说明:")
        print(f"  • 商业用途: commercial")
        print(f"  • 非商业用途: noncommercial") 
        print(f"  • 其他许可: other")
        print(f"  • 完整24M数据集: 24m")
        print(f"  • 支持格式: webdataset, parquet")
        
        return self.AVAILABLE_DATASETS
    
    def check_dataset_availability(self, repo_id: str) -> bool:
        """检查数据集是否可用"""
        if not HF_AVAILABLE:
            self.logger.warning("Hugging Face Hub不可用，无法检查数据集可用性")
            return False
            
        try:
            files = list_repo_files(repo_id=repo_id, repo_type="dataset")
            self.logger.info(f"数据集 {repo_id} 可用，包含 {len(files)} 个文件")
            return True
        except Exception as e:
            self.logger.error(f"数据集 {repo_id} 不可用: {e}")
            return False
    
    def download_webdataset(self, repo_id: str, output_dir: str) -> bool:
        """下载WebDataset格式的数据集"""
        if not WEBDATASET_AVAILABLE:
            self.logger.error("WebDataset库未安装，无法下载WebDataset格式")
            return False
            
        self.logger.info(f"开始下载WebDataset: {repo_id}")
        
        try:
            local_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=output_dir,
                resume_download=self.config['resume_download'],
                local_dir_use_symlinks=self.config['create_symlinks']
            )
            
            self.logger.info(f"✅ WebDataset下载完成: {local_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"WebDataset下载失败: {e}")
            return False
    
    def download_parquet(self, repo_id: str, output_dir: str) -> bool:
        """下载Parquet格式的数据集"""
        if not PARQUET_AVAILABLE:
            self.logger.error("Parquet库未安装，无法下载Parquet格式")
            return False
            
        self.logger.info(f"开始下载Parquet: {repo_id}")
        
        try:
            local_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset", 
                local_dir=output_dir,
                resume_download=self.config['resume_download'],
                local_dir_use_symlinks=self.config['create_symlinks']
            )
            
            self.logger.info(f"✅ Parquet下载完成: {local_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Parquet下载失败: {e}")
            return False
    
    def download_dataset(self, subset: str, format_type: str, output_dir: Optional[str] = None) -> bool:
        """下载指定子集和格式的数据集"""
        if subset not in self.AVAILABLE_DATASETS:
            self.logger.error(f"未知的子集: {subset}")
            return False
            
        if format_type not in self.AVAILABLE_DATASETS[subset]:
            self.logger.error(f"子集 {subset} 不支持格式 {format_type}")
            return False
        
        repo_id = self.AVAILABLE_DATASETS[subset][format_type]
        
        if not self.check_dataset_availability(repo_id):
            return False
        
        if output_dir is None:
            output_dir = Path(self.config['output_directory']) / f"{subset}_{format_type}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if format_type == "webdataset":
            return self.download_webdataset(repo_id, str(output_dir))
        elif format_type == "parquet":
            return self.download_parquet(repo_id, str(output_dir))
        else:
            self.logger.error(f"不支持的格式: {format_type}")
            return False
    
    def download_multiple_datasets(self, datasets: List[Dict[str, str]]) -> bool:
        """并行下载多个数据集"""
        if not datasets:
            self.logger.error("没有指定要下载的数据集")
            return False
        
        self.logger.info(f"开始并行下载 {len(datasets)} 个数据集")
        
        success_count = 0
        with ThreadPoolExecutor(max_workers=self.config['parallel_downloads']) as executor:
            futures = []
            
            for dataset in datasets:
                subset = dataset['subset']
                format_type = dataset['format']
                output_dir = dataset.get('output_dir')
                
                future = executor.submit(self.download_dataset, subset, format_type, output_dir)
                futures.append((future, subset, format_type))
            
            for future, subset, format_type in futures:
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                        self.logger.info(f"✅ {subset}_{format_type} 下载成功")
                    else:
                        self.logger.error(f"❌ {subset}_{format_type} 下载失败")
                except Exception as e:
                    self.logger.error(f"❌ {subset}_{format_type} 下载异常: {e}")
        
        self.logger.info(f"批量下载完成: {success_count}/{len(datasets)} 个数据集成功")
        return success_count == len(datasets)
    
    def create_webdataset_loader(self, dataset_path: str) -> Optional[wds.WebDataset]:
        """创建WebDataset加载器"""
        if not WEBDATASET_AVAILABLE:
            self.logger.error("WebDataset库未安装")
            return None
        
        try:
            dataset = wds.WebDataset(dataset_path)
            self.logger.info(f"WebDataset加载器创建成功: {dataset_path}")
            return dataset
        except Exception as e:
            self.logger.error(f"WebDataset加载失败: {e}")
            return None
    
    def create_parquet_loader(self, dataset_path: str) -> Optional[pd.DataFrame]:
        """创建Parquet加载器"""
        if not PARQUET_AVAILABLE:
            self.logger.error("Parquet库未安装")
            return None
        
        try:
            df = pd.read_parquet(dataset_path)
            self.logger.info(f"Parquet数据集加载成功: {dataset_path}, 形状: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Parquet加载失败: {e}")
            return None
    
    def preprocess_data(self, input_path: str, output_path: str, format_type: str) -> bool:
        """数据预处理"""
        self.logger.info(f"开始预处理数据: {input_path} -> {output_path}")
        
        try:
            if format_type == "webdataset":
                return self._preprocess_webdataset(input_path, output_path)
            elif format_type == "parquet":
                return self._preprocess_parquet(input_path, output_path)
            else:
                self.logger.error(f"不支持的格式: {format_type}")
                return False
        except Exception as e:
            self.logger.error(f"数据预处理失败: {e}")
            return False
    
    def _preprocess_webdataset(self, input_path: str, output_path: str) -> bool:
        """预处理WebDataset数据"""
        # 这里可以实现具体的WebDataset预处理逻辑
        # 例如：图像尺寸调整、质量过滤、格式转换等
        self.logger.info("WebDataset预处理功能待实现")
        return True
    
    def _preprocess_parquet(self, input_path: str, output_path: str) -> bool:
        """预处理Parquet数据"""
        # 这里可以实现具体的Parquet预处理逻辑
        # 例如：数据清洗、特征提取、格式转换等
        self.logger.info("Parquet预处理功能待实现")
        return True
    
    def get_dataset_info(self, dataset_path: str, format_type: str) -> Dict[str, Any]:
        """获取数据集信息"""
        info = {
            "path": dataset_path,
            "format": format_type,
            "size": 0,
            "files": 0,
            "available": False
        }
        
        try:
            path = Path(dataset_path)
            if path.exists():
                info["available"] = True
                info["size"] = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                info["files"] = len(list(path.rglob('*')))
                
                if format_type == "parquet" and PARQUET_AVAILABLE:
                    try:
                        df = pd.read_parquet(dataset_path)
                        info["rows"] = len(df)
                        info["columns"] = list(df.columns)
                    except:
                        pass
        except Exception as e:
            self.logger.error(f"获取数据集信息失败: {e}")
        
        return info


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版BioMedICA数据集下载器')
    parser.add_argument('--subset', type=str, choices=['commercial', 'noncommercial', 'other', '24m', 'all'],
                       help='数据集子集')
    parser.add_argument('--format', type=str, choices=['webdataset', 'parquet', 'both'],
                       help='数据格式')
    parser.add_argument('--output', type=str, help='输出目录')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--list-available', action='store_true', help='列出可用数据集')
    parser.add_argument('--parallel', type=int, default=4, help='并行下载数量')
    parser.add_argument('--preprocess', action='store_true', help='下载后进行预处理')
    parser.add_argument('--info', type=str, help='获取指定数据集信息')
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = EnhancedBioMedICADownloader(args.config)
    
    try:
        if args.list_available:
            # 列出可用数据集
            downloader.list_available_datasets()
            
        elif args.info:
            # 获取数据集信息
            info = downloader.get_dataset_info(args.info, "webdataset")
            print(f"数据集信息: {json.dumps(info, indent=2, ensure_ascii=False)}")
            
        elif args.subset and args.format:
            # 下载指定数据集
            if args.subset == 'all':
                # 下载所有子集
                datasets = []
                for subset in ['commercial', 'noncommercial', 'other']:
                    if args.format == 'both':
                        for fmt in ['webdataset', 'parquet']:
                            datasets.append({
                                'subset': subset,
                                'format': fmt,
                                'output_dir': args.output
                            })
                    else:
                        datasets.append({
                            'subset': subset,
                            'format': args.format,
                            'output_dir': args.output
                        })
                
                success = downloader.download_multiple_datasets(datasets)
            else:
                success = downloader.download_dataset(args.subset, args.format, args.output)
            
            if success:
                print("✅ 下载完成")
                if args.preprocess:
                    print("🔄 开始数据预处理...")
                    # 这里可以添加预处理逻辑
            else:
                print("❌ 下载失败")
                sys.exit(1)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
