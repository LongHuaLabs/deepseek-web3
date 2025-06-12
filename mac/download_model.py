#!/usr/bin/env python3
"""
DeepSeek-R1-Distill-Llama-8B 模型下载工具
支持断点续传和多线程下载
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from huggingface_hub import hf_hub_download, snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm
import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                 cache_dir: str = "./models", 
                 token: Optional[str] = None,
                 max_workers: int = 4):
        """
        初始化模型下载器
        
        Args:
            model_id: HuggingFace模型ID
            cache_dir: 本地缓存目录
            token: HuggingFace访问令牌（如果模型需要）
            max_workers: 并发下载线程数
        """
        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.token = token
        self.max_workers = max_workers
        self.api = HfApi(token=token)
        
        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化模型下载器: {model_id}")
        logger.info(f"缓存目录: {self.cache_dir.absolute()}")
    
    def get_model_info(self):
        """获取模型信息"""
        try:
            model_info = self.api.model_info(self.model_id)
            logger.info(f"模型信息:")
            logger.info(f"  - 模型ID: {model_info.modelId}")
            logger.info(f"  - 创建时间: {model_info.created_at}")
            logger.info(f"  - 最后修改: {model_info.last_modified}")
            logger.info(f"  - 下载数: {model_info.downloads}")
            
            # 获取文件列表
            files = [f.rfilename for f in model_info.siblings]
            logger.info(f"  - 文件数量: {len(files)}")
            
            # 计算总大小
            total_size = 0
            for file_info in model_info.siblings:
                if hasattr(file_info, 'size') and file_info.size:
                    total_size += file_info.size
            
            if total_size > 0:
                logger.info(f"  - 总大小: {self._format_size(total_size)}")
            
            return model_info
            
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return None
    
    def _format_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def download_file(self, filename: str, subfolder: Optional[str] = None) -> bool:
        """
        下载单个文件
        
        Args:
            filename: 文件名
            subfolder: 子文件夹路径
            
        Returns:
            bool: 下载是否成功
        """
        try:
            logger.info(f"开始下载文件: {filename}")
            
            # 构建本地文件路径
            if subfolder:
                local_path = self.cache_dir / self.model_id.replace('/', '--') / subfolder / filename
            else:
                local_path = self.cache_dir / self.model_id.replace('/', '--') / filename
            
            # 检查文件是否已存在
            if local_path.exists():
                logger.info(f"文件已存在，跳过下载: {filename}")
                return True
            
            # 确保目录存在
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 下载文件
            downloaded_path = hf_hub_download(
                repo_id=self.model_id,
                filename=filename,
                subfolder=subfolder,
                cache_dir=str(self.cache_dir),
                token=self.token,
                resume_download=True,  # 支持断点续传
                local_files_only=False
            )
            
            logger.info(f"文件下载完成: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"下载文件失败 {filename}: {e}")
            return False
    
    def download_all_files(self, ignore_patterns: Optional[List[str]] = None) -> bool:
        """
        下载所有模型文件
        
        Args:
            ignore_patterns: 要忽略的文件模式列表
            
        Returns:
            bool: 下载是否成功
        """
        try:
            logger.info("开始下载完整模型...")
            
            # 获取模型信息
            model_info = self.get_model_info()
            if not model_info:
                return False
            
            # 获取所有文件
            files_to_download = []
            for file_info in model_info.siblings:
                filename = file_info.rfilename
                
                # 检查是否需要忽略
                if ignore_patterns:
                    should_ignore = any(pattern in filename for pattern in ignore_patterns)
                    if should_ignore:
                        logger.info(f"忽略文件: {filename}")
                        continue
                
                files_to_download.append(filename)
            
            logger.info(f"需要下载 {len(files_to_download)} 个文件")
            
            # 使用线程池并发下载
            success_count = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有下载任务
                future_to_file = {
                    executor.submit(self.download_file, filename): filename 
                    for filename in files_to_download
                }
                
                # 处理完成的任务
                with tqdm(total=len(files_to_download), desc="下载进度") as pbar:
                    for future in as_completed(future_to_file):
                        filename = future_to_file[future]
                        try:
                            success = future.result()
                            if success:
                                success_count += 1
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"下载任务异常 {filename}: {e}")
                            pbar.update(1)
            
            logger.info(f"下载完成: {success_count}/{len(files_to_download)} 个文件成功")
            return success_count == len(files_to_download)
            
        except Exception as e:
            logger.error(f"批量下载失败: {e}")
            return False
    
    def download_snapshot(self, ignore_patterns: Optional[List[str]] = None) -> Optional[str]:
        """
        使用snapshot_download下载完整模型（推荐方式）
        
        Args:
            ignore_patterns: 要忽略的文件模式列表
            
        Returns:
            str: 下载的模型路径，失败返回None
        """
        try:
            logger.info("使用snapshot方式下载模型...")
            
            # 设置忽略模式
            if ignore_patterns is None:
                ignore_patterns = [
                    "*.git*",
                    "*.DS_Store*",
                    "*__pycache__*"
                ]
            
            # 下载模型
            model_path = snapshot_download(
                repo_id=self.model_id,
                cache_dir=str(self.cache_dir),
                token=self.token,
                ignore_patterns=ignore_patterns,
                resume_download=True,  # 支持断点续传
                local_files_only=False
            )
            
            logger.info(f"模型下载完成，路径: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Snapshot下载失败: {e}")
            return None
    
    def verify_download(self, model_path: str) -> bool:
        """
        验证下载的模型完整性
        
        Args:
            model_path: 模型路径
            
        Returns:
            bool: 验证是否通过
        """
        try:
            logger.info("验证模型完整性...")
            
            model_dir = Path(model_path)
            if not model_dir.exists():
                logger.error(f"模型目录不存在: {model_path}")
                return False
            
            # 检查关键文件
            required_files = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json"
            ]
            
            missing_files = []
            for file in required_files:
                if not (model_dir / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                logger.warning(f"缺少文件: {missing_files}")
            
            # 检查模型权重文件
            weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
            if not weight_files:
                logger.error("未找到模型权重文件")
                return False
            
            logger.info(f"找到 {len(weight_files)} 个权重文件")
            
            # 计算总大小
            total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            logger.info(f"模型总大小: {self._format_size(total_size)}")
            
            logger.info("模型验证通过")
            return True
            
        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            return False
    
    def create_deployment_info(self, model_path: str) -> str:
        """
        创建部署信息文件
        
        Args:
            model_path: 模型路径
            
        Returns:
            str: 部署信息文件路径
        """
        try:
            deployment_info = {
                "model_id": self.model_id,
                "model_path": model_path,
                "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cache_dir": str(self.cache_dir.absolute()),
                "deployment_instructions": {
                    "step1": "将模型文件夹上传到服务器",
                    "step2": "在服务器上安装vLLM: pip install vllm",
                    "step3": "启动vLLM服务",
                    "command_example": f"python -m vllm.entrypoints.openai.api_server --model {model_path} --host 0.0.0.0 --port 8000"
                },
                "server_config": {
                    "gpu_count": 8,
                    "gpu_model": "NVIDIA H20",
                    "gpu_memory": "95GB per GPU",
                    "cuda_version": "12.4",
                    "driver_version": "550.90.07",
                    "os": "Ubuntu 22.04 LTS"
                }
            }
            
            info_file = self.cache_dir / "deployment_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(deployment_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"部署信息已保存到: {info_file}")
            return str(info_file)
            
        except Exception as e:
            logger.error(f"创建部署信息失败: {e}")
            return ""


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-R1-Distill-Llama-8B 模型下载工具")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                       help="HuggingFace模型ID")
    parser.add_argument("--cache-dir", default="./models", 
                       help="本地缓存目录")
    parser.add_argument("--token", 
                       help="HuggingFace访问令牌")
    parser.add_argument("--max-workers", type=int, default=4, 
                       help="并发下载线程数")
    parser.add_argument("--method", choices=["snapshot", "individual"], default="snapshot",
                       help="下载方式: snapshot(推荐) 或 individual")
    parser.add_argument("--ignore-patterns", nargs="*", 
                       default=["*.git*", "*.DS_Store*", "*__pycache__*"],
                       help="要忽略的文件模式")
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = ModelDownloader(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        token=args.token,
        max_workers=args.max_workers
    )
    
    # 获取模型信息
    model_info = downloader.get_model_info()
    if not model_info:
        logger.error("无法获取模型信息，请检查网络连接和模型ID")
        sys.exit(1)
    
    # 下载模型
    if args.method == "snapshot":
        model_path = downloader.download_snapshot(args.ignore_patterns)
    else:
        success = downloader.download_all_files(args.ignore_patterns)
        model_path = str(downloader.cache_dir / args.model_id.replace('/', '--')) if success else None
    
    if not model_path:
        logger.error("模型下载失败")
        sys.exit(1)
    
    # 验证下载
    if not downloader.verify_download(model_path):
        logger.error("模型验证失败")
        sys.exit(1)
    
    # 创建部署信息
    deployment_info_file = downloader.create_deployment_info(model_path)
    
    logger.info("=" * 60)
    logger.info("模型下载完成！")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"部署信息: {deployment_info_file}")
    logger.info("=" * 60)
    
    print("\n下一步操作:")
    print("1. 将模型文件夹压缩并上传到服务器")
    print("2. 在服务器上解压模型文件")
    print("3. 使用vLLM部署模型服务")
    print(f"4. 参考部署信息文件: {deployment_info_file}")


if __name__ == "__main__":
    main() 