#!/usr/bin/env python3
"""
快速下载 DeepSeek-R1-Distill-Llama-8B 模型
简化版本，一键下载
"""

from huggingface_hub import snapshot_download
import os
from pathlib import Path
import time

def main():
    print("🚀 开始下载 DeepSeek-R1-Distill-Llama-8B 模型...")
    print("=" * 60)
    
    # 配置
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    cache_dir = "./models"
    
    # 创建目录
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        start_time = time.time()
        
        print(f"📦 模型ID: {model_id}")
        print(f"📁 下载目录: {os.path.abspath(cache_dir)}")
        print("⏳ 开始下载，请耐心等待...")
        print()
        
        # 下载模型
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,  # 支持断点续传
            ignore_patterns=[
                "*.git*",
                "*.DS_Store*", 
                "*__pycache__*",
                "*.md"  # 忽略README等文档文件
            ]
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("✅ 下载完成！")
        print("=" * 60)
        print(f"📍 模型路径: {model_path}")
        print(f"⏱️  下载耗时: {duration:.2f} 秒")
        print()
        
        # 显示文件信息
        model_dir = Path(model_path)
        files = list(model_dir.rglob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        
        print(f"📊 文件统计:")
        print(f"   - 文件数量: {len([f for f in files if f.is_file()])}")
        print(f"   - 总大小: {total_size / (1024**3):.2f} GB")
        print()
        
        print("🎯 下一步操作:")
        print("1. 压缩模型文件夹:")
        print(f"   tar -czf deepseek-model.tar.gz -C {cache_dir} .")
        print()
        print("2. 上传到服务器:")
        print("   scp deepseek-model.tar.gz user@server:/path/to/models/")
        print()
        print("3. 在服务器上解压:")
        print("   tar -xzf deepseek-model.tar.gz")
        print()
        print("4. 启动vLLM服务:")
        print(f"   python -m vllm.entrypoints.openai.api_server \\")
        print(f"     --model /path/to/models/{model_id.split('/')[-1]} \\")
        print(f"     --host 0.0.0.0 --port 8000 \\")
        print(f"     --tensor-parallel-size 8")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("💡 建议:")
        print("1. 检查网络连接")
        print("2. 确保有足够的磁盘空间")
        print("3. 如果是网络问题，可以重新运行脚本（支持断点续传）")

if __name__ == "__main__":
    main() 