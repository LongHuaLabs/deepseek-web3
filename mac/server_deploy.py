#!/usr/bin/env python3
"""
服务器端部署脚本
用于在服务器上启动 DeepSeek-R1-Distill-Llama-8B 模型服务
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
import psutil
import GPUtil

def check_system_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    # 检查GPU
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("❌ 未检测到GPU")
            return False
        
        print(f"✅ 检测到 {len(gpus)} 个GPU:")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name} ({gpu.memoryTotal}MB)")
    except:
        print("⚠️  无法检测GPU信息，请确保安装了nvidia-ml-py")
    
    # 检查内存
    memory = psutil.virtual_memory()
    print(f"💾 系统内存: {memory.total / (1024**3):.1f} GB")
    
    # 检查CUDA
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA环境正常")
        else:
            print("❌ CUDA环境异常")
            return False
    except FileNotFoundError:
        print("❌ 未找到nvidia-smi命令")
        return False
    
    return True

def install_dependencies():
    """安装依赖"""
    print("📦 安装依赖...")
    
    dependencies = [
        "vllm",
        "transformers",
        "torch",
        "fastapi",
        "uvicorn"
    ]
    
    for dep in dependencies:
        try:
            print(f"安装 {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
            print(f"✅ {dep} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {dep} 安装失败: {e}")
            return False
    
    return True

def create_vllm_config(model_path: str, port: int = 8000, 
                      tensor_parallel_size: int = 8,
                      max_model_len: int = 32768) -> str:
    """创建vLLM配置文件"""
    config = {
        "model": model_path,
        "host": "0.0.0.0",
        "port": port,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": True,
        "disable_log_stats": False,
        "served_model_name": "deepseek-r1-distill-llama-8b"
    }
    
    config_file = "vllm_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ 配置文件已创建: {config_file}")
    return config_file

def create_startup_script(model_path: str, port: int = 8000,
                         tensor_parallel_size: int = 8,
                         max_model_len: int = 32768) -> str:
    """创建启动脚本"""
    script_content = f"""#!/bin/bash
# DeepSeek-R1-Distill-Llama-8B vLLM 启动脚本

echo "🚀 启动 DeepSeek-R1-Distill-Llama-8B 服务..."
echo "模型路径: {model_path}"
echo "端口: {port}"
echo "GPU数量: {tensor_parallel_size}"
echo "最大序列长度: {max_model_len}"
echo "=" * 60

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_MODELSCOPE=False

# 启动vLLM服务
python -m vllm.entrypoints.openai.api_server \\
    --model {model_path} \\
    --host 0.0.0.0 \\
    --port {port} \\
    --tensor-parallel-size {tensor_parallel_size} \\
    --max-model-len {max_model_len} \\
    --gpu-memory-utilization 0.9 \\
    --trust-remote-code \\
    --served-model-name deepseek-r1-distill-llama-8b \\
    --disable-log-stats
"""
    
    script_file = "start_vllm.sh"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # 添加执行权限
    os.chmod(script_file, 0o755)
    
    print(f"✅ 启动脚本已创建: {script_file}")
    return script_file

def create_systemd_service(model_path: str, working_dir: str,
                          port: int = 8000, tensor_parallel_size: int = 8) -> str:
    """创建systemd服务文件"""
    service_content = f"""[Unit]
Description=DeepSeek-R1-Distill-Llama-8B vLLM Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={working_dir}
Environment=CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
Environment=VLLM_USE_MODELSCOPE=False
ExecStart=/usr/bin/python3 -m vllm.entrypoints.openai.api_server \\
    --model {model_path} \\
    --host 0.0.0.0 \\
    --port {port} \\
    --tensor-parallel-size {tensor_parallel_size} \\
    --max-model-len 32768 \\
    --gpu-memory-utilization 0.9 \\
    --trust-remote-code \\
    --served-model-name deepseek-r1-distill-llama-8b
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    service_file = "deepseek-vllm.service"
    with open(service_file, 'w') as f:
        f.write(service_content)
    
    print(f"✅ Systemd服务文件已创建: {service_file}")
    print("📝 要安装服务，请运行:")
    print(f"   sudo cp {service_file} /etc/systemd/system/")
    print("   sudo systemctl daemon-reload")
    print("   sudo systemctl enable deepseek-vllm")
    print("   sudo systemctl start deepseek-vllm")
    
    return service_file

def create_test_client() -> str:
    """创建测试客户端"""
    test_content = '''#!/usr/bin/env python3
"""
vLLM服务测试客户端
"""

import requests
import json

def test_vllm_service(host="localhost", port=8000):
    """测试vLLM服务"""
    base_url = f"http://{host}:{port}"
    
    # 测试健康检查
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ 服务健康检查通过")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        return False
    
    # 测试模型列表
    try:
        response = requests.get(f"{base_url}/v1/models")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ 可用模型: {[m['id'] for m in models['data']]}")
        else:
            print(f"❌ 获取模型列表失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 获取模型列表异常: {e}")
    
    # 测试文本生成
    try:
        payload = {
            "model": "deepseek-r1-distill-llama-8b",
            "messages": [
                {"role": "user", "content": "你好，请介绍一下你自己。"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        print("🧪 测试文本生成...")
        response = requests.post(f"{base_url}/v1/chat/completions", 
                               json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✅ 生成成功:")
            print(f"   {content}")
            return True
        else:
            print(f"❌ 生成失败: {response.status_code}")
            print(f"   {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 生成异常: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试vLLM服务")
    parser.add_argument("--host", default="localhost", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    
    args = parser.parse_args()
    
    print(f"🔍 测试vLLM服务 {args.host}:{args.port}")
    print("=" * 50)
    
    success = test_vllm_service(args.host, args.port)
    
    if success:
        print("🎉 所有测试通过！")
    else:
        print("❌ 测试失败，请检查服务状态")
'''
    
    test_file = "test_vllm.py"
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    os.chmod(test_file, 0o755)
    
    print(f"✅ 测试客户端已创建: {test_file}")
    return test_file

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-R1-Distill-Llama-8B 服务器部署工具")
    parser.add_argument("--model-path", required=True, help="模型路径")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--tensor-parallel-size", type=int, default=8, help="GPU并行数量")
    parser.add_argument("--max-model-len", type=int, default=32768, help="最大序列长度")
    parser.add_argument("--skip-install", action="store_true", help="跳过依赖安装")
    parser.add_argument("--create-service", action="store_true", help="创建systemd服务")
    
    args = parser.parse_args()
    
    print("🚀 DeepSeek-R1-Distill-Llama-8B 服务器部署")
    print("=" * 60)
    
    # 检查模型路径
    if not Path(args.model_path).exists():
        print(f"❌ 模型路径不存在: {args.model_path}")
        sys.exit(1)
    
    # 检查系统要求
    if not check_system_requirements():
        print("❌ 系统要求检查失败")
        sys.exit(1)
    
    # 安装依赖
    if not args.skip_install:
        if not install_dependencies():
            print("❌ 依赖安装失败")
            sys.exit(1)
    
    # 创建配置文件
    config_file = create_vllm_config(
        args.model_path, 
        args.port, 
        args.tensor_parallel_size,
        args.max_model_len
    )
    
    # 创建启动脚本
    startup_script = create_startup_script(
        args.model_path,
        args.port,
        args.tensor_parallel_size,
        args.max_model_len
    )
    
    # 创建测试客户端
    test_client = create_test_client()
    
    # 创建systemd服务（可选）
    if args.create_service:
        service_file = create_systemd_service(
            args.model_path,
            os.getcwd(),
            args.port,
            args.tensor_parallel_size
        )
    
    print("\n🎉 部署准备完成！")
    print("=" * 60)
    print("📝 下一步操作:")
    print(f"1. 启动服务: ./{startup_script}")
    print(f"2. 测试服务: python {test_client}")
    print("3. 查看日志: tail -f vllm.log")
    
    if args.create_service:
        print("\n🔧 Systemd服务:")
        print("1. 安装服务: sudo cp deepseek-vllm.service /etc/systemd/system/")
        print("2. 重载配置: sudo systemctl daemon-reload")
        print("3. 启用服务: sudo systemctl enable deepseek-vllm")
        print("4. 启动服务: sudo systemctl start deepseek-vllm")
        print("5. 查看状态: sudo systemctl status deepseek-vllm")

if __name__ == "__main__":
    main() 