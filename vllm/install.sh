#!/bin/bash

# DeepSeek-R1-Distill-Llama-8B H20*8服务器安装脚本
# 支持Ubuntu 22.04 LTS + CUDA 12.4 + H20*8 GPU

set -e

echo "=========================================="
echo "DeepSeek-R1 vLLM H20*8 服务器安装脚本"
echo "=========================================="

# # 检查是否为root用户
# if [[ $EUID -eq 0 ]]; then
#    echo "请不要使用root用户运行此脚本"
#    exit 1
# fi

# 检查CUDA版本
echo "检查CUDA环境..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 未检测到NVIDIA驱动，请先安装NVIDIA驱动"
    exit 1
fi

nvidia-smi
echo "✅ NVIDIA驱动检测正常"

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $GPU_COUNT 块GPU"

if [ $GPU_COUNT -lt 8 ]; then
    echo "⚠️  警告: 检测到GPU数量少于8块，将调整tensor_parallel_size"
fi

# 更新系统包
echo "更新系统包..."
sudo apt update && sudo apt upgrade -y

# 安装系统依赖
echo "安装系统依赖..."
sudo apt install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    htop \
    nvtop \
    vim \
    tmux \
    screen

# 创建Python虚拟环境
echo "创建Python虚拟环境..."
if [ ! -d "venv" ]; then
    python3.10 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip setuptools wheel

# 安装PyTorch (CUDA 12.1版本兼容CUDA 12.4)
echo "安装PyTorch with CUDA支持..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证PyTorch CUDA支持
echo "验证PyTorch CUDA支持..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 安装vLLM
echo "安装vLLM..."
pip install vllm

# 安装其他依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 安装Flash Attention 2 (可选，性能优化)
# echo "安装Flash Attention 2..."
# pip install flash-attn --no-build-isolation

# 创建模型缓存目录
echo "创建模型缓存目录..."
mkdir -p model_cache logs

# 设置环境变量
echo "配置环境变量..."
cat >> ~/.bashrc << EOF

# DeepSeek-R1 vLLM 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_CACHE=$(pwd)/model_cache
export HF_HOME=$(pwd)/model_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_USE_MODELSCOPE=false
EOF

# 重新加载环境变量
source ~/.bashrc

# 预下载模型 (可选)
echo "是否要预下载模型? (y/n)"
read -r download_model
if [[ $download_model == "y" || $download_model == "Y" ]]; then
    echo "开始下载DeepSeek-R1-Distill-Llama-8B模型..."
    export TRANSFORMERS_CACHE=$(pwd)/model_cache
    export HF_HOME=$(pwd)/model_cache
    export HF_HUB_ENABLE_HF_TRANSFER=1
    # 确保使用在线模式
    export TRANSFORMERS_OFFLINE=0
    unset HF_HUB_OFFLINE
    
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('开始下载模型...')
model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

# 下载tokenizer
print('下载tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir='./model_cache'
)

print('下载完成!')
print(f'模型缓存位置: {model_name}')
"
fi

# 创建系统服务文件 (可选)
echo "是否创建systemd服务? (y/n)"
read -r create_service
if [[ $create_service == "y" || $create_service == "Y" ]]; then
    sudo tee /etc/systemd/system/deepseek-web3.service > /dev/null << EOF
[Unit]
Description=DeepSeek-R1 vLLM Web3 Quant API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
Environment=CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
Environment=TRANSFORMERS_CACHE=$(pwd)/model_cache
Environment=HF_HOME=$(pwd)/model_cache
Environment=HF_HUB_ENABLE_HF_TRANSFER=1
ExecStart=$(pwd)/venv/bin/python DeepSeek-R1-Distill-Llama-8B-app.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    echo "✅ systemd服务已创建，使用以下命令管理:"
    echo "启动服务: sudo systemctl start deepseek-web3"
    echo "开机自启: sudo systemctl enable deepseek-web3"
    echo "查看状态: sudo systemctl status deepseek-web3"
    echo "查看日志: sudo journalctl -u deepseek-web3 -f"
fi

# 创建启动脚本
echo "创建启动脚本..."
cat > start_server.sh << 'EOF'
#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_CACHE=$(pwd)/model_cache
export HF_HOME=$(pwd)/model_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi

echo "启动DeepSeek-R1 vLLM服务器..."
python DeepSeek-R1-Distill-Llama-8B-app.py --host 0.0.0.0 --port 8000
EOF

chmod +x start_server.sh

# 创建测试脚本
cat > test_api.py << 'EOF'
#!/usr/bin/env python3
"""
API测试脚本
"""

import requests
import json
import time

def test_health():
    """测试健康检查接口"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        print("健康检查:", response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False

def test_chat():
    """测试聊天接口"""
    try:
        data = {
            "messages": [
                {"role": "user", "content": "你好，请介绍一下你自己"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(
            "http://localhost:8000/chat",
            json=data,
            timeout=60
        )
        
        result = response.json()
        print("聊天测试:")
        print(f"响应: {result.get('response', '')}")
        return response.status_code == 200
    except Exception as e:
        print(f"聊天测试失败: {e}")
        return False

def test_quant_strategy():
    """测试量化策略生成"""
    try:
        data = {
            "market_data": "BTC价格在65000-67000区间震荡，成交量较前日增加15%",
            "strategy_type": "网格交易",
            "risk_level": "中",
            "timeframe": "15m",
            "target_asset": "BTC/USDT",
            "capital": 10000
        }
        
        response = requests.post(
            "http://localhost:8000/generate-quant-strategy",
            json=data,
            timeout=120
        )
        
        result = response.json()
        print("量化策略测试:")
        print(f"策略ID: {result.get('strategy_id', '')}")
        print(f"策略预览: {result.get('strategy', '')[:200]}...")
        return response.status_code == 200
    except Exception as e:
        print(f"量化策略测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始API测试...")
    
    # 等待服务启动
    print("等待服务启动...")
    for i in range(30):
        if test_health():
            print("✅ 服务已启动")
            break
        time.sleep(2)
        print(f"等待中... ({i+1}/30)")
    else:
        print("❌ 服务启动超时")
        exit(1)
    
    # 运行测试
    tests = [
        ("聊天接口", test_chat),
        ("量化策略", test_quant_strategy),
    ]
    
    for name, test_func in tests:
        print(f"\n测试 {name}...")
        if test_func():
            print(f"✅ {name} 测试通过")
        else:
            print(f"❌ {name} 测试失败")
EOF

chmod +x test_api.py

echo "=========================================="
echo "✅ 安装完成!"
echo "=========================================="
echo ""
echo "使用方法:"
echo "1. 启动服务: ./start_server.sh"
echo "2. 测试API: python test_api.py"
echo "3. 服务地址: http://localhost:8000"
echo "4. API文档: http://localhost:8000/docs"
echo ""
echo "主要API端点:"
echo "- /health - 健康检查"
echo "- /chat - 聊天对话"
echo "- /generate-quant-strategy - 生成量化策略"
echo "- /analyze-market - 市场分析"
echo "- /optimize-strategy - 策略优化"
echo ""
echo "如果安装了systemd服务，可以使用:"
echo "- sudo systemctl start deepseek-web3"
echo "- sudo systemctl status deepseek-web3"
echo ""
echo "注意事项:"
echo "- 首次启动会自动下载模型，需要一些时间"
echo "- 确保有足够的磁盘空间存储模型(约15GB)"
echo "- 模型加载需要几分钟时间，请耐心等待"
echo "" 