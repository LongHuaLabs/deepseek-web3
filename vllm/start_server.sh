#!/bin/bash

# 激活虚拟环境
# source venv/bin/activate
# 或者如果你使用的是 conda 环境：
# conda activate deepseek_env

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_CACHE=$(pwd)/model_cache
export HF_HOME=$(pwd)/model_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi

# 后台启动DeepSeek-R1 vLLM服务器
nohup python DeepSeek-R1-Distill-Llama-8B-app.py --host 0.0.0.0 --port 8822 > output.log 2>&1 &
echo $! > server.pid

echo "DeepSeek-R1 vLLM服务器已经在后台启动。"
