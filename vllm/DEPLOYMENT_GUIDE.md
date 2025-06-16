# DeepSeek-R1-Distill-Llama-8B H20*8 服务器部署指南

```bash
cd /project/deepseek-web3/

conda activate deepseek_env

cd vllm

#启动服务
./start_server.sh

# 测试
python test_api.py
```


## 🖥️ 服务器环境
- **GPU**: 8 × NVIDIA H20 (每块 ~95GB 显存)
- **CUDA**: 12.4
- **驱动**: 550.90.07
- **操作系统**: Ubuntu 22.04 LTS
- **总显存**: ~760GB (8 × 95GB)

## 🚀 快速部署

### 1. 环境检查
```bash
# 检查CUDA和GPU状态
nvidia-smi

# 检查CUDA版本
nvcc --version

# 检查驱动版本
cat /proc/driver/nvidia/version
```

### 2. 一键安装
```bash
# 克隆项目到vllm目录
cd vllm

# 给安装脚本执行权限
chmod +x install.sh

# 运行自动化安装脚本
./install.sh
```

安装脚本会自动完成：
- ✅ 系统依赖安装
- ✅ Python 3.10 虚拟环境创建
- ✅ PyTorch (CUDA 12.1) 安装
- ✅ vLLM 框架安装
- ✅ 项目依赖安装
- ✅ Flash Attention 2 优化
- ✅ 环境变量配置
- ✅ 模型预下载 (可选)
- ✅ systemd 服务创建 (可选)

### 3. 启动服务
```bash
# 方式1: 直接启动
./start_server.sh

# 方式2: 使用systemd服务 (如果已创建)
sudo systemctl start deepseek-vllm
sudo systemctl enable deepseek-vllm  # 开机自启

# 方式3: 后台运行
nohup ./start_server.sh > server.log 2>&1 &

# 方式4: 使用screen/tmux
screen -S deepseek
./start_server.sh
# Ctrl+A+D 分离会话
```

### 4. 验证部署
```bash
# 测试API功能
python test_api.py

# 检查服务状态
curl http://localhost:8000/health

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

## 📊 性能配置

### H20*8 优化配置

当前配置已针对你的H20*8服务器优化：

```python
class VLLMConfig:
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # 8B模型
        self.tensor_parallel_size = 8           # 使用全部8张H20
        self.gpu_memory_utilization = 0.85      # H20显存利用率85%
        self.max_model_len = 32768             # 32K上下文长度
        self.dtype = "bfloat16"                # bfloat16精度
        self.quantization = None               # 显存充足，无需量化
        self.max_num_seqs = 256                # 支持256并发请求
        self.max_num_batched_tokens = 8192     # 批处理tokens
```

### 启动参数
```bash
python DeepSeek-R1-Distill-Llama-8B-app.py \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 \
    --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

## 🔧 API 使用指南

### 健康检查
```bash
curl http://localhost:8000/health
```

### 聊天对话
```bash
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "你好，请介绍一下你自己"}
  ],
  "max_tokens": 100,
  "temperature": 0.7
}'
```

### 生成量化策略
```bash
curl -X POST "http://localhost:8000/generate-quant-strategy" \
-H "Content-Type: application/json" \
-d '{
  "market_data": "BTC价格在65000-67000区间震荡，成交量较前日增加15%",
  "strategy_type": "网格交易",
  "risk_level": "中",
  "timeframe": "15m",
  "target_asset": "BTC/USDT",
  "capital": 10000
}'
```

### 市场分析
```bash
curl -X POST "http://localhost:8000/analyze-market" \
-H "Content-Type: application/json" \
-d '{
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "indicators": ["RSI", "MACD", "Bollinger"]
}'
```

## 📈 监控和日志

### 系统监控
```bash
# GPU使用情况
nvidia-smi -l 1

# 更详细的GPU监控
nvtop

# 系统资源
htop

# 磁盘使用
df -h
```

### 应用日志
```bash
# 查看应用日志
tail -f logs/vllm_app_*.log

# systemd服务日志
sudo journalctl -u deepseek-vllm -f

# 实时监控日志
tail -f server.log
```

### Prometheus指标
```bash
# 获取监控指标
curl http://localhost:8000/metrics
```

## 🛠️ 故障排除

### 常见问题

#### 1. 模型加载失败
```bash
# 检查模型缓存
ls -la model_cache/

# 重新下载模型
rm -rf model_cache/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    cache_dir='./model_cache',
    trust_remote_code=True
)
"
```

#### 2. GPU内存不足
```bash
# 降低显存利用率
python DeepSeek-R1-Distill-Llama-8B-app.py --gpu-memory-utilization 0.7

# 减少并发请求数
# 修改VLLMConfig中的max_num_seqs = 128
```

#### 3. 服务启动慢
首次启动会下载模型，大约需要：
- **模型下载**: 15GB，约10-30分钟 (取决于网络)
- **模型加载**: 约2-5分钟
- **引擎初始化**: 约1-2分钟

#### 4. API响应慢
```bash
# 检查GPU使用率
nvidia-smi

# 调整温度参数降低推理时间
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d '{
  "messages": [{"role": "user", "content": "你好"}],
  "temperature": 0.1,
  "max_tokens": 50
}'
```

## 🔧 高级配置

### 环境变量
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_CACHE=/path/to/model_cache
export HF_HOME=/path/to/model_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_USE_MODELSCOPE=false
```

### 自定义模型路径
```bash
# 如果使用本地模型
python DeepSeek-R1-Distill-Llama-8B-app.py \
    --model-name /path/to/local/model
```

### 调整并发配置
修改 `VLLMConfig` 类中的参数：
```python
self.max_num_seqs = 128          # 减少并发降低显存使用
self.max_num_batched_tokens = 4096  # 减少批处理大小
```

## 📊 性能预期

### H20*8 配置下的预期性能

- **模型大小**: ~15GB
- **显存占用**: 约35-45GB (每张H20)
- **推理速度**: ~50-100 tokens/s
- **并发能力**: 100+ 并发请求
- **上下文长度**: 32K tokens
- **启动时间**: 3-8分钟 (首次)

### 性能优化建议

1. **显存优化**:
   - 使用 `bfloat16` 精度
   - 适当调整 `gpu_memory_utilization`
   - 启用KV cache优化

2. **推理优化**:
   - 使用 Flash Attention 2
   - 启用 CUDA 图优化
   - 合理设置batch大小

3. **并发优化**:
   - 调整 `max_num_seqs` 参数
   - 使用连接池
   - 启用异步处理

## 🔄 更新和维护

### 更新模型
```bash
# 停止服务
sudo systemctl stop deepseek-vllm

# 清除模型缓存
rm -rf model_cache/models--deepseek-ai--*

# 重新下载模型
./install.sh

# 启动服务
sudo systemctl start deepseek-vllm
```

### 更新代码
```bash
# 备份配置
cp DeepSeek-R1-Distill-Llama-8B-app.py DeepSeek-R1-Distill-Llama-8B-app.py.bak

# 更新代码
git pull

# 重启服务
sudo systemctl restart deepseek-vllm
```

## 📞 支持

如果遇到问题，请检查：
1. 日志文件: `logs/vllm_app_*.log`
2. 系统日志: `sudo journalctl -u deepseek-vllm`
3. GPU状态: `nvidia-smi`
4. 磁盘空间: `df -h`

常用调试命令：
```bash
# 查看详细错误
python DeepSeek-R1-Distill-Llama-8B-app.py --log-level debug

# 测试CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 测试vLLM
python -c "import vllm; print('vLLM imported successfully')"
``` 