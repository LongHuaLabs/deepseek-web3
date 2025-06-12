# DeepSeek-R1-Distill-Llama-8B 模型下载与部署工具

这是一个用于在Mac上下载DeepSeek-R1-Distill-Llama-8B模型，并在服务器上部署vLLM服务的完整工具包。

## 🚀 快速开始

### 1. Mac端模型下载

#### 安装依赖
```bash
pip install -r requirements.txt
```

#### 快速下载（推荐）
```bash
python quick_download.py
```

#### 高级下载（更多选项）
```bash
python download_model.py --help
```

常用参数：
- `--cache-dir`: 指定下载目录（默认：./models）
- `--max-workers`: 并发下载线程数（默认：4）
- `--token`: HuggingFace访问令牌（如果需要）

### 2. 模型传输到服务器

下载完成后，将模型压缩并上传到服务器：

```bash
# 压缩模型文件
tar -czf deepseek-model.tar.gz -C ./models .

# 上传到服务器
scp deepseek-model.tar.gz user@your-server:/path/to/models/

# 在服务器上解压
ssh user@your-server
cd /path/to/models/
tar -xzf deepseek-model.tar.gz
```

### 3. 服务器端部署

将`server_deploy.py`复制到服务器，然后运行：

```bash
# 基本部署
python server_deploy.py --model-path /path/to/models/deepseek-ai--DeepSeek-R1-Distill-Llama-8B

# 创建systemd服务
python server_deploy.py --model-path /path/to/models/deepseek-ai--DeepSeek-R1-Distill-Llama-8B --create-service

# 自定义配置
python server_deploy.py \
    --model-path /path/to/models/deepseek-ai--DeepSeek-R1-Distill-Llama-8B \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 32768
```

## 📁 文件说明

### Mac端文件
- `requirements.txt`: Python依赖包
- `quick_download.py`: 快速下载脚本（推荐使用）
- `download_model.py`: 完整功能的下载工具
- `README.md`: 使用说明文档

### 服务器端文件
- `server_deploy.py`: 服务器部署脚本
- `test_vllm.py`: 服务测试客户端（由部署脚本生成）
- `start_vllm.sh`: 启动脚本（由部署脚本生成）
- `vllm_config.json`: vLLM配置文件（由部署脚本生成）
- `deepseek-vllm.service`: Systemd服务文件（可选生成）

## 🔧 服务器配置要求

根据您的服务器配置：
- **GPU**: 8 × NVIDIA H20 (95GB显存/块)
- **CUDA**: 12.4
- **驱动**: 550.90.07
- **系统**: Ubuntu 22.04 LTS

推荐配置：
- `tensor-parallel-size`: 8 (使用所有8块GPU)
- `gpu-memory-utilization`: 0.9 (90%显存利用率)
- `max-model-len`: 32768 (最大序列长度)

## 📊 模型信息

- **模型名称**: DeepSeek-R1-Distill-Llama-8B
- **模型大小**: 约15-20GB
- **架构**: Llama-based
- **用途**: 通用对话和推理

## 🛠️ 使用示例

### 启动服务
```bash
# 使用生成的启动脚本
./start_vllm.sh

# 或者直接使用vLLM命令
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9
```

### 测试服务
```bash
# 使用生成的测试脚本
python test_vllm.py

# 或者使用curl测试
curl -X POST "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-r1-distill-llama-8b",
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己。"}
        ],
        "max_tokens": 100
    }'
```

### API调用示例
```python
import requests

url = "http://your-server:8000/v1/chat/completions"
payload = {
    "model": "deepseek-r1-distill-llama-8b",
    "messages": [
        {"role": "user", "content": "解释一下量子计算的基本原理"}
    ],
    "max_tokens": 500,
    "temperature": 0.7
}

response = requests.post(url, json=payload)
result = response.json()
print(result['choices'][0]['message']['content'])
```

## 🔍 故障排除

### 下载问题
1. **网络连接失败**: 检查网络连接，考虑使用代理
2. **磁盘空间不足**: 确保有足够的存储空间（至少30GB）
3. **权限问题**: 确保对下载目录有写权限

### 部署问题
1. **GPU检测失败**: 检查CUDA驱动和nvidia-smi命令
2. **内存不足**: 调整`gpu-memory-utilization`参数
3. **端口占用**: 更改端口或停止占用端口的进程

### 服务问题
1. **启动失败**: 检查模型路径和权限
2. **推理慢**: 调整`tensor-parallel-size`和`max-model-len`
3. **内存溢出**: 减少`max-model-len`或`gpu-memory-utilization`

## 📝 日志和监控

### 查看服务日志
```bash
# 如果使用systemd服务
sudo journalctl -u deepseek-vllm -f

# 如果使用启动脚本
tail -f vllm.log
```

### 监控GPU使用
```bash
# 实时监控
nvidia-smi -l 1

# 查看详细信息
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

## 🔄 更新和维护

### 更新模型
1. 下载新版本模型
2. 停止服务
3. 替换模型文件
4. 重启服务

### 更新vLLM
```bash
pip install --upgrade vllm
```

## 📞 支持

如果遇到问题，请检查：
1. 系统要求是否满足
2. 依赖是否正确安装
3. 模型文件是否完整
4. 网络连接是否正常

## 📄 许可证

本工具遵循MIT许可证。模型使用请遵循DeepSeek的许可证条款。 