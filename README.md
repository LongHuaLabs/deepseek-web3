# DeepSeek-R1 Web3 量化策略生成服务

基于 DeepSeek-R1-0528 模型的 Web3 量化交易策略生成和聊天服务，专为英伟达 H20*8 服务器优化。

## 🚀 特性

- **🤖 大模型推理**: 基于 DeepSeek-R1-0528 模型，671亿参数
- **📈 量化策略生成**: 专业的 Web3 量化交易策略生成
- **💬 智能聊天**: 支持多轮对话和上下文理解
- **🖥️ 多GPU支持**: 针对 H20*8 服务器优化的设备映射
- **📊 实时监控**: 集成 Prometheus + Grafana 监控
- **🔧 自动化部署**: 一键部署脚本和 Docker 容器化
- **⚡ 高性能**: Flash Attention 2 加速和内存优化

## 📋 系统要求

### 硬件要求
- **GPU**: 英伟达 H20 × 8 (推荐) 或其他高端 GPU
- **显存**: 每卡至少 24GB
- **内存**: 64GB+ RAM 推荐
- **存储**: 200GB+ 可用空间 (模型约 70GB)

### 软件要求
- **操作系统**: Ubuntu 20.04/22.04 或 CentOS 7/8
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Driver**: 525.60.13+
- **CUDA**: 11.8+ / 12.0+

## 🛠️ 快速部署

### 1. 环境准备

```bash
# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装 Docker Compose
sudo apt install docker-compose-plugin

# 安装 NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. 克隆项目

```bash
git clone <repository-url>
cd deepseek-web3
```

### 3. 一键部署

```bash
# 给部署脚本执行权限
chmod +x deploy.sh

# 运行部署脚本
./deploy.sh
```

### 4. 验证部署

```bash
# 安装测试依赖
pip install requests

# 运行API测试
python test_api.py
```

## 📊 监控面板

部署完成后，您可以访问：

- **API 服务**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)

## 🔧 API 使用示例

### 聊天接口

```python
import requests

# 聊天请求
response = requests.post("http://localhost:8000/chat", json={
    "messages": [
        {"role": "user", "content": "解释一下DeFi流动性挖矿的原理"}
    ],
    "max_tokens": 1000,
    "temperature": 0.7
})

print(response.json()["response"])
```

### 量化策略生成

```python
# 策略生成请求
response = requests.post("http://localhost:8000/generate-quant-strategy", json={
    "market_data": "BTC/USDT价格在42000-45000区间震荡，RSI=65",
    "strategy_type": "网格交易",
    "risk_level": "中",
    "timeframe": "1h",
    "target_asset": "BTC/USDT",
    "capital": 10000
})

strategy = response.json()["strategy"]
print(strategy)
```

### 市场分析

```python
# 市场分析请求
market_data = {
    "symbol": "ETH/USDT",
    "price": 2500,
    "volume_24h": 800000000,
    "rsi": 58,
    "ma_20": 2480
}

response = requests.post("http://localhost:8000/analyze-market", json=market_data)
analysis = response.json()["analysis"]
print(analysis)
```

## 🎛️ 配置参数

### 模型配置

在 `app.py` 中的 `ModelConfig` 类：

```python
class ModelConfig:
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-R1-0528"
        self.max_memory_per_gpu = "24GB"  # 根据你的GPU显存调整
        self.torch_dtype = torch.bfloat16
        self.load_in_8bit = False  # 显存不足时设为True
```

### GPU 设备映射

针对不同GPU数量，系统会自动优化设备映射：

- **8卡配置**: 层均匀分布到8个GPU
- **4卡配置**: 自动调整分布策略
- **单卡配置**: 使用 `device_map="auto"`

## 📈 性能优化

### 内存优化
- 使用 `bfloat16` 数据类型
- Flash Attention 2 加速
- 合理的设备映射策略
- 模型权重缓存

### 推理加速
- 多GPU并行推理
- KV-Cache 优化
- 批量处理支持

## 🔍 故障排除

### 常见问题

1. **GPU内存不足**
   ```bash
   # 检查GPU使用情况
   nvidia-smi
   
   # 清理GPU内存
   docker-compose restart deepseek-r1
   ```

2. **模型加载失败**
   ```bash
   # 查看详细日志
   docker-compose logs -f deepseek-r1
   
   # 检查磁盘空间
   df -h
   ```

3. **网络连接问题**
   ```bash
   # 测试网络连接
   curl -I https://huggingface.co
   
   # 设置代理（如需要）
   export HF_ENDPOINT=https://hf-mirror.com
   ```

### 性能调优

1. **提高并发性能**
   ```yaml
   # docker-compose.yml 中增加内存限制
   deploy:
     resources:
       limits:
         memory: 64G
   ```

2. **优化模型加载速度**
   ```bash
   # 预下载模型到本地缓存
   huggingface-cli download deepseek-ai/DeepSeek-R1-0528
   ```

## 📊 监控指标

系统提供以下监控指标：

- **GPU 使用率**: 每个GPU的利用率和内存使用
- **CPU 使用率**: 系统CPU负载
- **内存使用**: 系统内存占用
- **推理延迟**: API响应时间
- **错误率**: 请求失败率

## 🚨 生产环境建议

1. **安全配置**
   - 设置防火墙规则
   - 使用 HTTPS 证书
   - 配置 API 认证

2. **高可用性**
   - 配置负载均衡
   - 设置自动重启策略
   - 数据备份机制

3. **性能监控**
   - 设置告警阈值
   - 定期性能评估
   - 日志轮转策略

## 🤝 支持与贡献

如果遇到问题或有改进建议，请：

1. 查看文档和FAQ
2. 提交 Issue
3. 贡献代码 (Pull Request)

## 📝 许可证

本项目采用 MIT 许可证。

---

**注意**: 本服务仅供研究和学习使用，实际交易请谨慎评估风险。 