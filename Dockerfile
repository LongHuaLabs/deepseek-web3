# 使用更新的PyTorch镜像，支持最新的CUDA版本
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# 设置工作目录
WORKDIR /app

# 安装必要的系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    htop \
    nvtop \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 创建模型缓存目录
RUN mkdir -p /app/model_cache

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码到容器中
COPY . .

# 创建用于存储模型权重的挂载点
VOLUME ["/app/model_cache", "/app/logs"]

# 暴露端口用于FastAPI服务
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令，增加内存配置
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
