#!/bin/bash

# DeepSeek-R1 H20*8 服务器部署脚本
# 作者: AI Assistant
# 描述: 自动化部署DeepSeek-R1-0528模型到H20*8服务器

set -e

echo "🚀 开始部署 DeepSeek-R1-0528 到 H20*8 服务器..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker 未安装，请先安装 Docker${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker 已安装${NC}"
}

# 检查Docker Compose是否安装
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose 未安装，请先安装 Docker Compose${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker Compose 已安装${NC}"
}

# 检查NVIDIA Docker运行时
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        echo -e "${RED}❌ NVIDIA Docker 运行时未正确配置${NC}"
        echo "请确保安装了 nvidia-docker2 并重启 Docker 服务"
        exit 1
    fi
    echo -e "${GREEN}✅ NVIDIA Docker 运行时正常${NC}"
}

# 检查GPU数量
check_gpu_count() {
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${YELLOW}🔍 检测到 ${GPU_COUNT} 个GPU${NC}"
    
    if [ "$GPU_COUNT" -lt 8 ]; then
        echo -e "${YELLOW}⚠️  警告: 检测到的GPU数量少于8个，建议使用8卡H20配置${NC}"
        read -p "是否继续部署? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✅ GPU数量符合要求${NC}"
    fi
}

# 创建必要的目录
create_directories() {
    echo -e "${YELLOW}📁 创建必要的目录...${NC}"
    mkdir -p model_cache
    mkdir -p logs
    mkdir -p config
    mkdir -p monitoring
    echo -e "${GREEN}✅ 目录创建完成${NC}"
}

# 创建监控配置
create_monitoring_config() {
    echo -e "${YELLOW}📊 创建监控配置...${NC}"
    
    # Prometheus 配置
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'deepseek-r1'
    static_configs:
      - targets: ['deepseek-r1:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
EOF

    echo -e "${GREEN}✅ 监控配置创建完成${NC}"
}

# 下载模型（可选，首次运行时模型会自动下载）
download_model() {
    echo -e "${YELLOW}🤖 检查模型缓存...${NC}"
    
    if [ ! -d "model_cache/models--deepseek-ai--DeepSeek-R1-0528" ]; then
        echo -e "${YELLOW}📥 首次运行，模型将在启动时自动下载...${NC}"
        echo -e "${YELLOW}⚠️  注意: 模型大小约70GB，请确保网络连接稳定且磁盘空间充足${NC}"
    else
        echo -e "${GREEN}✅ 模型缓存已存在${NC}"
    fi
}

# 构建Docker镜像
build_image() {
    echo -e "${YELLOW}🔨 构建Docker镜像...${NC}"
    docker-compose build --no-cache
    echo -e "${GREEN}✅ 镜像构建完成${NC}"
}

# 启动服务
start_services() {
    echo -e "${YELLOW}🚀 启动服务...${NC}"
    docker-compose up -d
    echo -e "${GREEN}✅ 服务启动完成${NC}"
}

# 检查服务状态
check_service_status() {
    echo -e "${YELLOW}🔍 检查服务状态...${NC}"
    
    # 等待服务启动
    echo "等待服务启动..."
    sleep 30
    
    # 检查容器状态
    if docker-compose ps | grep -q "Up"; then
        echo -e "${GREEN}✅ 容器正在运行${NC}"
    else
        echo -e "${RED}❌ 容器启动失败${NC}"
        docker-compose logs
        exit 1
    fi
    
    # 检查健康状态
    echo "等待健康检查..."
    sleep 60
    
    HEALTH_CHECK=$(curl -s http://localhost:8000/health || echo "failed")
    if echo "$HEALTH_CHECK" | grep -q "healthy"; then
        echo -e "${GREEN}✅ 服务健康检查通过${NC}"
    else
        echo -e "${RED}❌ 服务健康检查失败${NC}"
        echo "健康检查响应: $HEALTH_CHECK"
    fi
}

# 显示访问信息
show_access_info() {
    echo -e "${GREEN}🎉 部署完成！${NC}"
    echo
    echo -e "${YELLOW}📋 访问信息:${NC}"
    echo "• API 服务: http://localhost:8000"
    echo "• API 文档: http://localhost:8000/docs"
    echo "• 健康检查: http://localhost:8000/health"
    echo "• 模型信息: http://localhost:8000/model-info"
    echo "• Prometheus: http://localhost:9090"
    echo "• Grafana: http://localhost:3000 (admin/admin123)"
    echo
    echo -e "${YELLOW}📝 常用命令:${NC}"
    echo "• 查看日志: docker-compose logs -f deepseek-r1"
    echo "• 重启服务: docker-compose restart"
    echo "• 停止服务: docker-compose down"
    echo "• 查看GPU状态: nvidia-smi"
    echo
    echo -e "${YELLOW}🔍 监控信息:${NC}"
    echo "• GPU使用率监控: nvidia-smi -l 1"
    echo "• 容器资源监控: docker stats"
}

# 主函数
main() {
    echo -e "${GREEN}=== DeepSeek-R1 H20*8 服务器部署脚本 ===${NC}"
    
    check_docker
    check_docker_compose
    check_nvidia_docker
    check_gpu_count
    create_directories
    create_monitoring_config
    download_model
    build_image
    start_services
    check_service_status
    show_access_info
}

# 如果脚本被直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 