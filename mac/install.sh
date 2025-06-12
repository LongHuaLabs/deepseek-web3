#!/bin/bash

# DeepSeek-R1-Distill-Llama-8B 模型下载工具安装脚本

echo "🚀 DeepSeek-R1-Distill-Llama-8B 模型下载工具"
echo "=" * 60

# # 检查Python版本
# python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
# if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 1 ]]; then
#     echo "✅ Python版本: $(python3 --version)"
# else
#     echo "❌ 需要Python 3.8或更高版本"
#     exit 1
# fi

# # 检查pip
# if command -v pip3 &> /dev/null; then
#     echo "✅ pip3已安装"
# else
#     echo "❌ 未找到pip3"
#     exit 1
# fi

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境？(y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv deepseek_env
    source deepseek_env/bin/activate
    echo "✅ 虚拟环境已创建并激活"
    echo "💡 下次使用前请运行: source deepseek_env/bin/activate"
fi

# 安装依赖
echo "📦 安装Python依赖..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ 依赖安装成功"
else
    echo "❌ 依赖安装失败"
    exit 1
fi

# 设置执行权限
chmod +x quick_download.py
chmod +x download_model.py

echo ""
echo "🎉 安装完成！"
echo "=" * 60
echo "📝 使用方法:"
echo "1. 快速下载: python3 quick_download.py"
echo "2. 高级下载: python3 download_model.py --help"
echo "3. 查看文档: cat README.md"
echo ""
echo "💡 提示:"
echo "- 确保有足够的磁盘空间（至少30GB）"
echo "- 下载过程支持断点续传"
echo "- 如有网络问题，可重新运行下载脚本" 