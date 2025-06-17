#!/bin/bash

# stop_server.sh - 停止后台运行的 vLLM 服务

PID_FILE="server.pid"
LOG_FILE="output.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")

    if ps -p $PID > /dev/null; then
        echo "正在尝试优雅地停止服务器 (PID: $PID)..."
        kill $PID

        # 等待最多 10 秒
        for i in {1..10}; do
            if ! ps -p $PID > /dev/null; then
                echo "服务器已成功停止。"
                rm -f "$PID_FILE"
                exit 0
            fi
            sleep 1
        done

        echo "服务器仍未响应，执行强制终止..."
        kill -9 $PID
        if ps -p $PID > /dev/null 2>&1; then
            echo "ERROR: 无法终止进程 (PID: $PID)"
            exit 1
        else
            echo "服务器已被强制终止。"
            rm -f "$PID_FILE"
            exit 0
        fi
    else
        echo "进程文件存在但进程不存在 (PID: $PID)，可能已停止。"
        rm -f "$PID_FILE"
        exit 0
    fi
else
    echo "未找到进程文件 ($PID_FILE)，服务器可能未运行。"
    exit 0
fi