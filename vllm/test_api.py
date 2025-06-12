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
