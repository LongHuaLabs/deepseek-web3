#!/usr/bin/env python3
"""
DeepSeek-R1 API 测试脚本
用于验证部署是否成功并测试各种功能
"""

import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """测试健康检查接口"""
    print("🔍 测试健康检查...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查通过")
            print(f"   状态: {data['status']}")
            print(f"   模型已加载: {data['model_loaded']}")
            print(f"   GPU数量: {data['gpu_count']}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查错误: {e}")
        return False

def test_model_info():
    """测试模型信息接口"""
    print("\n🤖 测试模型信息...")
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 模型信息获取成功")
            print(f"   模型名称: {data['model_name']}")
            print(f"   参数量: {data['parameters']}")
            print(f"   数据类型: {data['dtype']}")
            return True
        else:
            print(f"❌ 模型信息获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 模型信息错误: {e}")
        return False

def test_chat():
    """测试聊天接口"""
    print("\n💬 测试聊天功能...")
    try:
        chat_request = {
            "messages": [
                {"role": "user", "content": "你好，请简单介绍一下Web3量化交易的基本概念。"}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        print("发送聊天请求...")
        response = requests.post(
            f"{API_BASE_URL}/chat", 
            json=chat_request, 
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 聊天功能测试成功")
            print(f"   响应长度: {len(data['response'])} 字符")
            print(f"   响应内容预览: {data['response'][:200]}...")
            return True
        else:
            print(f"❌ 聊天功能测试失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 聊天功能错误: {e}")
        return False

def test_quant_strategy():
    """测试量化策略生成"""
    print("\n📈 测试量化策略生成...")
    try:
        strategy_request = {
            "market_data": "BTC/USDT在过去24小时内价格波动较大，RSI指标显示超买状态",
            "strategy_type": "均值回归",
            "risk_level": "中",
            "timeframe": "1h",
            "target_asset": "BTC/USDT",
            "capital": 10000
        }
        
        print("发送策略生成请求...")
        response = requests.post(
            f"{API_BASE_URL}/generate-quant-strategy", 
            json=strategy_request, 
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 量化策略生成成功")
            print(f"   策略ID: {data['strategy_id']}")
            print(f"   策略长度: {len(data['strategy'])} 字符")
            print(f"   策略预览: {data['strategy'][:300]}...")
            return True
        else:
            print(f"❌ 量化策略生成失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 量化策略生成错误: {e}")
        return False

def test_market_analysis():
    """测试市场分析功能"""
    print("\n📊 测试市场分析功能...")
    try:
        market_data = {
            "symbol": "BTC/USDT",
            "price": 45000,
            "volume_24h": 1500000000,
            "change_24h": 2.5,
            "rsi": 65,
            "ma_20": 44500,
            "ma_50": 43800
        }
        
        print("发送市场分析请求...")
        response = requests.post(
            f"{API_BASE_URL}/analyze-market", 
            json=market_data, 
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 市场分析功能测试成功")
            print(f"   分析长度: {len(data['analysis'])} 字符")
            print(f"   分析预览: {data['analysis'][:300]}...")
            return True
        else:
            print(f"❌ 市场分析功能测试失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 市场分析功能错误: {e}")
        return False

def test_performance():
    """测试性能指标"""
    print("\n⚡ 测试性能指标...")
    try:
        # 简单的性能测试
        start_time = time.time()
        
        chat_request = {
            "messages": [
                {"role": "user", "content": "请用一句话解释DeFi的核心概念。"}
            ],
            "max_tokens": 100,
            "temperature": 0.5
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat", 
            json=chat_request, 
            timeout=30
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            print(f"✅ 性能测试完成")
            print(f"   响应时间: {response_time:.2f} 秒")
            
            if response_time < 10:
                print("   🚀 响应速度: 优秀")
            elif response_time < 20:
                print("   ⚡ 响应速度: 良好")
            else:
                print("   🐌 响应速度: 需要优化")
            
            return True
        else:
            print(f"❌ 性能测试失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 性能测试错误: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 DeepSeek-R1 API 测试开始...\n")
    
    tests = [
        ("健康检查", test_health_check),
        ("模型信息", test_model_info),
        ("聊天功能", test_chat),
        ("量化策略生成", test_quant_strategy),
        ("市场分析", test_market_analysis),
        ("性能测试", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 打印测试结果总结
    print("\n" + "="*50)
    print("🏁 测试结果总结:")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！DeepSeek-R1 部署成功！")
        return True
    else:
        print("⚠️  部分测试失败，请检查服务状态")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 