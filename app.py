from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import torch
import os
import json
import time
import asyncio
from datetime import datetime
from loguru import logger
import psutil
import GPUtil
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from accelerate import Accelerator
from contextlib import asynccontextmanager
import threading

# 配置日志
logger.add("logs/app_{time}.log", rotation="500 MB", retention="10 days")

# 全局变量存储模型
model = None
tokenizer = None
accelerator = None
device_map = None

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    top_p: Optional[float] = 0.9
    frequency_penalty: Optional[float] = 0.0

class QuantStrategyRequest(BaseModel):
    market_data: str  # 市场数据描述
    strategy_type: str  # 策略类型：趋势跟踪、均值回归、套利、网格等
    risk_level: str  # 风险等级：低、中、高
    timeframe: str  # 时间框架：1m、5m、15m、1h、4h、1d
    target_asset: Optional[str] = "BTC/USDT"  # 目标资产
    capital: Optional[float] = 10000  # 投入资金

class ModelConfig:
    """模型配置类"""
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-R1-0528"
        self.cache_dir = os.getenv("TRANSFORMERS_CACHE", "/app/model_cache")
        self.max_memory_per_gpu = "24GB"  # H20每卡24GB显存
        self.torch_dtype = torch.bfloat16
        self.load_in_8bit = False  # H20显存充足，不需要量化
        self.trust_remote_code = True

def get_optimal_device_map():
    """为H20*8服务器优化设备映射"""
    gpu_count = torch.cuda.device_count()
    logger.info(f"检测到 {gpu_count} 个GPU")
    
    if gpu_count >= 8:
        # 8卡H20的设备映射策略
        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0, "model.layers.1": 0, "model.layers.2": 0, "model.layers.3": 0,
            "model.layers.4": 1, "model.layers.5": 1, "model.layers.6": 1, "model.layers.7": 1,
            "model.layers.8": 2, "model.layers.9": 2, "model.layers.10": 2, "model.layers.11": 2,
            "model.layers.12": 3, "model.layers.13": 3, "model.layers.14": 3, "model.layers.15": 3,
            "model.layers.16": 4, "model.layers.17": 4, "model.layers.18": 4, "model.layers.19": 4,
            "model.layers.20": 5, "model.layers.21": 5, "model.layers.22": 5, "model.layers.23": 5,
            "model.layers.24": 6, "model.layers.25": 6, "model.layers.26": 6, "model.layers.27": 6,
            "model.norm": 7, "lm_head": 7
        }
        return device_map
    else:
        return "auto"

async def monitor_gpu_usage():
    """监控GPU使用情况"""
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            "id": gpu.id,
            "name": gpu.name,
            "memory_used": f"{gpu.memoryUsed}MB",
            "memory_total": f"{gpu.memoryTotal}MB",
            "memory_percent": f"{gpu.memoryPercent:.1f}%",
            "temperature": f"{gpu.temperature}°C",
            "load": f"{gpu.load * 100:.1f}%"
        })
    return gpu_info

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    global model, tokenizer, accelerator, device_map
    
    try:
        logger.info("开始加载 DeepSeek-R1-0528 模型...")
        config = ModelConfig()
        
        # 初始化 Accelerator
        accelerator = Accelerator()
        
        # 加载分词器
        logger.info("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 获取优化的设备映射
        device_map = get_optimal_device_map()
        
        # 配置内存管理
        max_memory = {}
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            max_memory[i] = config.max_memory_per_gpu
        
        logger.info("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
            torch_dtype=config.torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=config.trust_remote_code,
            load_in_8bit=config.load_in_8bit,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        )
        
        logger.info("模型加载成功!")
        logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"设备映射: {device_map}")
        
        # 打印GPU使用情况
        gpu_info = await monitor_gpu_usage()
        for gpu in gpu_info:
            logger.info(f"GPU {gpu['id']} ({gpu['name']}): {gpu['memory_percent']} 内存使用")
        
        yield
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise e
    
    finally:
        # 关闭时清理资源
        logger.info("清理模型资源...")
        if model:
            del model
        if tokenizer:
            del tokenizer
        torch.cuda.empty_cache()
        logger.info("资源清理完成")

app = FastAPI(
    title="DeepSeek-R1 Web3 Quant API Server",
    description="基于DeepSeek-R1-0528的Web3量化策略生成和聊天服务，优化支持H20*8服务器",
    version="2.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """格式化聊天提示词，针对DeepSeek优化"""
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"<|start_system|>{msg.content}<|end_system|>\n"
        elif msg.role == "user":
            prompt += f"<|start_user|>{msg.content}<|end_user|>\n"
        elif msg.role == "assistant":
            prompt += f"<|start_assistant|>{msg.content}<|end_assistant|>\n"
    
    prompt += "<|start_assistant|>"
    return prompt

async def generate_response(
    prompt: str, 
    max_tokens: int, 
    temperature: float, 
    top_p: float = 0.9,
    frequency_penalty: float = 0.0
) -> str:
    """生成模型响应，优化内存使用"""
    try:
        start_time = time.time()
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096
        )
        
        # 将输入移到模型所在设备
        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.0 + frequency_penalty,
            "use_cache": True,
        }
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
        
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        generation_time = time.time() - start_time
        logger.info(f"生成完成，耗时: {generation_time:.2f}秒")
        
        return response.strip()
    
    except Exception as e:
        logger.error(f"生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """聊天接口"""
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    prompt = format_chat_prompt(request.messages)
    response = await generate_response(
        prompt, 
        request.max_tokens, 
        request.temperature, 
        request.top_p, 
        request.frequency_penalty
    )
    
    return {
        "response": response,
        "model": "DeepSeek-R1-0528",
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokenizer.encode(response)),
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate-quant-strategy")
async def generate_quant_strategy(request: QuantStrategyRequest):
    """生成Web3量化策略"""
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 构建专门的量化策略提示词
    prompt = f"""<|start_system|>你是一个专业的Web3量化交易策略专家，拥有丰富的DeFi、CEX交易经验。请基于提供的信息生成详细的量化交易策略。<|end_system|>

<|start_user|>请为以下场景设计一个量化交易策略：

**交易对象**: {request.target_asset}
**市场数据**: {request.market_data}
**策略类型**: {request.strategy_type}
**风险等级**: {request.risk_level}
**时间框架**: {request.timeframe}
**初始资金**: ${request.capital:,.2f}

请提供以下内容：

1. **策略概述**
   - 策略核心逻辑
   - 适用市场环境
   - 预期收益率

2. **技术指标和信号**
   - 入场信号
   - 出场信号
   - 止损止盈设置

3. **风险管理**
   - 仓位管理
   - 最大回撤控制
   - 风险评估指标

4. **参数配置**
   - 具体参数设置
   - 参数优化建议

5. **Python实现代码**
   - 使用ccxt库的完整代码
   - 包含回测框架
   - 实盘交易接口

6. **性能预期**
   - 预期年化收益
   - 最大回撤预估
   - 夏普比率估算<|end_user|>

<|start_assistant|>"""

    response = await generate_response(prompt, 4000, 0.3)
    
    # 记录策略生成日志
    logger.info(f"生成量化策略: {request.strategy_type} - {request.target_asset}")
    
    return {
        "strategy": response,
        "parameters": {
            "target_asset": request.target_asset,
            "strategy_type": request.strategy_type,
            "risk_level": request.risk_level,
            "timeframe": request.timeframe,
            "capital": request.capital
        },
        "model": "DeepSeek-R1-0528",
        "timestamp": datetime.now().isoformat(),
        "strategy_id": f"{request.strategy_type}_{int(time.time())}"
    }

@app.post("/analyze-market")
async def analyze_market(market_data: Dict[str, Any]):
    """市场分析接口"""
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    prompt = f"""<|start_system|>你是一个专业的Web3市场分析师，请分析提供的市场数据并给出专业见解。<|end_system|>

<|start_user|>请分析以下市场数据：

{json.dumps(market_data, indent=2, ensure_ascii=False)}

请提供：
1. 市场趋势分析
2. 关键支撑和阻力位
3. 交易机会识别
4. 风险提示
5. 操作建议<|end_user|>

<|start_assistant|>"""

    response = await generate_response(prompt, 2000, 0.4)
    
    return {
        "analysis": response,
        "market_data": market_data,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    gpu_info = await monitor_gpu_usage()
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_count": torch.cuda.device_count(),
        "gpu_info": gpu_info,
        "cpu_usage": f"{cpu_percent}%",
        "memory_usage": f"{memory.percent}%",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
async def model_info():
    """模型详细信息"""
    if not model:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    gpu_info = await monitor_gpu_usage()
    
    return {
        "model_name": "DeepSeek-R1-0528",
        "parameters": f"{sum(p.numel() for p in model.parameters()):,}",
        "device_map": str(device_map),
        "dtype": str(model.dtype) if hasattr(model, 'dtype') else "bfloat16",
        "gpu_info": gpu_info,
        "cache_dir": os.getenv("TRANSFORMERS_CACHE", "/app/model_cache"),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus格式的监控指标"""
    gpu_info = await monitor_gpu_usage()
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    metrics = []
    metrics.append(f"cpu_usage_percent {cpu_percent}")
    metrics.append(f"memory_usage_percent {memory.percent}")
    metrics.append(f"model_loaded {1 if model else 0}")
    
    for gpu in gpu_info:
        gpu_id = gpu['id']
        metrics.append(f"gpu_memory_percent{{gpu=\"{gpu_id}\"}} {gpu['memory_percent'].rstrip('%')}")
        metrics.append(f"gpu_temperature{{gpu=\"{gpu_id}\"}} {gpu['temperature'].rstrip('°C')}")
        metrics.append(f"gpu_load_percent{{gpu=\"{gpu_id}\"}} {gpu['load'].rstrip('%')}")
    
    return "\n".join(metrics)

@app.post("/batch-strategy")
async def batch_generate_strategies(requests: List[QuantStrategyRequest]):
    """批量生成量化策略"""
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    results = []
    for req in requests:
        try:
            strategy_result = await generate_quant_strategy(req)
            results.append({
                "success": True,
                "strategy": strategy_result
            })
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e),
                "request": req.dict()
            })
    
    return {
        "results": results,
        "total": len(requests),
        "successful": sum(1 for r in results if r["success"]),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    # 配置uvicorn日志
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # 由于模型较大，使用单进程
        log_level="info",
        access_log=True,
        reload=False  # 生产环境禁用自动重载
    )