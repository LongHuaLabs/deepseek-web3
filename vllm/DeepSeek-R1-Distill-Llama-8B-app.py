#!/usr/bin/env python3
"""
基于vLLM的DeepSeek-R1 Web3量化策略API服务器
优化支持H20*8服务器配置
"""

import os
# 设置HF镜像站，解决国内网络访问问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator
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
from contextlib import asynccontextmanager
import threading
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import queue
import uuid

# 配置日志
os.makedirs("logs", exist_ok=True)
logger.add("logs/vllm_app_{time}.log", rotation="500 MB", retention="10 days")

# 全局变量存储vLLM引擎
vllm_engine = None
model_config = None

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

class StreamingResponse(BaseModel):
    text: str
    finish_reason: Optional[str] = None

class VLLMConfig:
    """vLLM配置类，针对H20*8优化"""
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # 使用8B模型更稳定
        self.tensor_parallel_size = 8  # 使用全部8张H20
        self.gpu_memory_utilization = 0.85  # H20显存利用率
        self.max_model_len = 32768  # 支持长上下文
        self.dtype = "bfloat16"  # 使用bfloat16精度
        self.quantization = None  # H20显存充足，不需要量化
        self.seed = 42
        self.trust_remote_code = True
        self.max_num_seqs = 256  # 支持更多并发请求
        self.max_num_batched_tokens = 8192
        # 性能优化参数
        self.use_v2_block_manager = True
        self.block_size = 16
        self.swap_space = 4  # GB
        self.enforce_eager = False  # 使用CUDA图优化

async def get_gpu_info():
    """获取GPU使用情况"""
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
           # 计算显存使用百分比
            memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0
            
            gpu_info.append({
                "id": gpu.id,
                "name": gpu.name,
                "memory_used": f"{gpu.memoryUsed}MB",
                "memory_total": f"{gpu.memoryTotal}MB",
                "memory_percent": f"{memory_percent:.1f}%",
                "temperature": f"{gpu.temperature}°C",
                "load": f"{gpu.load * 100:.1f}%"
            })
        return gpu_info
    except Exception as e:
        logger.error(f"获取GPU信息失败: {e}")
        return []

async def initialize_vllm_engine():
    """初始化vLLM异步引擎"""
    global vllm_engine, model_config
    
    try:
        logger.info("开始初始化vLLM引擎...")
        model_config = VLLMConfig()
        
        # 检查GPU数量
        gpu_count = torch.cuda.device_count()
        if gpu_count < model_config.tensor_parallel_size:
            logger.warning(f"可用GPU数量({gpu_count})少于配置的tensor_parallel_size({model_config.tensor_parallel_size})")
            model_config.tensor_parallel_size = gpu_count
        
        # 配置引擎参数
        engine_args = AsyncEngineArgs(
            model=model_config.model_name,
            tensor_parallel_size=model_config.tensor_parallel_size,
            gpu_memory_utilization=model_config.gpu_memory_utilization,
            max_model_len=model_config.max_model_len,
            dtype=model_config.dtype,
            quantization=model_config.quantization,
            seed=model_config.seed,
            trust_remote_code=model_config.trust_remote_code,
            max_num_seqs=model_config.max_num_seqs,
            max_num_batched_tokens=model_config.max_num_batched_tokens,
            use_v2_block_manager=model_config.use_v2_block_manager,
            block_size=model_config.block_size,
            swap_space=model_config.swap_space,
            enforce_eager=model_config.enforce_eager,
        )
        
        # 创建异步引擎
        vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        logger.info("vLLM引擎初始化成功!")
        
        # 打印配置信息
        logger.info(f"模型: {model_config.model_name}")
        logger.info(f"并行度: {model_config.tensor_parallel_size}")
        logger.info(f"最大序列长度: {model_config.max_model_len}")
        logger.info(f"显存利用率: {model_config.gpu_memory_utilization}")
        
        # 打印GPU使用情况
        gpu_info = await get_gpu_info()
        for gpu in gpu_info:
            logger.info(f"GPU {gpu['id']} ({gpu['name']}): {gpu['memory_percent']} 内存使用")
        
        return True
        
    except Exception as e:
        logger.error(f"vLLM引擎初始化失败: {str(e)}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化vLLM引擎
    success = await initialize_vllm_engine()
    if not success:
        logger.error("vLLM引擎初始化失败，服务启动失败")
        raise RuntimeError("vLLM引擎初始化失败")
    
    yield
    
    # 关闭时清理资源
    logger.info("清理vLLM引擎资源...")
    if vllm_engine:
        # vLLM引擎会自动清理资源
        pass
    logger.info("资源清理完成")

app = FastAPI(
    title="vLLM DeepSeek-R1 Web3 Quant API Server",
    description="基于vLLM和DeepSeek-R1的高性能Web3量化策略生成服务，优化支持H20*8服务器",
    version="3.0.0",
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
            prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
        elif msg.role == "user":
            prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif msg.role == "assistant":
            prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
    
    prompt += "<|im_start|>assistant\n"
    return prompt

async def generate_with_vllm_stream(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 0.9,
    frequency_penalty: float = 0.0
):
    """使用vLLM生成流式响应"""
    if not vllm_engine:
        raise HTTPException(status_code=503, detail="vLLM引擎未初始化")
    
    try:
        # 配置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=1.0 + frequency_penalty,
            use_beam_search=False,
            skip_special_tokens=True,
        )
        
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 流式生成
        result_generator = vllm_engine.generate(prompt, sampling_params, request_id)
        
        async for request_output in result_generator:
            if request_output.outputs:
                yield request_output.outputs[0].text
    
    except Exception as e:
        logger.error(f"vLLM流式生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

async def generate_with_vllm(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 0.9,
    frequency_penalty: float = 0.0
) -> str:
    """使用vLLM生成响应"""
    if not vllm_engine:
        raise HTTPException(status_code=503, detail="vLLM引擎未初始化")
    
    try:
        # 配置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=1.0 + frequency_penalty,
            use_beam_search=False,
            skip_special_tokens=True,
        )
        
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        # 非流式生成
        result_generator = vllm_engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in result_generator:
            final_output = request_output
        
        if final_output and final_output.outputs:
            response = final_output.outputs[0].text
            generation_time = time.time() - start_time
            
            logger.info(f"生成完成，耗时: {generation_time:.2f}秒")
            logger.info(f"输入tokens: {len(prompt.split())}, 输出tokens: {len(response.split())}")
            
            return response.strip()
        else:
            raise HTTPException(status_code=500, detail="生成失败：无输出")
    
    except Exception as e:
        logger.error(f"vLLM生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """聊天接口"""
    prompt = format_chat_prompt(request.messages)
    
    if request.stream:
        # 流式响应
        from fastapi.responses import StreamingResponse
        
        async def generate_stream():
            async for chunk in generate_with_vllm_stream(
                prompt, 
                request.max_tokens, 
                request.temperature, 
                request.top_p, 
                request.frequency_penalty
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/plain")
    else:
        # 普通响应
        response = await generate_with_vllm(
            prompt, 
            request.max_tokens, 
            request.temperature, 
            request.top_p, 
            request.frequency_penalty
        )
        
        return {
            "response": response,
            "model": model_config.model_name,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
            },
            "timestamp": datetime.now().isoformat()
        }

@app.post("/generate-quant-strategy")
async def generate_quant_strategy(request: QuantStrategyRequest):
    """生成Web3量化策略"""
    
    # 构建专门的量化策略提示词
    system_prompt = """你是一个专业的Web3量化交易策略专家，拥有丰富的DeFi、CEX交易经验。请基于提供的信息生成详细、可执行的量化交易策略。"""
    
    user_prompt = f"""请为以下场景设计一个量化交易策略：

**交易对象**: {request.target_asset}
**市场数据**: {request.market_data}
**策略类型**: {request.strategy_type}
**风险等级**: {request.risk_level}
**时间框架**: {request.timeframe}
**初始资金**: ${request.capital:,.2f}

请提供以下内容：

1. **策略概述**
   - 策略核心逻辑和原理
   - 适用的市场环境和条件
   - 预期收益率和风险水平

2. **技术指标和交易信号**
   - 入场信号的具体条件
   - 出场信号的判断标准
   - 止损止盈的设置方法

3. **风险管理体系**
   - 仓位管理规则
   - 最大回撤控制机制
   - 关键风险评估指标

4. **参数配置和优化**
   - 策略的具体参数设置
   - 参数优化的建议方法
   - 市场适应性调整

5. **Python实现代码**
   - 使用ccxt库的完整交易代码
   - 包含完整的回测框架
   - 实盘交易的接口实现

6. **性能预期和评估**
   - 预期年化收益率
   - 最大回撤预估
   - 夏普比率和其他指标估算

请确保策略具有实用性和可操作性。"""

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt)
    ]
    
    prompt = format_chat_prompt(messages)
    
    response = await generate_with_vllm(prompt, 4000, 0.3)
    
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
        "model": model_config.model_name,
        "timestamp": datetime.now().isoformat(),
        "strategy_id": f"{request.strategy_type}_{int(time.time())}"
    }

@app.post("/analyze-market")
async def analyze_market(market_data: Dict[str, Any]):
    """市场分析接口"""
    
    system_prompt = """你是一个专业的Web3市场分析师，擅长技术分析和基本面分析。请基于提供的市场数据给出专业、准确的分析见解。"""
    
    user_prompt = f"""请分析以下市场数据：

{json.dumps(market_data, indent=2, ensure_ascii=False)}

请提供详细分析，包括：
1. **市场趋势分析** - 当前趋势方向和强度
2. **关键价位识别** - 重要支撑位和阻力位
3. **交易机会识别** - 潜在的交易机会和时机
4. **风险评估** - 主要风险因素和注意事项
5. **操作建议** - 具体的交易建议和策略

请基于技术分析和市场情绪给出专业判断。"""

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt)
    ]
    
    prompt = format_chat_prompt(messages)
    response = await generate_with_vllm(prompt, 2000, 0.4)
    
    return {
        "analysis": response,
        "market_data": market_data,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/batch-strategy")
async def batch_generate_strategies(requests: List[QuantStrategyRequest]):
    """批量生成量化策略"""
    
    results = []
    for req in requests:
        try:
            strategy_result = await generate_quant_strategy(req)
            results.append({
                "success": True,
                "strategy": strategy_result
            })
        except Exception as e:
            logger.error(f"批量策略生成失败: {e}")
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

@app.post("/optimize-strategy")
async def optimize_strategy(strategy_params: Dict[str, Any]):
    """策略参数优化接口"""
    
    system_prompt = """你是一个量化策略优化专家，擅长策略参数调优和性能提升。请基于提供的策略参数和历史表现数据，给出优化建议。"""
    
    user_prompt = f"""请优化以下量化策略参数：

{json.dumps(strategy_params, indent=2, ensure_ascii=False)}

请提供：
1. **参数优化建议** - 具体的参数调整建议
2. **优化理由** - 为什么这样调整的原理
3. **预期改进** - 优化后的预期性能提升
4. **风险评估** - 参数调整可能带来的风险
5. **测试建议** - 如何验证优化效果"""

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt)
    ]
    
    prompt = format_chat_prompt(messages)
    response = await generate_with_vllm(prompt, 2000, 0.4)
    
    return {
        "optimization": response,
        "original_params": strategy_params,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    gpu_info = await get_gpu_info()
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    return {
        "status": "healthy",
        "vllm_engine_loaded": vllm_engine is not None,
        "model_name": model_config.model_name if model_config else None,
        "tensor_parallel_size": model_config.tensor_parallel_size if model_config else None,
        "gpu_count": torch.cuda.device_count(),
        "gpu_info": gpu_info,
        "cpu_usage": f"{cpu_percent}%",
        "memory_usage": f"{memory.percent}%",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
async def model_info():
    """模型详细信息"""
    if not vllm_engine or not model_config:
        raise HTTPException(status_code=503, detail="vLLM引擎未加载")
    
    gpu_info = await get_gpu_info()
    
    return {
        "model_name": model_config.model_name,
        "tensor_parallel_size": model_config.tensor_parallel_size,
        "max_model_len": model_config.max_model_len,
        "gpu_memory_utilization": model_config.gpu_memory_utilization,
        "dtype": model_config.dtype,
        "max_num_seqs": model_config.max_num_seqs,
        "gpu_info": gpu_info,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus格式的监控指标"""
    gpu_info = await get_gpu_info()
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    metrics = []
    metrics.append(f"cpu_usage_percent {cpu_percent}")
    metrics.append(f"memory_usage_percent {memory.percent}")
    metrics.append(f"vllm_engine_loaded {1 if vllm_engine else 0}")
    
    for gpu in gpu_info:
        gpu_id = gpu['id']
        metrics.append(f"gpu_memory_percent{{gpu=\"{gpu_id}\"}} {gpu['memory_percent'].rstrip('%')}")
        metrics.append(f"gpu_temperature{{gpu=\"{gpu_id}\"}} {gpu['temperature'].rstrip('°C')}")
        metrics.append(f"gpu_load_percent{{gpu=\"{gpu_id}\"}} {gpu['load'].rstrip('%')}")
    
    return "\n".join(metrics)

@app.post("/generate")
async def generate(request: Dict[str, Any]):
    """通用生成接口"""
    prompt = request.get("prompt", "")
    max_tokens = request.get("max_tokens", 2048)
    temperature = request.get("temperature", 0.7)
    top_p = request.get("top_p", 0.9)
    
    response = await generate_with_vllm(prompt, max_tokens, temperature, top_p)
    
    return {
        "generated_text": response,
        "model": model_config.model_name,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--log-level", default="info", help="日志级别")
    parser.add_argument("--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="模型名称")
    parser.add_argument("--tensor-parallel-size", type=int, default=8, help="张量并行大小")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU显存利用率")
    
    args = parser.parse_args()
    
    # 动态调整tensor_parallel_size
    available_gpus = torch.cuda.device_count()
    if args.tensor_parallel_size > available_gpus:
        logger.warning(f"tensor_parallel_size ({args.tensor_parallel_size}) 大于可用GPU数量 ({available_gpus})，调整为 {available_gpus}")
        args.tensor_parallel_size = available_gpus
    
    logger.info(f"启动vLLM Web3量化策略服务器...")
    logger.info(f"服务地址: {args.host}:{args.port}")
    logger.info(f"模型: {args.model_name}")
    logger.info(f"张量并行: {args.tensor_parallel_size}")
    logger.info(f"GPU显存利用率: {args.gpu_memory_utilization}")
    
    # 更新全局配置
    if model_config:
        model_config.model_name = args.model_name
        model_config.tensor_parallel_size = args.tensor_parallel_size
        model_config.gpu_memory_utilization = args.gpu_memory_utilization
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True,
        reload=False  # 生产环境禁用自动重载
    )