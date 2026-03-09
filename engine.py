import time
import requests
import pandas as pd
import streamlit as st
from typing import List, Dict, Any

# ======================================
# Hugging Face Inference API 核心封装
# ======================================

def _get_hf_headers() -> Dict[str, str]:
    """从 st.secrets 中读取 HF_TOKEN"""
    try:
        # 确保你在 .streamlit/secrets.toml 里写的是 HF_TOKEN = "xxx"
        token = st.secrets["HF_TOKEN"]
    except Exception as e:
        raise RuntimeError(
            "未在 st.secrets 中找到 HF_TOKEN，请在 .streamlit/secrets.toml 中配置。"
        ) from e

    token = str(token).strip()
    if not token:
        raise RuntimeError("HF_TOKEN 为空，请检查 .streamlit/secrets.toml 配置。")

    return {"Authorization": f"Bearer {token}"}

def _call_hf_text_model(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """
    针对 410/404 报错优化的调用函数
    强制使用最新的 router.huggingface.co 接口
    """
    # 强制去除 model_id 前后可能的空格，防止路径解析失败
    model_id = (model_id or "").strip()
    if not model_id:
        model_id = "Qwen/Qwen2.5-7B-Instruct"

    # 必须使用新的 router 域名，不要加任何多余后缀
    url = f"https://router.huggingface.co/models/{model_id}"

    # 针对 Qwen/Mistral 等模型的标准 Payload 格式
    payload: Dict[str, Any] = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False, # 只要 AI 回答的内容，不要原始 Prompt
        },
    }

    headers = _get_hf_headers()

    # 打印真实请求 URL，方便排查 404/410
    print(f"DEBUG: 请求路径为 {url}")

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        
        # 如果还是报错，输出详细原因到终端供调试
        if resp.status_code != 200:
            print(f"[ERROR] API 响应失败: {resp.status_code}, 内容: {resp.text}")
            raise RuntimeError(
                f"Hugging Face API 调用失败 ({resp.status_code}): {resp.text}"
            )

        data = resp.json()

        # 解析 Hugging Face 常见的两种返回格式
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", str(data)).strip()
        elif isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        
        return str(data)

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"网络连接失败: {str(e)}")

# ============================
# 多智能体协同逻辑
# ============================

# 统一默认模型，确保不再报 404
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

def process_data_agent(data: pd.DataFrame) -> str:
    """大模型 A：数据摘要特征提取"""
    preview_csv = data.head(15).to_csv(index=False)
    prompt = f"你是一个工业数据专家。请简要分析下表的时序特征（压力、温度等趋势），用3句中文总结：\n{preview_csv}"

    start = time.time()
    summary = _call_hf_text_model(DEFAULT_MODEL_ID, prompt)
    elapsed = time.time() - start
    return f"{summary} （分析耗时: {elapsed:.1f}s）"

def process_intent_agent(text: str) -> str:
    """大模型 B：意图识别"""
    prompt = f"用户工业监测需求：'{text}'。请提取其核心监测目标和关注指标，用3个要点描述（中文）："
    
    start = time.time()
    summary = _call_hf_text_model(DEFAULT_MODEL_ID, prompt)
    elapsed = time.time() - start
    return f"{summary} （分析耗时: {elapsed:.1f}s）"

def search_expert_agent() -> List[str]:
    """大模型 C：模型库检索"""
    prompt = "请列举5个适合工业异常检测和预测的AI模型名称（如LSTM, Transformer等），每行一个，不要解释。"
    
    text = _call_hf_text_model(DEFAULT_MODEL_ID, prompt, max_new_tokens=100)
    models = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
    
    return models if models else ["Informer", "TCN", "LSTM", "Autoformer", "DLinear"]

def final_decision_agent(data_feat: str, intent_feat: str, model_list: List[str]) -> Dict[str, Any]:
    """中央决策大模型：生成报告"""
    models_str = ", ".join(model_list)
    prompt = f"""
    作为专家，根据以下信息推荐最佳模型：
    数据特征：{data_feat}
    用户意图：{intent_feat}
    候选模型：{models_str}
    
    请输出 Markdown 格式报告，包含：
    1. 最终推荐模型：XXX
    2. 推荐理由
    3. 部署建议
    """

    start = time.time()
    report = _call_hf_text_model(DEFAULT_MODEL_ID, prompt)
    elapsed = time.time() - start

    return {
        "final_model": "Qwen-Industrial-Optimized", # 简化逻辑
        "report": f"{report}\n\n_（智能体决策耗时: {elapsed:.1f}s）_"
    }