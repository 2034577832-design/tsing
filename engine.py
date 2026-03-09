import time
from typing import List, Dict, Any

import pandas as pd
import requests
import streamlit as st


# ======================================
# Hugging Face Inference API 通用封装
# ======================================

def _get_hf_headers() -> Dict[str, str]:
    """
    从 st.secrets 中读取 HF_TOKEN，构造请求头。
    需要在 .streamlit/secrets.toml 中配置：

    HF_TOKEN = "your_hf_api_token_here"
    """
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception as e:  # KeyError 或 Streamlit 未初始化等
        raise RuntimeError(
            "未在 st.secrets 中找到 HF_TOKEN，请在 .streamlit/secrets.toml 中配置。"
        ) from e

    return {"Authorization": f"Bearer {token}"}


def _call_hf_text_model(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """
    调用 Hugging Face Inference API 上的文本生成模型。

    为了通用，这里使用 text-generation 风格的接口，
    支持大多数指令微调的大模型。
    """
    url = f"https://api-inference.huggingface.co/models/{model_id}"

    payload: Dict[str, Any] = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
        },
    }

    headers = _get_hf_headers()

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Hugging Face API 调用失败 (model={model_id}, status={resp.status_code}): {resp.text}"
        )

    data = resp.json()

    # 常见返回格式：[{ "generated_text": "..." }]
    if isinstance(data, list) and data and "generated_text" in data[0]:
        generated = data[0]["generated_text"]
        # 一般 generated_text 会附带原始 prompt，这里只截取 prompt 之后的部分
        if generated.startswith(prompt):
            return generated[len(prompt) :].strip()
        return generated.strip()

    # 兜底：直接转成字符串
    return str(data)


# ============================
# 多智能体（多大模型）协同架构（真实 HF 版本）
# ============================

# 你可以在 secrets.toml 里增加一个 [HF_MODELS] 小节，自定义每个智能体使用的模型：
#
# [HF_MODELS]
# data_agent   = "your-data-model-id"
# intent_agent = "your-intent-model-id"
# search_agent = "your-search-model-id"
# decision_agent = "your-decision-model-id"
#

_hf_models = st.secrets.get("HF_MODELS", {}) if hasattr(st, "secrets") else {}

DATA_AGENT_MODEL = _hf_models.get("data_agent", "mistralai/Mixtral-8x7B-Instruct-v0.1")
INTENT_AGENT_MODEL = _hf_models.get(
    "intent_agent", "mistralai/Mixtral-8x7B-Instruct-v0.1"
)
SEARCH_AGENT_MODEL = _hf_models.get(
    "search_agent", "mistralai/Mixtral-8x7B-Instruct-v0.1"
)
DECISION_AGENT_MODEL = _hf_models.get(
    "decision_agent", "mistralai/Mixtral-8x7B-Instruct-v0.1"
)


def process_data_agent(data: pd.DataFrame) -> str:
    """
    大模型 A：专精时序数据特征提取。

    输入：工业时序 DataFrame
    输出：简洁的中文“数据特征摘要”，供后续智能体使用。
    """
    n_rows, n_cols = data.shape
    preview_csv = data.head(20).to_csv(index=False)

    prompt = f"""
你是一个工业过程监测与时序分析专家，请用简洁的中文总结下面数据的关键统计特征，
用于后续大模型选择，不要输出表格，只输出一个自然语言摘要，长度控制在 3~5 句。

数据规模：{n_rows} 行，{n_cols} 列。
示例数据（前 20 行 CSV）：
{preview_csv}
"""

    start = time.time()
    summary = _call_hf_text_model(DATA_AGENT_MODEL, prompt)
    elapsed = time.time() - start

    return f"{summary} （本次分析耗时约 {elapsed:.1f} 秒）"


def process_intent_agent(text: str) -> str:
    """
    大模型 B：专精自然语言意图分析。

    输入：用户自然语言意图
    输出：结构化的意图与场景摘要（中文自然语言）
    """
    prompt = f"""
你是一个工业 AI 解决方案架构师。
请阅读下面的用户需求描述，提取：
1）主要监测/预测目标；
2）关键指标（例如精度、实时性、鲁棒性等）；
3）应用场景（例如哪种工艺过程、控制目标）；
4）对模型的特殊要求（如支持长序列、异常检测、可解释性等）。

要求：
- 用中文回答；
- 以 3~5 条要点形式输出；
- 语言简洁，便于后续模型选择。

用户描述：{text}
"""

    start = time.time()
    summary = _call_hf_text_model(INTENT_AGENT_MODEL, prompt)
    elapsed = time.time() - start

    return f"{summary.strip()} （意图分析耗时约 {elapsed:.1f} 秒）"


def search_expert_agent() -> List[str]:
    """
    大模型 C：联网搜索 + 专家知识库匹配（简化版）。

    当前版本：调用 HF 上的大模型，让它基于已有知识给出适合
    “工业过程控制 / 时序预测”的代表性模型名称列表。
    """
    prompt = """
你是一名熟悉工业过程控制与时序预测的专家。
请给出 5~8 个适合用于工业时序数据建模的大模型或典型架构名称，
例如某些 Transformer 结构、时序卷积网络、残差网络变体等。

要求：
- 只输出模型或架构的名称，每行一个，不要解释；
- 优先选择在实际工业场景中常见或有代表性的方案。
"""

    text = _call_hf_text_model(SEARCH_AGENT_MODEL, prompt, max_new_tokens=256)

    # 解析为列表：按行拆分，去掉空行和序号
    models: List[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-•0123456789.、").strip()
        if line:
            models.append(line)

    # 兜底：如果解析不到，就给一个固定占位列表
    if not models:
        models = [
            "Informer",
            "Temporal Convolutional Network (TCN)",
            "LSTNet",
            "ResNet-Time",
        ]

    return models


def final_decision_agent(
    data_feat: str,
    intent_feat: str,
    model_list: List[str],
) -> Dict[str, Any]:
    """
    中央推理与决策大模型。

    输入：
    - data_feat: 时序数据特征摘要（process_data_agent 输出）
    - intent_feat: 用户意图与场景摘要（process_intent_agent 输出）
    - model_list: 候选大模型 / 架构名称列表（search_expert_agent 输出）

    输出：
    - final_model: 推荐的大模型名称
    - report: 自然语言的专业化诊断报告
    """
    candidate_block = "\n".join(f"- {m}" for m in model_list)

    prompt = f"""
你是一名工业 AI 总架构师，负责根据“数据特征 + 意图特征 + 候选模型列表”
为用户推荐一个最合适的时序建模大模型，并给出清晰的中文说明。

【数据特征摘要】
{data_feat}

【用户意图与场景摘要】
{intent_feat}

【候选大模型列表】
{candidate_block}

请完成以下任务：
1）在候选列表中选出一个最合适的模型（如果确实需要，可允许组合两个）；
2）用 2~3 句解释为什么这个模型适合当前的数据特征和业务场景；
3）给出 1~2 条关于部署或训练的高层次建议（例如需要多少历史窗口、是否需要在线更新等）；
4）输出时请使用 Markdown，包含：
   - 一行以“最终推荐模型：XXX”开头的标题；
   - 一个“推荐理由”小节；
   - 一个“实施建议”小节。
"""

    start = time.time()
    report = _call_hf_text_model(DECISION_AGENT_MODEL, prompt, max_new_tokens=512)
    elapsed = time.time() - start

    # 尝试从报告中抽取“最终推荐模型”这一行
    final_model = ""
    for line in report.splitlines():
        if "最终推荐模型" in line:
            final_model = line.replace("最终推荐模型", "").replace("：", ":").strip(" :#")
            break

    if not final_model and model_list:
        final_model = model_list[0]

    full_report = report.strip() + f"\n\n_（中央决策大模型推理耗时约 {elapsed:.1f} 秒）_"

    return {
        "final_model": final_model or "（未能从报告中解析模型名称）",
        "report": full_report,
    }
