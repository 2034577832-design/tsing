import time
import base64
from pathlib import Path

import pandas as pd
import streamlit as st

# 导入后端多智能体逻辑
import engine


# =========================
# 设置页面背景图（使用本地图片）
# =========================
def set_background(image_filename: str = "background.jpg"):
    """
    使用本地图片作为整个应用的背景。
    注意：图片文件需要和 main.py 放在同一个文件夹下。
    """
    img_path = Path(__file__).parent / image_filename
    if not img_path.exists():
        # 如果找不到图片，就什么都不做，避免程序报错
        return

    # 读取图片并转成 base64，这样可以直接嵌入到 CSS 里
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    # 通过 st.markdown 注入自定义 CSS，设置 .stApp 的背景和文字颜色
    css = f"""
    <style>
    html, body {{
        height: 100%;
    }}

    body {{
        background-color: transparent;
    }}

    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
    }}

    /* 标题、正文等文字统一设置为白色，提升对比度 */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp label, .stApp span, .stApp li, .stApp strong {{
        color: #ffffff;
    }}

    /* 顶部原生 header 变成透明，让背景图覆盖到页面最上端 */
    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    header[data-testid="stHeader"] > div {{
        background: transparent;
    }}

    /* 主内容整体往下挪一些，让关键信息更接近屏幕视觉中心 */
    .main .block-container {{
        padding-top: 18vh;
    }}

    /* 文件上传整体容器：加深背景，做成深色条形区域 */
    div[data-testid="stFileUploader"] {{
        background-color: rgba(0, 0, 0, 0.75);
        border-radius: 12px;
        padding: 8px 12px;
    }}

    /* 具体拖拽区域：改成接近纯黑背景，去掉浅灰底 */
    div[data-testid="stFileUploadDropzone"] {{
        background-color: rgba(0, 0, 0, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }}

    /* 拖拽区域里的文字全部改成白色，保证清晰可见 */
    div[data-testid="stFileUploadDropzone"] * {{
        color: #ffffff !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =========================
# Streamlit 页面基础配置
# =========================
st.set_page_config(
    page_title="工业过程监测 - AI 模型推荐工具",  # 浏览器标签页标题
    layout="wide",  # 宽屏布局
)

# 先设置背景图，再渲染页面内容
set_background("background.jpg")

# 页面主标题（居中区域）
st.title("工业过程监测 - AI 模型推荐工具")
st.write("请上传工业时序数据，并用自然语言描述你的监测意图，系统将协同调用多智能体大模型完成诊断与推荐。")


# =========================
# 侧边栏：输入配置
# =========================
with st.sidebar:
    st.header("🛠️ 配置输入")
    uploaded_file = st.file_uploader(
        label="上传工业时序数据（CSV 文件）",
        type=["csv"],
        help="示例：包含时间戳、多个传感器读数等列。",
    )

    user_intent = st.text_area(
        "描述您的监测意图",
        placeholder="例如：我想在高频噪声环境下，对关键压力与温度的未来 30 分钟走势进行高精度预测，用于铝电解过程的精细控制……",
        height=150,
    )


@st.cache_data
def load_csv_to_df(file) -> pd.DataFrame:
    """
    将上传的 CSV 文件读取为 Pandas DataFrame。
    使用 cache_data 装饰器可以避免在每次交互时重复读取文件，提高性能。
    """
    df = pd.read_csv(file)
    return df


# =========================
# 执行诊断按钮 + 多智能体“思维链”展示
# =========================

st.markdown("### 🤖 多智能体大模型协同诊断")
st.write("系统将依次调用：**数据智能体 → 语义智能体 → 搜索智能体 → 决策智能体**，生成最终的大模型推荐报告。")

if st.button("🚀 开始多智能体协同诊断"):
    if uploaded_file and user_intent:
        # 读取 CSV 为 DataFrame
        try:
            data_df = load_csv_to_df(uploaded_file)
        except Exception as e:
            st.error(f"读取 CSV 文件失败，请检查格式是否正确。错误信息：{e}")
            data_df = None

        if data_df is not None:
            # 数据预览
            st.subheader("数据预览")
            st.dataframe(data_df.head(), use_container_width=True)

            # “思维链 / 调用链”状态展示
            with st.status("🤖 多智能体协同分析中...", expanded=True) as status:
                st.write("🔍 **[数据智能体]** 正在提取时序特征……")
                data_feat = engine.process_data_agent(data_df)

                st.write("🧠 **[语义智能体]** 正在解析用户意图与场景……")
                intent_feat = engine.process_intent_agent(user_intent)

                st.write("🌐 **[搜索智能体]** 正在检索专家知识库与公网方案……")
                model_list = engine.search_expert_agent()

                st.write("⚖️ **[决策智能体]** 正在综合评估并生成最终报告……")
                decision_result = engine.final_decision_agent(
                    data_feat, intent_feat, model_list
                )

                status.update(label="✅ 诊断完成！", state="complete", expanded=False)

            # =========================
            # 结果展示：卡片 + 折线图
            # =========================
            st.success(f"### 推荐模型：{decision_result['final_model']}")

            # 专业化诊断报告（Markdown 渲染）
            st.markdown("#### 📄 专家诊断报告")
            st.markdown(decision_result["report"])

            # 将数据/意图/候选列表的“中间推理信息”也展示出来，增强可解释性
            st.markdown("#### 🧩 中间推理要点")
            st.markdown(f"- **数据特征摘要：** {data_feat}")
            st.markdown(f"- **意图与场景理解：** {intent_feat}")
            st.markdown(f"- **候选模型集合：** {', '.join(model_list)}")

        else:
            st.error("数据读取失败，无法继续诊断。")
    else:
        st.warning("请在左侧同时上传工业时序数据，并填写自然语言意图。")
