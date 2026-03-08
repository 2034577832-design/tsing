import time
import base64
from pathlib import Path

import pandas as pd
import streamlit as st

# 直接从同目录下的 engine.py 导入 get_recommendation
from engine import get_recommendation


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

# 页面主标题
st.title("工业过程监测 - AI 模型推荐工具")
st.write("请上传工业过程的传感器数据（CSV），然后点击下方按钮执行 AI 诊断。")


# =========================
# 文件上传组件
# =========================
uploaded_file = st.file_uploader(
    label="上传工业传感器数据（CSV 文件）",  # 组件标题
    type=["csv"],  # 只允许 CSV
    help="示例：包含时间戳、多个传感器读数等列。",  # 鼠标悬浮时的提示
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
# 执行诊断按钮 + 进度条
# =========================
if uploaded_file is None:
    # 如果还没有上传文件，就给出提示信息
    st.info("请先上传一个 CSV 文件，然后再点击『执行诊断』。")
else:
    # 读取 CSV 为 DataFrame
    try:
        data_df = load_csv_to_df(uploaded_file)
    except Exception as e:
        # 如果读取失败，在页面上显示错误信息
        st.error(f"读取 CSV 文件失败，请检查格式是否正确。错误信息：{e}")
        data_df = None

    if data_df is not None:
        st.subheader("数据预览")
        # 显示前几行数据，方便你确认上传是否正确
        st.dataframe(data_df.head(), use_container_width=True)

        # 创建一个按钮，用户点击后开始诊断
        if st.button("执行诊断"):
            # 创建进度条组件和一个文本占位
            progress_bar = st.progress(0)
            status_text = st.empty()

            # spinner：显示“正在计算...”的加载动画
            with st.spinner("正在计算，请稍候..."):
                # 这里的循环只是为了做出“进度条在动”的效果
                for percent in range(0, 101, 20):
                    status_text.text(f"正在计算... {percent}%")
                    progress_bar.progress(percent)
                    time.sleep(0.2)

                # 真正的推荐逻辑：调用你在 engine.py 中的 get_recommendation
                # 这里把 data_df 传进去，便于你将来在函数里使用真实数据做计算
                recommendation = get_recommendation(data_df)

            # 计算结束后，清空进度条和状态文字
            status_text.empty()
            progress_bar.empty()

            # =========================
            # 结果展示：卡片 + 折线图
            # =========================
            st.success("诊断完成！以下是推荐结果：")

            # 从返回的字典中取出需要展示的字段
            model_name = recommendation.get("model_name", "未知模型")
            suitability = recommendation.get("suitability", None)
            reason = recommendation.get("reason", "暂无推荐理由")
            suggested_params = recommendation.get("suggested_params", "暂无参数建议")
            performance_score = recommendation.get("performance_score", [])

            # 适用度转成百分比显示
            suitability_text = (
                f"{(suitability * 100):.1f}%"
                if isinstance(suitability, (int, float))
                else "未知"
            )

            # 使用简单的 HTML + CSS 构造一个“卡片”样式（深色半透明背景 + 白色文字）
            card_html = f"""
            <div style="
                background-color: rgba(0, 0, 0, 0.55);
                border-radius: 12px;
                padding: 20px 24px;
                border: 1px solid rgba(255, 255, 255, 0.25);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
                margin-bottom: 16px;
                backdrop-filter: blur(4px);
            ">
                <h3 style="margin: 0 0 12px 0; color: #ffffff;">推荐模型：{model_name}</h3>
                <p style="margin: 4px 0; color: #ffffff;">
                    <strong>推荐理由：</strong>{reason}
                </p>
                <p style="margin: 4px 0; color: #ffffff;">
                    <strong>建议参数：</strong>{suggested_params}
                </p>
                <p style="margin: 4px 0; color: #ffffff;">
                    <strong>适用度评分：</strong>{suitability_text}
                </p>
            </div>
            """

            # unsafe_allow_html=True 允许我们渲染上面的 HTML 代码
            st.markdown(card_html, unsafe_allow_html=True)

            # 折线图：展示模型性能走势
            st.subheader("模型历史性能走势")

            if isinstance(performance_score, (list, tuple)) and len(performance_score) > 0:
                # 构造一个 DataFrame，方便 Streamlit 画图
                history_df = pd.DataFrame(
                    {
                        "轮次": list(range(1, len(performance_score) + 1)),
                        "性能评分": performance_score,
                    }
                ).set_index("轮次")

                # 使用 line_chart 绘制折线图
                st.line_chart(history_df, use_container_width=True)
            else:
                st.info("当前推荐结果中没有提供性能走势数据。")

