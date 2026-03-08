import time
import pandas as pd

def get_recommendation(input_data=None):
    """确保函数名完全一致"""
    time.sleep(1) # 模拟计算
    return {
        "model_name": "深度残差网络 (ResNet-50)",
        "suitability": 0.92,
        "reason": "由于您的传感器数据具有明显的时序特征，ResNet 能更好地捕捉异常波动。",
        "suggested_params": "学习率: 0.001, Batch Size: 32",
        "performance_score": [85, 90, 88, 92, 95]
    }