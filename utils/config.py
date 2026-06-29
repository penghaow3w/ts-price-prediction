"""
全局配置文件
Global configuration for the ts-price-prediction project
"""

import os

# ==================== 数据配置 ====================
DEFAULT_SYMBOLS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "AAPL": "Apple Inc.",
    "GOOGL": "Alphabet Inc.",
    "TSLA": "Tesla Inc.",
    "MSFT": "Microsoft Corp.",
}

DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = None  # 使用今天

# ==================== 模型超参数 ====================
# 滑动窗口参数
SEQUENCE_LENGTH = 60      # 用过去60天数据
PREDICTION_HORIZON = 7    # 预测未来7天

# LSTM 参数
LSTM_CONFIG = {
    "input_size": 1,
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "early_stop_patience": 8,
}

# Transformer 参数
TRANSFORMER_CONFIG = {
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "learning_rate": 0.0005,
    "batch_size": 32,
    "epochs": 50,
    "early_stop_patience": 8,
}

# Prophet 参数
PROPHET_CONFIG = {
    "changepoint_prior_scale": 0.05,
    "seasonality_mode": "multiplicative",
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
}

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# 确保目录存在
for _dir in [DATA_DIR, MODELS_DIR, ASSETS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ==================== 创新功能：多特征配置 ====================
# 技术指标特征开关
TECHNICAL_INDICATORS = {
    "sma": True,          # 简单移动平均
    "ema": True,          # 指数移动平均
    "rsi": True,          # 相对强弱指标
    "macd": True,         # MACD
    "bollinger": True,    # 布林带
    "vwap": False,        # 成交量加权平均价（需要成交量数据）
}

# 多特征窗口参数
MULTI_FEATURE_WINDOW = 20  # 技术指标计算窗口
