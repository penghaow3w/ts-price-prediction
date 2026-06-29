"""
数据预处理与评估工具模块
Data preprocessing, technical indicators, and evaluation metrics
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_technical_indicators(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    计算技术指标特征（创新点：多特征增强）
    Calculate technical indicators as additional features
    """
    result = df.copy()

    # 兼容大小写列名
    close_col = "Close" if "Close" in result.columns else "close"

    # 简单移动平均 (SMA)
    result["SMA_{}".format(window)] = result[close_col].rolling(window=window).mean()

    # 指数移动平均 (EMA)
    result["EMA_{}".format(window)] = result[close_col].ewm(span=window, adjust=False).mean()

    # 相对强弱指标 (RSI)
    delta = result[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)
    result["RSI_{}".format(window)] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = result[close_col].ewm(span=12, adjust=False).mean()
    ema26 = result[close_col].ewm(span=26, adjust=False).mean()
    result["MACD"] = ema12 - ema26
    result["MACD_Signal"] = result["MACD"].ewm(span=9, adjust=False).mean()

    # 布林带
    sma = result[close_col].rolling(window=window).mean()
    std = result[close_col].rolling(window=window).std()
    result["BB_upper"] = sma + 2 * std
    result["BB_lower"] = sma - 2 * std
    result["BB_width"] = (result["BB_upper"] - result["BB_lower"]) / sma

    # 填充NaN
    result = result.bfill().ffill()

    return result


def create_sequences(data: np.ndarray, seq_length: int, pred_length: int) -> tuple:
    """
    创建滑动窗口序列
    Create sliding window sequences for time series prediction
    """
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_length])
    return np.array(X), np.array(y)


def scale_data(prices: np.ndarray, scaler=None) -> tuple:
    """
    归一化数据
    Scale data using MinMaxScaler
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    else:
        scaled = scaler.transform(prices.reshape(-1, 1)).flatten()
    return scaled, scaler


def inverse_scale(scaled_data: np.ndarray, scaler) -> np.ndarray:
    """反归一化"""
    return scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    计算评估指标: MAE, RMSE, MAPE, R²
    Calculate evaluation metrics
    """
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE - 避免除以零
    nonzero_mask = actual != 0
    mape = np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])) * 100

    # R²
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    # 方向准确率 (Directional Accuracy) - 创新指标
    if len(actual) > 1:
        actual_dir = np.diff(actual) > 0
        pred_dir = np.diff(predicted) > 0
        dir_acc = np.mean(actual_dir == pred_dir) * 100
    else:
        dir_acc = 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R2": round(r2, 6),
        "Direction_Accuracy": round(dir_acc, 2),
    }


def train_test_split_by_time(data: np.ndarray, test_ratio: float = 0.2) -> tuple:
    """
    按时间顺序划分训练集和测试集（避免数据泄露）
    """
    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]
