"""
Prophet (Facebook) 基线模型
Prophet baseline model for time series forecasting
Prophet provides trend, seasonality, and holiday effect decomposition
"""

import os
import sys
import argparse
import warnings
import pickle

import numpy as np
import pandas as pd
from prophet import Prophet
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import SEQUENCE_LENGTH, PREDICTION_HORIZON, PROPHET_CONFIG, MODELS_DIR
from utils.preprocessing import calculate_metrics


def train_prophet(symbol: str = "BTC-USD", verbose: bool = True):
    """
    训练Prophet模型的完整流程
    Full Prophet training pipeline
    """
    # 加载数据
    from fetch_data import load_data
    df = load_data(symbol)

    if verbose:
        print(f"\n开始训练 / Training Prophet for {symbol}...")

    # Prophet需要特定的列名格式
    prophet_df = pd.DataFrame({
        "ds": df["Date"],
        "y": df["close"],
    })

    # 时间序列划分 (80%训练，20%测试)
    split_idx = int(len(prophet_df) * 0.8)
    train_df = prophet_df.iloc[:split_idx].copy()
    test_df = prophet_df.iloc[split_idx:].copy()

    if verbose:
        print(f"  训练数据 / Train: {len(train_df)} 天")
        print(f"  测试数据 / Test: {len(test_df)} 天")

    # 初始化Prophet模型
    model = Prophet(
        changepoint_prior_scale=PROPHET_CONFIG["changepoint_prior_scale"],
        seasonality_mode=PROPHET_CONFIG["seasonality_mode"],
        yearly_seasonality=PROPHET_CONFIG["yearly_seasonality"],
        weekly_seasonality=PROPHET_CONFIG["weekly_seasonality"],
        daily_seasonality=PROPHET_CONFIG["daily_seasonality"],
    )

    # 创新点：添加自定义季节性
    # 加密货币可能有25天月度周期
    model.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=5,
    )

    # 拟合模型
    model.fit(train_df)

    # 预测 - 使用所有日期确保覆盖测试集
    test_dates = test_df["ds"].values
    last_train_date = train_df["ds"].max()
    days_after = (pd.Timestamp(max(test_dates)) - last_train_date).days + 10
    future = model.make_future_dataframe(periods=days_after)
    forecast = model.predict(future)

    # 提取测试集预测结果 - 通过日期匹配
    test_forecast = forecast[forecast["ds"].isin(test_df["ds"])].copy()
    test_forecast = test_forecast.sort_values("ds").reset_index(drop=True)

    # 确保长度一致 - 合并测试数据和预测
    merged = pd.merge(test_df, test_forecast[["ds", "yhat"]], on="ds", how="inner")
    merged = merged.sort_values("ds").reset_index(drop=True)

    # 计算评估指标
    actual = merged["y"].values
    predicted = merged["yhat"].values

    metrics = calculate_metrics(actual, predicted)

    if verbose:
        print(f"\n========== Prophet 测试集评估结果 / Test Evaluation ==========")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # 保存模型
    model_path = os.path.join(
        MODELS_DIR,
        f"prophet_{symbol.replace('-', '_').replace('/', '_')}.pkl",
    )
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "metrics": metrics,
            "train_size": len(train_df),
        }, f)

    if verbose:
        print(f"\n✓ 模型已保存 / Model saved to: {model_path}")

    return model, forecast, metrics


def predict_prophet(model, periods: int = 7) -> pd.DataFrame:
    """
    使用Prophet模型预测未来periods天
    Predict future periods using Prophet model
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # 只返回未来的预测
    result = forecast.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result.columns = ["Date", "Predicted", "Lower_Bound", "Upper_Bound"]
    return result


def get_prophet_components(model, df: pd.DataFrame) -> dict:
    """
    获取Prophet分解组件（趋势、季节性等）
    Get Prophet decomposition components
    """
    forecast = model.predict(df)
    components = {}
    for comp in ["trend", "weekly", "yearly", "monthly", "additive_terms", "multiplicative_terms"]:
        if comp in forecast.columns:
            components[comp] = forecast[["ds", comp]].copy()
            components[comp].columns = ["Date", comp]
    return components


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prophet 价格预测模型训练 / Prophet Price Prediction Training")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="股票代码 / Symbol")

    args = parser.parse_args()
    train_prophet(symbol=args.symbol)
