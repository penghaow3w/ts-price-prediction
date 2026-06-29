"""
数据获取模块 - 使用 yfinance 获取股票/加密货币历史价格数据
Data fetching module using yfinance library
Supports both stocks and cryptocurrencies with technical indicator enrichment
包含降级策略：yfinance失败时使用模拟数据
"""

import os
import sys
import warnings
import argparse
import datetime
import time

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import (
    DEFAULT_SYMBOLS, DEFAULT_START_DATE, DEFAULT_END_DATE,
    DATA_DIR, TECHNICAL_INDICATORS, MULTI_FEATURE_WINDOW,
)
from utils.preprocessing import calculate_technical_indicators


def generate_simulated_data(symbol: str, start_date: str, end_date: str, seed: int = 42) -> pd.DataFrame:
    """
    生成逼真的模拟金融数据（yfinance不可用时的降级方案）
    Generate realistic simulated financial data as fallback
    使用几何布朗运动 (GBM) 模型模拟价格路径
    """
    np.random.seed(seed)

    # 根据标的设定不同的初始价格和波动率
    symbol_configs = {
        "BTC-USD":  {"init_price": 9000, "mu": 0.0003, "sigma": 0.035},
        "ETH-USD":  {"init_price": 200, "mu": 0.0004, "sigma": 0.04},
        "AAPL":     {"init_price": 80, "mu": 0.0005, "sigma": 0.018},
        "GOOGL":    {"init_price": 60, "mu": 0.0004, "sigma": 0.020},
        "TSLA":     {"init_price": 30, "mu": 0.0006, "sigma": 0.032},
        "MSFT":     {"init_price": 100, "mu": 0.0003, "sigma": 0.017},
    }

    config = symbol_configs.get(symbol, {"init_price": 100, "mu": 0.0003, "sigma": 0.02})

    dates = pd.date_range(start=start_date, end=end_date, freq="B")  # 工作日
    n = len(dates)

    # 几何布朗运动
    dt = 1
    price = config["init_price"]
    prices = [price]

    for _ in range(1, n):
        # 带有随机扰动的GBM
        shock = np.random.normal(0, 1) * np.sqrt(dt)
        drift = (config["mu"] - 0.5 * config["sigma"] ** 2) * dt
        diffusion = config["sigma"] * shock
        price = price * np.exp(drift + diffusion)
        prices.append(price)

    # 生成OHLCV
    df = pd.DataFrame({"Date": dates})
    close_prices = np.array(prices)

    # High/Low在Close附近波动
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_prices = close_prices * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.randint(100000, 10000000, n)

    df["open"] = open_prices
    df["high"] = np.maximum(high_prices, open_prices)
    df["low"] = np.minimum(low_prices, open_prices)
    df["close"] = close_prices
    df["volume"] = volume

    # 确保high >= max(open, close) and low <= min(open, close)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df


def fetch_stock_data(
    symbol: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = None,
    add_indicators: bool = True,
    use_simulated: bool = False,
) -> pd.DataFrame:
    """
    获取股票/加密货币历史数据
    Fetch historical price data for stocks or cryptocurrencies

    Parameters:
        symbol: 股票代码或加密货币对 (e.g., "AAPL", "BTC-USD")
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期，默认今天
        add_indicators: 是否添加技术指标
        use_simulated: 强制使用模拟数据
    Returns:
        DataFrame 包含价格数据和技术指标
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    print(f"正在获取 {symbol} 的数据 ({start_date} 至 {end_date})...")
    print(f"Fetching {symbol} data ({start_date} to {end_date})...")

    df = pd.DataFrame()

    if not use_simulated:
        # 尝试 yfinance 下载
        try:
            # 方法1: 使用 download
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if df.empty:
                raise ValueError("Empty data from download")

            # 处理MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df = df.reset_index()

        except Exception as e:
            print(f"  yfinance download failed: {e}")
            try:
                # 方法2: 使用 Ticker.history
                time.sleep(2)
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                if df.empty:
                    raise ValueError("Empty data from Ticker.history")
                df = df.reset_index()
            except Exception as e2:
                print(f"  yfinance Ticker.history also failed: {e2}")

    # 如果yfinance都失败了，使用模拟数据
    if df.empty:
        print(f"  ⚠ yfinance不可用，使用模拟数据 / Using simulated data (GBM model)")
        df = generate_simulated_data(symbol, start_date, end_date)

    # 标准化列名
    df.columns = [col.replace(" ", "_").lower() if col != "Date" else "Date" for col in df.columns]

    # 保留需要的列
    keep_cols = ["Date"]
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            keep_cols.append(col)
    df = df[keep_cols].copy()

    # 去掉时区信息
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    # 去除NaN行
    df = df.dropna().reset_index(drop=True)

    # 添加技术指标（创新功能）
    if add_indicators:
        df = calculate_technical_indicators(df, window=MULTI_FEATURE_WINDOW)

    print(f"  ✓ 成功获取 {len(df)} 条记录 / Successfully fetched {len(df)} records")
    return df


def save_data(df: pd.DataFrame, symbol: str) -> str:
    """保存数据到CSV文件"""
    filename = symbol.replace("-", "_").replace("/", "_") + ".csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"  ✓ 数据已保存至 / Data saved to: {filepath}")
    return filepath


def load_data(symbol: str) -> pd.DataFrame:
    """从CSV加载数据"""
    filename = symbol.replace("-", "_").replace("/", "_") + ".csv"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在 / Data file not found: {filepath}")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def fetch_all_symbols(symbols: dict = None) -> dict:
    """
    批量获取多个标的的数据
    Batch fetch data for multiple symbols
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    results = {}
    for symbol, name in symbols.items():
        try:
            df = fetch_stock_data(symbol)
            save_data(df, symbol)
            results[symbol] = {"name": name, "records": len(df), "status": "success"}
        except Exception as e:
            print(f"✗ 获取 {symbol} 失败 / Failed to fetch {symbol}: {e}")
            results[symbol] = {"name": name, "records": 0, "status": f"error: {str(e)}"}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="股票/加密货币数据获取工具 / Stock & Crypto Data Fetcher")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="股票代码 / Stock symbol (e.g., BTC-USD, AAPL)")
    parser.add_argument("--start", type=str, default=DEFAULT_START_DATE, help="开始日期 / Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="结束日期 / End date (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="获取所有标的 / Fetch all symbols")
    parser.add_argument("--no-indicators", action="store_true", help="不添加技术指标 / Skip technical indicators")
    parser.add_argument("--simulated", action="store_true", help="使用模拟数据 / Use simulated data")

    args = parser.parse_args()

    if args.all:
        results = fetch_all_symbols()
        print("\n========== 获取结果汇总 / Summary ==========")
        for sym, info in results.items():
            status_icon = "✓" if info["status"] == "success" else "✗"
            print(f"  {status_icon} {sym} ({info['name']}): {info['status']} - {info['records']} records")
    else:
        df = fetch_stock_data(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            add_indicators=not args.no_indicators,
            use_simulated=args.simulated,
        )
        save_data(df, args.symbol)
        print(f"\n数据预览 / Data preview:")
        print(df.head(10).to_string(index=False))
