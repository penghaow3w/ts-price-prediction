"""
统一运行脚本 - 一键获取数据、训练模型、评估预测
One-click pipeline: fetch data -> train models -> evaluate
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_full_pipeline(symbols=None, epochs=20):
    """
    完整的流水线：数据获取 -> LSTM训练 -> Transformer训练 -> Prophet训练
    Full pipeline: data fetch -> LSTM -> Transformer -> Prophet
    """
    from utils.config import DEFAULT_SYMBOLS

    if symbols is None:
        symbols = ["BTC-USD", "ETH-USD"]  # 默认先做加密货币

    print("=" * 60)
    print("  TS Price Prediction - 完整流水线 / Full Pipeline")
    print("=" * 60)

    # Step 1: 获取数据
    print("\n" + "=" * 40)
    print("  Step 1: 数据获取 / Data Fetching")
    print("=" * 40)
    from fetch_data import fetch_stock_data, save_data
    for symbol in symbols:
        try:
            df = fetch_stock_data(symbol)
            save_data(df, symbol)
        except Exception as e:
            print(f"  跳过 {symbol}: {e}")

    # Step 2: 训练LSTM
    print("\n" + "=" * 40)
    print("  Step 2: 训练LSTM模型 / Training LSTM")
    print("=" * 40)
    from train_lstm import train_lstm
    for symbol in symbols:
        try:
            train_lstm(symbol=symbol, epochs=epochs)
        except Exception as e:
            print(f"  LSTM {symbol} 训练失败: {e}")

    # Step 3: 训练Transformer
    print("\n" + "=" * 40)
    print("  Step 3: 训练Transformer模型 / Training Transformer")
    print("=" * 40)
    from train_transformer import train_transformer
    for symbol in symbols:
        try:
            train_transformer(symbol=symbol, epochs=epochs)
        except Exception as e:
            print(f"  Transformer {symbol} 训练失败: {e}")

    # Step 4: 训练Prophet
    print("\n" + "=" * 40)
    print("  Step 4: 训练Prophet模型 / Training Prophet")
    print("=" * 40)
    from prophet_model import train_prophet
    for symbol in symbols:
        try:
            train_prophet(symbol=symbol)
        except Exception as e:
            print(f"  Prophet {symbol} 训练失败: {e}")

    print("\n" + "=" * 60)
    print("  流水线完成！/ Pipeline Complete!")
    print("  运行 streamlit 启动看板: streamlit run dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一键运行完整流水线 / Run Full Pipeline")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮次 / Epochs (default: 20 for quick run)")
    parser.add_argument("--symbols", type=str, nargs="+", default=["BTC-USD", "ETH-USD"],
                        help="股票代码 / Symbols (e.g., BTC-USD AAPL)")

    args = parser.parse_args()
    run_full_pipeline(symbols=args.symbols, epochs=args.epochs)
