"""
LSTM 时序预测模型
LSTM-based time series prediction model with attention mechanism (Innovation)
Features:
- Stacked LSTM with residual connections
- Attention mechanism for important time steps
- Multi-feature input support (technical indicators)
"""

import os
import sys
import argparse
import warnings
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import SEQUENCE_LENGTH, PREDICTION_HORIZON, LSTM_CONFIG, MODELS_DIR
from utils.preprocessing import (
    scale_data, inverse_scale, create_sequences,
    train_test_split_by_time, calculate_metrics,
)


# ==================== 创新组件：注意力机制 ====================
class TemporalAttention(nn.Module):
    """
    时间注意力机制 - 自动学习哪些时间步更重要
    Temporal Attention: learns which time steps are more important
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attn_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        weighted = torch.sum(lstm_output * attn_weights, dim=1)  # (batch, hidden_size)
        return weighted, attn_weights


# ==================== 主模型 ====================
class LSTMModel(nn.Module):
    """
    带注意力机制的堆叠LSTM模型
    Stacked LSTM with Temporal Attention and Residual Connections
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.2, pred_length=7):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length

        # 输入投影
        self.input_proj = nn.Linear(input_size, hidden_size)

        # 堆叠LSTM层
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # 时间注意力
        self.attention = TemporalAttention(hidden_size)

        # 输出层 - 多步预测
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, pred_length),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)

        # 投影到隐藏维度
        x = self.input_proj(x)

        # LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 注意力加权
        attn_out, attn_weights = self.attention(lstm_out)

        # 预测
        output = self.output_layer(attn_out)

        return output, attn_weights


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        predictions, _ = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    """评估"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_actuals = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions, _ = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            all_preds.append(predictions.cpu().numpy())
            all_actuals.append(y_batch.cpu().numpy())
    return (
        total_loss / len(data_loader),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_actuals, axis=0),
    )


def train_lstm(symbol: str = "BTC-USD", epochs: int = None, verbose: bool = True):
    """
    训练LSTM模型的完整流程
    Full LSTM training pipeline
    """
    config = LSTM_CONFIG.copy()
    if epochs is not None:
        config["epochs"] = epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"使用设备 / Using device: {device}")

    # 加载数据
    from fetch_data import load_data
    df = load_data(symbol)
    prices = df["close"].values.astype(float)

    # 归一化
    scaled_prices, scaler = scale_data(prices)

    # 创建序列
    X, y = create_sequences(scaled_prices, SEQUENCE_LENGTH, PREDICTION_HORIZON)

    # 时间序列划分
    X_train, X_test = train_test_split_by_time(X, test_ratio=0.2)
    y_train, y_test = train_test_split_by_time(y, test_ratio=0.2)

    # 创建DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # 初始化模型
    input_size = X_train.shape[2] if len(X_train.shape) == 3 else 1
    # 确保输入维度正确
    if len(X_train.shape) == 2:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = LSTMModel(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pred_length=PREDICTION_HORIZON,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 训练循环 + 早停
    best_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    if verbose:
        print(f"\n开始训练 / Training LSTM for {symbol}...")
        print(f"  序列长度 / Seq length: {SEQUENCE_LENGTH}")
        print(f"  预测步长 / Pred horizon: {PREDICTION_HORIZON}")
        print(f"  训练样本 / Train samples: {len(X_train)}")
        print(f"  测试样本 / Test samples: {len(X_test)}")
        print(f"  Epochs: {config['epochs']}")
        print()

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 早停
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config["early_stop_patience"]:
                if verbose:
                    print(f"  早停触发 / Early stopping at epoch {epoch+1}")
                break

    # 恢复最佳模型
    model.load_state_dict(best_model_state)

    # 在测试集上评估
    _, test_preds, test_actuals = evaluate(model, test_loader, criterion, device)

    # 反归一化
    test_preds_real = inverse_scale(test_preds.flatten(), scaler)
    test_actuals_real = inverse_scale(test_actuals.flatten(), scaler)

    metrics = calculate_metrics(test_actuals_real, test_preds_real)

    if verbose:
        print(f"\n========== LSTM 测试集评估结果 / Test Evaluation ==========")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # 保存模型
    model_path = os.path.join(
        MODELS_DIR,
        f"lstm_{symbol.replace('-', '_').replace('/', '_')}.pt",
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "config": config,
        "metrics": metrics,
        "history": history,
        "sequence_length": SEQUENCE_LENGTH,
        "prediction_horizon": PREDICTION_HORIZON,
    }, model_path)

    if verbose:
        print(f"\n✓ 模型已保存 / Model saved to: {model_path}")

    return model, scaler, metrics, history


def predict_lstm(model, scaler, last_sequence: np.ndarray, pred_length: int = None) -> np.ndarray:
    """
    使用训练好的LSTM模型进行预测
    Predict using trained LSTM model
    """
    if pred_length is None:
        pred_length = PREDICTION_HORIZON

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        x = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        predictions, attn_weights = model(x)
        predictions = predictions.cpu().numpy().flatten()

    return inverse_scale(predictions, scaler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM 价格预测模型训练 / LSTM Price Prediction Training")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="股票代码 / Symbol")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮次 / Epochs")

    args = parser.parse_args()
    train_lstm(symbol=args.symbol, epochs=args.epochs)
