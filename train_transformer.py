"""
Transformer 时序预测模型
Transformer-based time series prediction model (Innovation)
Features:
- Custom positional encoding for time series
- Multi-head self-attention mechanism
- Learnable trend decomposition
"""

import os
import sys
import argparse
import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import SEQUENCE_LENGTH, PREDICTION_HORIZON, TRANSFORMER_CONFIG, MODELS_DIR
from utils.preprocessing import (
    scale_data, inverse_scale, create_sequences,
    train_test_split_by_time, calculate_metrics,
)


# ==================== 创新组件：可学习位置编码 ====================
class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码 - 适配时间序列数据
    Learnable positional encoding adapted for financial time series
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pos_encoding[:, : x.size(1), :]


# ==================== 创新组件：趋势分解模块 ====================
class TrendDecomposition(nn.Module):
    """
    趋势分解 - 将序列分解为趋势和残差
    Decompose time series into trend and residual components
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.trend_conv = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, bias=False)
        self.residual_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x_t = x.transpose(1, 2)  # (batch, d_model, seq_len)
        trend = self.trend_conv(x_t).transpose(1, 2)  # (batch, seq_len, d_model)
        residual = self.residual_proj(x - trend)
        return trend, residual


# ==================== 主模型 ====================
class TransformerPredictor(nn.Module):
    """
    Transformer 时序预测模型
    Transformer-based Time Series Predictor with Trend Decomposition
    """

    def __init__(
        self,
        input_size=1,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        pred_length=7,
    ):
        super().__init__()
        self.d_model = d_model
        self.pred_length = pred_length

        # 输入投影
        self.input_proj = nn.Linear(input_size, d_model)

        # 可学习位置编码
        self.pos_encoder = LearnablePositionalEncoding(d_model)

        # 趋势分解
        self.trend_decomp = TrendDecomposition(d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # 自适应池化 - 创新点
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_length),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)

        # 趋势分解
        trend, residual = self.trend_decomp(x)

        # 趋势特征通过Transformer
        trend_encoded = self.transformer_encoder(trend)

        # 残差特征也通过Transformer
        residual_encoded = self.transformer_encoder(residual)

        # 池化
        trend_pooled = self.adaptive_pool(trend_encoded.transpose(1, 2)).squeeze(-1)
        residual_pooled = self.adaptive_pool(residual_encoded.transpose(1, 2)).squeeze(-1)

        # 拼接特征
        combined = torch.cat([trend_pooled, residual_pooled], dim=-1)

        # 预测
        output = self.predictor(combined)

        return output


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
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
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            all_preds.append(predictions.cpu().numpy())
            all_actuals.append(y_batch.cpu().numpy())
    return (
        total_loss / len(data_loader),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_actuals, axis=0),
    )


def train_transformer(symbol: str = "BTC-USD", epochs: int = None, verbose: bool = True):
    """
    训练Transformer模型的完整流程
    Full Transformer training pipeline
    """
    config = TRANSFORMER_CONFIG.copy()
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

    # 确保输入维度
    if len(X_train.shape) == 2:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # 创建DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    input_size = X_train.shape[2]

    # 初始化模型
    model = TransformerPredictor(
        input_size=input_size,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        pred_length=PREDICTION_HORIZON,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)

    # 训练循环
    best_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    if verbose:
        print(f"\n开始训练 / Training Transformer for {symbol}...")
        print(f"  序列长度 / Seq length: {SEQUENCE_LENGTH}")
        print(f"  预测步长 / Pred horizon: {PREDICTION_HORIZON}")
        print(f"  训练样本 / Train samples: {len(X_train)}")
        print(f"  测试样本 / Test samples: {len(X_test)}")
        print(f"  Epochs: {config['epochs']}")
        print()

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

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

    # 测试集评估
    _, test_preds, test_actuals = evaluate(model, test_loader, criterion, device)

    # 反归一化
    test_preds_real = inverse_scale(test_preds.flatten(), scaler)
    test_actuals_real = inverse_scale(test_actuals.flatten(), scaler)

    metrics = calculate_metrics(test_actuals_real, test_preds_real)

    if verbose:
        print(f"\n========== Transformer 测试集评估结果 / Test Evaluation ==========")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # 保存模型
    model_path = os.path.join(
        MODELS_DIR,
        f"transformer_{symbol.replace('-', '_').replace('/', '_')}.pt",
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


def predict_transformer(model, scaler, last_sequence: np.ndarray, pred_length: int = None) -> np.ndarray:
    """
    使用训练好的Transformer模型进行预测
    Predict using trained Transformer model
    """
    if pred_length is None:
        pred_length = PREDICTION_HORIZON

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        x = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        predictions = model(x)
        predictions = predictions.cpu().numpy().flatten()

    return inverse_scale(predictions, scaler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer 价格预测模型训练 / Transformer Price Prediction Training")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="股票代码 / Symbol")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮次 / Epochs")

    args = parser.parse_args()
    train_transformer(symbol=args.symbol, epochs=args.epochs)
