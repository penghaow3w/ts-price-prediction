# 📈 TS Price Prediction | 时序价格预测系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Prophet-1.1-green.svg" alt="Prophet">
  <img src="https://img.shields.io/badge/Streamlit-1.28-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

A comprehensive time series price prediction system for **stocks and cryptocurrencies**, featuring three distinct model architectures with an interactive visualization dashboard.

一个面向**股票和加密货币**的综合时序价格预测系统，包含三种不同的模型架构和交互式可视化看板。

---

## 🌟 Key Features / 核心特性

### 🧠 Models / 模型
| Model | Architecture | Innovation |
|-------|-------------|------------|
| **LSTM-Attention** | 3-layer Stacked LSTM + Temporal Attention | Automatically learns important time steps |
| **Transformer-TD** | Transformer Encoder + Trend Decomposition | Separates trend from residual patterns |
| **Prophet** | Facebook Prophet + Custom Monthly Seasonality | Baseline with confidence intervals |

### 🔬 Innovations / 创新点

1. **Temporal Attention Mechanism** - LSTM with attention weights to focus on the most relevant historical time steps
   - 时间注意力机制 - LSTM自动关注最重要的历史时间步

2. **Trend Decomposition Module** - Transformer splits input into trend and residual components for better pattern capture
   - 趋势分解模块 - Transformer将输入分解为趋势和残差，更好地捕捉模式

3. **Technical Indicator Enrichment** - Automatically computes SMA, EMA, RSI, MACD, and Bollinger Bands as features
   - 技术指标特征增强 - 自动计算SMA/EMA/RSI/MACD/布林带等特征

4. **Direction Accuracy Metric** - Beyond MAE/RMSE, evaluates if the model correctly predicts price movement direction
   - 方向准确率指标 - 除MAE/RMSE外，还衡量模型是否正确预测了涨跌方向

5. **GBM Fallback Data Generator** - Geometric Brownian Motion simulation when yfinance API is rate-limited
   - GBM降级数据生成 - yfinance限速时使用几何布朗运动模型生成逼真数据

### 📊 Evaluation Metrics / 评估指标
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (R-squared)
- **Direction Accuracy** (涨跌方向准确率)

---

## 📁 Project Structure / 项目结构

```
ts-price-prediction/
├── fetch_data.py           # Data fetching with yfinance + GBM fallback
├── train_lstm.py           # LSTM-Attention model training
├── train_transformer.py    # Transformer-TD model training
├── prophet_model.py        # Prophet baseline model
├── dashboard.py            # Streamlit interactive dashboard
├── run_pipeline.py         # One-click full pipeline script
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md               # This file (bilingual)
├── utils/
│   ├── __init__.py
│   ├── config.py           # Global configuration & hyperparameters
│   └── preprocessing.py    # Data preprocessing & evaluation metrics
├── data/                   # Cached CSV data (generated at runtime)
├── models/                 # Saved model checkpoints (generated at runtime)
└── assets/                 # Static assets for dashboard
```

---

## 🚀 Quick Start / 快速开始

### 1. Install Dependencies / 安装依赖

```bash
pip install -r requirements.txt
```

### 2. One-Click Pipeline / 一键运行

```bash
# Fetch data + train all models for BTC and ETH
python run_pipeline.py --epochs 20

# Or specify symbols
python run_pipeline.py --symbols BTC-USD ETH-USD AAPL --epochs 30
```

### 3. Individual Commands / 单独运行

```bash
# Step 1: Fetch data / 获取数据
python fetch_data.py --symbol BTC-USD
python fetch_data.py --all                    # Fetch all symbols

# Step 2: Train models / 训练模型
python train_lstm.py --symbol BTC-USD --epochs 30
python train_transformer.py --symbol BTC-USD --epochs 30
python prophet_model.py --symbol BTC-USD

# Step 3: Launch dashboard / 启动看板
streamlit run dashboard.py
```

### 4. Use Simulated Data / 使用模拟数据

If yfinance is rate-limited or unavailable:
```bash
python fetch_data.py --symbol BTC-USD --simulated
```

---

## 📱 Dashboard Features / 看板功能

The Streamlit dashboard provides:

| Tab | Description | 描述 |
|-----|-------------|------|
| **Price & Prediction** | Candlestick chart + model predictions with confidence intervals | K线图 + 多模型预测曲线 + 置信区间 |
| **Model Evaluation** | Metrics table + MAE/RMSE bar chart + Direction Accuracy | 指标表格 + 柱状图 + 方向准确率 |
| **Technical Analysis** | SMA, EMA, RSI, MACD, Bollinger Bands visualization | 技术指标可视化 |
| **Model Comparison** | Radar chart for multi-dimensional model comparison | 多维度模型雷达图对比 |

---

## 🔧 Configuration / 配置说明

All hyperparameters are centralized in `utils/config.py`:

```python
# Sliding window / 滑动窗口
SEQUENCE_LENGTH = 60      # Past N days / 过去N天
PREDICTION_HORIZON = 7    # Future M days / 未来M天

# LSTM
LSTM_CONFIG = {
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.2,
    "epochs": 50,
}

# Transformer
TRANSFORMER_CONFIG = {
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 3,
    "epochs": 50,
}
```

### Supported Symbols / 支持标的

| Symbol | Name | Type |
|--------|------|------|
| `BTC-USD` | Bitcoin | Cryptocurrency |
| `ETH-USD` | Ethereum | Cryptocurrency |
| `AAPL` | Apple Inc. | Stock |
| `GOOGL` | Alphabet Inc. | Stock |
| `TSLA` | Tesla Inc. | Stock |
| `MSFT` | Microsoft Corp. | Stock |

You can add any symbol supported by `yfinance`.

---

## 🧪 Technical Architecture / 技术架构

### LSTM-Attention Model
```
Input → Linear Projection → Stacked LSTM (3 layers)
  → Temporal Attention (learned weights)
  → FC Output Layer → Multi-step Predictions
```

### Transformer-TD Model
```
Input → Linear Projection → Positional Encoding
  → Trend Decomposition (Conv1D)
    ├─ Trend Path → Transformer Encoder
    └─ Residual Path → Transformer Encoder
  → Adaptive Pool → Concatenate → FC Output
```

### Prophet Baseline
```
Input → Prophet (trend + yearly + weekly + monthly seasonality)
  → Predictions with Confidence Intervals
```

---

## 📋 Dependencies / 依赖

```
yfinance>=0.2.31     # Data fetching
numpy>=1.24.0        # Numerical computing
pandas>=2.0.0        # Data manipulation
scikit-learn>=1.3.0  # Evaluation metrics
torch>=2.1.0         # Deep learning
prophet>=1.1.5       # Baseline model
streamlit>=1.28.0    # Dashboard
plotly>=5.18.0        # Visualization
matplotlib>=3.8.0     # Plotting
seaborn>=0.13.0       # Statistical plots
tqdm>=4.66.0          # Progress bars
```

---

## ⚠️ Disclaimer / 免责声明

**This project is for educational and research purposes only.** The predictions generated by these models should NOT be used as financial advice or for actual trading decisions. Financial markets are inherently unpredictable, and past performance does not guarantee future results.

**本项目仅供教育和研究目的。** 模型生成的预测不应作为金融建议或实际交易决策的依据。金融市场本质上是不可预测的，过往表现不代表未来收益。

---

## 📄 License

MIT License

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<p align="center">
  Built with ❤️ using Python, PyTorch, Prophet & Streamlit
</p>
