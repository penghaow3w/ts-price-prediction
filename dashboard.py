"""
Streamlit 可视化看板
Interactive visualization dashboard for price prediction comparison
Features:
- Multi-symbol selection
- Multi-model comparison
- Real-time prediction visualization
- Confidence intervals
- Model evaluation metrics
- Technical indicators overlay
"""

import os
import sys
import warnings
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import (
    DEFAULT_SYMBOLS, SEQUENCE_LENGTH, PREDICTION_HORIZON,
    DATA_DIR, MODELS_DIR,
)
from utils.preprocessing import scale_data, inverse_scale, create_sequences

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="TS Price Prediction | 时序价格预测",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 自定义样式 ====================
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .model-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==================== 辅助函数 ====================
@st.cache_data(ttl=3600)
def get_available_symbols() -> list:
    """获取可用的股票/加密货币"""
    symbols = []
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith(".csv"):
                symbols.append(f.replace(".csv", "").replace("_", "-"))
    return symbols if symbols else list(DEFAULT_SYMBOLS.keys())


@st.cache_data(ttl=3600)
def load_symbol_data(symbol: str) -> pd.DataFrame:
    """加载指定标的的数据"""
    from fetch_data import load_data
    return load_data(symbol)


def check_model_exists(model_type: str, symbol: str) -> bool:
    """检查模型文件是否存在"""
    safe_name = symbol.replace("-", "_").replace("/", "_")
    if model_type == "lstm":
        return os.path.exists(os.path.join(MODELS_DIR, f"lstm_{safe_name}.pt"))
    elif model_type == "transformer":
        return os.path.exists(os.path.join(MODELS_DIR, f"transformer_{safe_name}.pt"))
    elif model_type == "prophet":
        return os.path.exists(os.path.join(MODELS_DIR, f"prophet_{safe_name}.pkl"))
    return False


@st.cache_resource(ttl=3600)
def load_prophet_model(symbol: str):
    """加载Prophet模型"""
    import torch
    safe_name = symbol.replace("-", "_").replace("/", "_")
    model_path = os.path.join(MODELS_DIR, f"prophet_{safe_name}.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        return data["model"], data["metrics"]
    return None, None


@st.cache_resource(ttl=3600)
def load_pytorch_model(model_type: str, symbol: str):
    """加载PyTorch模型 (LSTM/Transformer)"""
    import torch
    safe_name = symbol.replace("-", "_").replace("/", "_")

    if model_type == "lstm":
        model_path = os.path.join(MODELS_DIR, f"lstm_{safe_name}.pt")
        from train_lstm import LSTMModel
        config_key = "LSTM_CONFIG"
        model_cls = LSTMModel
    else:
        model_path = os.path.join(MODELS_DIR, f"transformer_{safe_name}.pt")
        from train_transformer import TransformerPredictor
        config_key = "TRANSFORMER_CONFIG"
        model_cls = TransformerPredictor

    if not os.path.exists(model_path):
        return None, None, None

    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint["config"]
    scaler = checkpoint["scaler"]
    metrics = checkpoint["metrics"]

    model = model_cls(
        input_size=config["input_size"] if model_type == "lstm" else 1,
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 3),
        dropout=0,  # 推理时不使用dropout
        pred_length=checkpoint["prediction_horizon"],
        d_model=config.get("d_model", 64),
        nhead=config.get("nhead", 4),
        num_encoder_layers=config.get("num_encoder_layers", 3),
        dim_feedforward=config.get("dim_feedforward", 256),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scaler, metrics


def predict_with_dl_model(model, scaler, prices, seq_length, pred_length):
    """使用深度学习模型进行滚动预测"""
    import torch
    scaled, _ = scale_data(prices, scaler=scaler)
    last_seq = scaled[-seq_length:].reshape(1, seq_length, 1)

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(last_seq).to(device)
        if hasattr(model, "attention"):
            pred, _ = model(x)
        else:
            pred = model(x)
        pred = pred.cpu().numpy().flatten()

    return inverse_scale(pred, scaler)


# ==================== 主界面 ====================
def main():
    # 标题
    st.markdown('<h1 class="main-header">📈 TS Price Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#666;">'
        '股票/加密货币价格预测系统 | LSTM + Transformer + Prophet'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 配置 / Settings")

        available_symbols = get_available_symbols()
        selected_symbol = st.selectbox(
            "选择标的 / Select Symbol",
            available_symbols,
            index=0,
            format_func=lambda x: f"{x} ({DEFAULT_SYMBOLS.get(x, 'Unknown')})",
        )

        st.subheader("模型选择 / Model Selection")
        use_lstm = st.checkbox("LSTM (Attention)", value=True, disabled=not check_model_exists("lstm", selected_symbol))
        use_transformer = st.checkbox("Transformer (Trend Decomposition)", value=True, disabled=not check_model_exists("transformer", selected_symbol))
        use_prophet = st.checkbox("Prophet (Baseline)", value=True, disabled=not check_model_exists("prophet", selected_symbol))

        if not (use_lstm or use_transformer or use_prophet):
            st.warning("没有选中的模型！请先训练模型。/ No model selected! Please train models first.")

        st.subheader("预测参数 / Prediction Parameters")
        pred_days = st.slider("预测天数 / Prediction Days", 1, 30, PREDICTION_HORIZON)

        show_indicators = st.checkbox("显示技术指标 / Show Technical Indicators", value=False)
        show_volume = st.checkbox("显示成交量 / Show Volume", value=True)

    # 加载数据
    try:
        df = load_symbol_data(selected_symbol)
    except FileNotFoundError:
        st.error(f"数据文件不存在，请先运行 `python fetch_data.py --symbol {selected_symbol}` 获取数据")
        st.stop()

    # ==================== Tab 1: 价格走势与预测 ====================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 价格走势 & 预测 / Price & Prediction",
        "📊 模型评估 / Model Evaluation",
        "🔍 技术指标 / Technical Analysis",
        "📋 模型对比 / Model Comparison",
    ])

    with tab1:
        st.subheader(f"{selected_symbol} - 价格走势与预测")

        fig = make_subplots(
            rows=2 if show_volume else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3] if show_volume else [1.0],
        )

        # 历史价格
        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="历史价格 / Historical",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=1, col=1,
        )

        # 收盘价折线
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["close"],
                name="收盘价 / Close",
                line=dict(color="#1e88e5", width=1.5),
                opacity=0.7,
            ),
            row=1, col=1,
        )

        # 模型预测
        last_date = df["Date"].iloc[-1]
        prediction_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=pred_days,
            freq="B",
        )

        colors = {"lstm": "#ff6f00", "transformer": "#7c4dff", "prophet": "#00c853"}

        if use_lstm:
            model, scaler, metrics = load_pytorch_model("lstm", selected_symbol)
            if model is not None:
                pred = predict_with_dl_model(model, scaler, df["close"].values, SEQUENCE_LENGTH, pred_days)
                fig.add_trace(
                    go.Scatter(
                        x=prediction_dates[:len(pred)],
                        y=pred,
                        name=f"LSTM预测 (MAE: {metrics['MAE']})",
                        line=dict(color=colors["lstm"], width=2, dash="dash"),
                        mode="lines+markers",
                    ),
                    row=1, col=1,
                )

        if use_transformer:
            model, scaler, metrics = load_pytorch_model("transformer", selected_symbol)
            if model is not None:
                pred = predict_with_dl_model(model, scaler, df["close"].values, SEQUENCE_LENGTH, pred_days)
                fig.add_trace(
                    go.Scatter(
                        x=prediction_dates[:len(pred)],
                        y=pred,
                        name=f"Transformer预测 (MAE: {metrics['MAE']})",
                        line=dict(color=colors["transformer"], width=2, dash="dot"),
                        mode="lines+markers",
                    ),
                    row=1, col=1,
                )

        if use_prophet:
            prophet_model, prophet_metrics = load_prophet_model(selected_symbol)
            if prophet_model is not None:
                from prophet_model import predict_prophet
                prophet_pred = predict_prophet(prophet_model, periods=pred_days)
                fig.add_trace(
                    go.Scatter(
                        x=prophet_pred["Date"],
                        y=prophet_pred["Predicted"],
                        name=f"Prophet预测 (MAE: {prophet_metrics['MAE']})",
                        line=dict(color=colors["prophet"], width=2, dash="dashdot"),
                        mode="lines+markers",
                    ),
                    row=1, col=1,
                )
                # 置信区间
                fig.add_trace(
                    go.Scatter(
                        x=prophet_pred["Date"],
                        y=prophet_pred["Upper_Bound"],
                        name="Prophet上界 / Upper",
                        line=dict(color=colors["prophet"], width=0),
                        showlegend=False,
                    ),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=prophet_pred["Date"],
                        y=prophet_pred["Lower_Bound"],
                        name="Prophet下界 / Lower",
                        fill="tonexty",
                        fillcolor="rgba(0, 200, 83, 0.1)",
                        line=dict(color=colors["prophet"], width=0),
                        showlegend=False,
                    ),
                    row=1, col=1,
                )

        # 成交量
        if show_volume and "volume" in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df["Date"],
                    y=df["volume"],
                    name="成交量 / Volume",
                    marker_color="rgba(100, 100, 200, 0.3)",
                    showlegend=False,
                ),
                row=2 if show_volume else 1, col=1,
            )

        fig.update_layout(
            height=700 if show_volume else 550,
            template="plotly_white",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(title_text="日期 / Date")
        fig.update_yaxes(title_text="价格 / Price ($)", row=1, col=1)
        if show_volume:
            fig.update_yaxes(title_text="成交量 / Volume", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    # ==================== Tab 2: 模型评估 ====================
    with tab2:
        st.subheader("模型评估指标 / Model Evaluation Metrics")

        all_metrics = {}
        if use_lstm:
            _, _, metrics = load_pytorch_model("lstm", selected_symbol)
            if metrics:
                all_metrics["LSTM"] = metrics
        if use_transformer:
            _, _, metrics = load_pytorch_model("transformer", selected_symbol)
            if metrics:
                all_metrics["Transformer"] = metrics
        if use_prophet:
            _, metrics = load_prophet_model(selected_symbol)
            if metrics:
                all_metrics["Prophet"] = metrics

        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics).T
            metrics_df = metrics_df[["MAE", "RMSE", "MAPE", "R2", "Direction_Accuracy"]]
            metrics_df.columns = ["MAE", "RMSE", "MAPE(%)", "R²", "方向准确率(%)"]
            st.dataframe(
                metrics_df.style.format("{:.4f}").background_gradient(
                    subset=["MAE", "RMSE", "MAPE(%)"],
                    cmap="RdYlGn_r",
                ).background_gradient(
                    subset=["R²", "方向准确率(%)"],
                    cmap="RdYlGn",
                ),
                use_container_width=True,
                height=300,
            )

            # 指标对比柱状图
            col1, col2 = st.columns(2)
            with col1:
                fig_metrics = go.Figure()
                for model_name, model_metrics in all_metrics.items():
                    fig_metrics.add_trace(go.Bar(
                        name=model_name,
                        x=["MAE", "RMSE"],
                        y=[model_metrics["MAE"], model_metrics["RMSE"]],
                    ))
                fig_metrics.update_layout(barmode="group", title="MAE & RMSE 对比", template="plotly_white")
                st.plotly_chart(fig_metrics, use_container_width=True)
            with col2:
                fig_dir = go.Figure()
                models = list(all_metrics.keys())
                dir_accs = [all_metrics[m]["Direction_Accuracy"] for m in models]
                fig_dir.add_trace(go.Bar(
                    x=models,
                    y=dir_accs,
                    text=[f"{d:.1f}%" for d in dir_accs],
                    textposition="auto",
                    marker_color=["#ff6f00", "#7c4dff", "#00c853"][:len(models)],
                ))
                fig_dir.update_layout(title="方向准确率对比 / Direction Accuracy", template="plotly_white")
                fig_dir.update_yaxes(range=[0, 100])
                st.plotly_chart(fig_dir, use_container_width=True)

    # ==================== Tab 3: 技术分析 ====================
    with tab3:
        st.subheader("技术指标分析 / Technical Analysis")

        tech_cols = [c for c in df.columns if c not in ["Date", "open", "high", "low", "close", "volume"]]
        if tech_cols:
            selected_indicators = st.multiselect(
                "选择指标 / Select Indicators",
                tech_cols,
                default=tech_cols[:4],
            )

            if selected_indicators:
                fig_tech = make_subplots(
                    rows=min(len(selected_indicators), 3),
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                )
                for i, col in enumerate(selected_indicators[:3]):
                    fig_tech.add_trace(
                        go.Scatter(
                            x=df["Date"],
                            y=df[col],
                            name=col,
                        ),
                        row=i + 1, col=1,
                    )
                fig_tech.update_layout(height=300 * min(len(selected_indicators), 3), template="plotly_white")
                st.plotly_chart(fig_tech, use_container_width=True)

    # ==================== Tab 4: 模型对比 ====================
    with tab4:
        st.subheader("模型综合对比 / Comprehensive Model Comparison")

        if all_metrics:
            # 雷达图对比
            categories = ["MAE", "RMSE", "MAPE", "R²", "Dir_Acc"]
            fig_radar = go.Figure()

            for model_name, model_metrics in all_metrics.items():
                # 归一化指标到0-1范围用于雷达图
                values = [
                    1 - model_metrics["MAE"] / max(m["MAE"] for m in all_metrics.values()),
                    1 - model_metrics["RMSE"] / max(m["RMSE"] for m in all_metrics.values()),
                    1 - model_metrics["MAPE"] / max(m["MAPE"] for m in all_metrics.values()),
                    model_metrics["R2"],
                    model_metrics["Direction_Accuracy"] / 100,
                ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    name=model_name,
                    fill="toself",
                    opacity=0.3,
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="模型综合能力雷达图 / Model Capability Radar",
                template="plotly_white",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")
        st.markdown(
            "**创新点 / Innovations:**\n"
            "1. LSTM + Attention机制 - 自动关注重要时间步\n"
            "2. Transformer + 趋势分解 - 捕捉长期趋势和短期波动\n"
            "3. 技术指标特征增强 - SMA/EMA/RSI/MACD/布林带\n"
            "4. 多模型集成对比 - 综合评估预测能力\n"
            "5. 方向准确率指标 - 衡量涨跌方向预测能力"
        )


if __name__ == "__main__":
    main()
