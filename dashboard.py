import requests
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Page config
st.set_page_config(
    page_title="Transformer Quant AI",
    layout="wide"
)

# Title
st.title(
    "📈 Transformer Quant AI Platform"
)

st.markdown(
    """
    Real-time Transformer-based
    quantitative forecasting system
    using:
    - macroeconomic features
    - wavelets
    - deep learning
    """
)

# Sidebar
st.sidebar.header(
    "Controls"
)

refresh = st.sidebar.button(
    "Generate Live Prediction"
)

# Backend URL
API_URL = (
    "https://sensex-transformer-system-1.onrender.com/live-predict"
)

# Main logic
if refresh:

    with st.spinner(
        "Fetching live market data..."
    ):

        response = requests.post(
            API_URL
        )

        result = response.json()

        # Debug response
        st.write("API Response:", result)

        if "predicted_return" not in result:

            st.error(
                "Backend API error."
            )

            st.stop()

        prediction = result[
            "predicted_return"
        ]

        signal = result[
            "signal"
        ]

    # Metrics
    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Return",
            f"{prediction:.4f}"
        )

    with col2:

        if signal == "BUY":

            st.success(
                f"Signal: {signal}"
            )

        elif signal == "SELL":

            st.error(
                f"Signal: {signal}"
            )

        else:

            st.warning(
                f"Signal: {signal}"
            )

    # Gauge chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction * 100,

            title={
                "text":
                "Expected Return %"
            },

            gauge={
                "axis": {
                    "range":
                    [-10, 10]
                }
            }
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Simulated equity curve
    np.random.seed(42)

    equity = np.cumprod(
        1 + np.random.normal(
            0.001,
            0.01,
            100
        )
    )

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            y=equity,
            mode="lines",
            name="Strategy Equity"
        )
    )

    fig2.update_layout(
        title="Strategy Equity Curve",
        xaxis_title="Trades",
        yaxis_title="Portfolio Value"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# Footer
st.markdown("---")

st.markdown(
    """
    Built with:
    - FastAPI
    - Streamlit
    - PyTorch
    - Wavelet Transforms
    - Transformer Architecture
    """
)