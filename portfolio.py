import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIG: Page and global styling ---
st.set_page_config(page_title="Optimal Portfolio Management — Finance Modeling", layout="wide")

st.markdown(
    """
    <style>
    /* ----- GLOBAL STYLES ----- */
    html, body, [class*="css"] {
        background-color: #050915;
        color: #E1E6ED;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp { 
        background-color: #050915; 
    }
    
    /* ----- SIDEBAR STYLING ----- */
    .stSidebar {
        background-color: #0a3d62;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5,
    .stSidebar p, .stSidebar label, .stSidebar span {
        color: white !important;
    }
    
    /* Hapus border, shadow, dan background dari container form jika ada */
    section[data-testid="stSidebar"] form[data-testid="stForm"],
    section[data-testid="stSidebar"] form[data-testid="stForm"] * {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }
    
    /* Hapus border pada input fields */
    input[data-baseweb="input"], textarea[data-baseweb="textarea"] {
        border: none !important;
        box-shadow: none !important;
        background-color: #0a3d62 !important;
        color: white !important;
    }
    input[data-baseweb="input"]:focus, textarea[data-baseweb="textarea"]:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* ----- HEADINGS ----- */
    h1 {
        font-size: 26px !important;
        color: #F0F4F8;
    }
    h4 {
        font-size: 18px !important;
        color: #F0F4F8;
    }
    
    /* ----- METRIC BOXES ----- */
    .metric-box {
        background-color: #2c3e50;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 1rem;
    }
    .metric-box-rule {
        background-color: #1e2a47;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 1rem;
    }
    
    /* ----- FOOTER TEXT ----- */
    .footer-text {
        text-align: center;
        font-size: 17px;
        font-weight: bold;
        color: white;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    /* Membuat tombol Submit di sidebar melebar dan beri jarak sedikit dari atas */
    section[data-testid="stSidebar"] .stButton button {
        width: 100% !important;
        margin-top: 6px !important; /* Ganti 5px sesuai selera */
    }

    /* Samakan dengan project ARIMA + sedikit penyesuaian */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }
    
    /* ----- DIVIDER STYLE ----- */
    .stMarkdown hr {
        border-top: 2px solid #34495e;
    }

    /* Ubah tampilan tombol submit di sidebar dengan warna yang lebih gelap */
    section[data-testid="stSidebar"] .stButton button {
        background-color: #d35400 !important; /* Dark orange */
        color: white !important;             /* Teks putih kontras */
        width: 100% !important;
        padding: 10px 0 !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 5px !important;
        cursor: pointer !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: #ba4a00 !important; /* Sedikit lebih gelap saat hover */
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# --- SIDEBAR: Input Parameters (Tanpa Form Container) ---
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.header("📁 Portfolio Settings")

ticker_input = st.sidebar.text_input(
    "Enter Stock Tickers (Yahoo Finance Format)",
    value="BBCA.JK, BBRI.JK, INDF.JK, ASII.JK"
)

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%)", 
    min_value=0.0, max_value=100.0, 
    value=6.0, step=0.1
)

start_date = st.sidebar.date_input(
    "Start Date", value=pd.to_datetime("2021-12-12")
)

end_date = st.sidebar.date_input(
    "End Date", value=datetime.today() - timedelta(days=1)
)

simulations = st.sidebar.number_input(
    "Monte Carlo Simulations", 
    min_value=5000, max_value=20000, 
    value=10000
)

# Tombol submit langsung tanpa menggunakan form container
submitted = st.sidebar.button("Submit")

# --- MAIN LOGIC: Process input jika tombol submit ditekan ---
if submitted:
    # Proses ticker
    stock_list = [ticker.strip() for ticker in ticker_input.split(",") if ticker.strip() != ""]
    if len(stock_list) < 2:
        st.warning("Please enter at least 2 tickers.")
        st.stop()

    st.title("📊 Optimal Portfolio Management")
    st.markdown("<hr style='margin-top:0; border-color:#34495e; margin-bottom:2rem;'>", unsafe_allow_html=True)

    # --- FETCH DATA WITH yfinance ---
    stocks = {}
    for ticker in stock_list:
        try:
            stocks[ticker] = yf.download(ticker, start=start_date, end=end_date)[['Close']]
        except Exception as e:
            st.warning(f"Warning: Could not fetch data for {ticker}: {e}")
            st.stop()

    price_df = pd.DataFrame()
    for ticker, df in stocks.items():
        if not df.empty:
            price_df[ticker] = df['Close']
        else:
            st.warning(f"Warning: No data for ticker {ticker}. Check the ticker or date range.")
            st.stop()
    price_df.dropna(inplace=True)
    stock_returns = price_df.pct_change().dropna()

    # --- MONTE CARLO SIMULATIONS FOR PORTFOLIO OPTIMIZATION ---
    scenarios = simulations
    weights_array = np.zeros((scenarios, len(stock_list)))
    returns_array = np.zeros(scenarios)
    volatility_array = np.zeros(scenarios)
    sharpe_array = np.zeros(scenarios)

    np.random.seed(3)
    for i in range(scenarios):
        weights = np.random.random(len(stock_list))
        weights /= np.sum(weights)
        weights_array[i] = weights
        returns_array[i] = np.sum(stock_returns.mean() * 252 * weights)
        volatility_array[i] = np.sqrt(np.dot(weights.T, np.dot(stock_returns.cov() * 252, weights)))
        sharpe_array[i] = (returns_array[i] - risk_free_rate / 100) / volatility_array[i]

    max_idx = sharpe_array.argmax()
    optimal_weights = weights_array[max_idx]
    optimal_return = returns_array[max_idx]
    optimal_volatility = volatility_array[max_idx]
    optimal_sharpe = sharpe_array[max_idx]

    # --- DISPLAY KEY METRICS ---

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-box'>📈 Sharpe Ratio<br>{optimal_sharpe:.2f}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'>💰 Annual Return<br>{optimal_return:.2%}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-box'>📉 Volatility<br>{optimal_volatility:.2%}</div>", unsafe_allow_html=True)

    # --- STOCK PERFORMANCE CHART ---
    fig_performance = go.Figure()
    for i, column in enumerate(price_df.columns):
        fig_performance.add_trace(go.Scatter(
            x=price_df.index, y=price_df[column], mode='lines', name=stock_list[i]
        ))
    fig_performance.update_layout(
        title="Equal-Weighted Portfolio Stock Performance",
        xaxis_title="",
        yaxis_title="Total Value",
        plot_bgcolor='#050915',
        paper_bgcolor='#050915',
        font=dict(color='white')
    )
    st.plotly_chart(fig_performance, use_container_width=True)

    # --- PIE CHART & SCATTER PLOT ---
    # Garis pemisah sebelum Pie & Scatter
    st.markdown("<hr style='border-color:#34495e; margin:2rem 0;'>", unsafe_allow_html=True)

    weights_df = pd.DataFrame({
        'Stock': stock_list,
        'Weight': optimal_weights
    })
    pie_chart = go.Figure(data=[go.Pie(
        labels=weights_df['Stock'], values=weights_df['Weight'], hole=0.4
    )])
    pie_chart.update_layout(
        title="Optimal Portfolio Allocation", 
        font=dict(color='white'), paper_bgcolor="#050915"
    )

    col4, col5 = st.columns([1, 2])
    with col4:
        st.plotly_chart(pie_chart, use_container_width=True)
    with col5:
        scatter_fig = go.Figure()
        scatter_fig.add_trace(go.Scatter(
            x=volatility_array, y=returns_array, mode='markers',
            marker=dict(color=sharpe_array, colorscale='Viridis', showscale=True),
            name='Portfolios'
        ))
        scatter_fig.add_trace(go.Scatter(
            x=[optimal_volatility], y=[optimal_return], mode='markers',
            marker=dict(color='orange', size=12, line=dict(width=2, color='black')),
            name='Optimal'
        ))
        scatter_fig.update_layout(
            title="Portfolio Optimization — Return vs Volatility",
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            plot_bgcolor='#050915', paper_bgcolor='#050915', font=dict(color='white')
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    # --- TABLE OF OPTIMAL WEIGHTS ---
    # Garis pemisah sebelum table
    st.markdown("<hr style='border-color:#34495e; margin:2rem 0;'>", unsafe_allow_html=True)

    st.subheader("🔢 Optimal Portfolio Weights")
    st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))

    st.markdown(
    f"""
    <div style="background-color:#1e2a47; padding: 10px; border-radius: 5px; margin-top: 10px;">
        <p style="margin: 0; color: white; font-size: 16px;">
            The above allocation represents the optimal weights based on the highest 
            <strong>Sharpe Ratio</strong> achieved from <strong>{scenarios}</strong> Monte Carlo simulation trials.
            This suggests that this portfolio is expected to deliver the best risk-adjusted return.
        </p>
    </div>
    """,
    unsafe_allow_html=True
    )
    
        # --- ANALYSIS ---
    # Garis pemisah sebelum analysis
    st.markdown("<hr style='border-color:#34495e; margin:2rem 0;'>", unsafe_allow_html=True)

    st.markdown(f"""
### 📝 Analysis:
The **Sharpe Ratio** measures risk-adjusted return by subtracting the risk-free rate from the portfolio’s return and dividing by its volatility. A higher Sharpe Ratio means the portfolio yields more return per unit of risk.

**Program Functionality:**
- **Monte Carlo Simulations:** Generate {scenarios} random portfolio allocations.
- **Risk Evaluation:** Identify the portfolio with the highest Sharpe Ratio.
- **Dynamic Analysis:** Input any valid stock tickers; the tool fetches data from Yahoo Finance and computes optimal allocations.
    """)
    
    # Tentukan rating berdasarkan nilai optimal_sharpe
    if optimal_sharpe < 1:
        rating = "🔻 Poor"
    elif optimal_sharpe < 2:
        rating = "⚖️ Acceptable"
    elif optimal_sharpe < 3:
        rating = "👍 Good"
    else:
        rating = "🌟 Excellent"

    # Tampilkan kesimpulan dalam container
    st.markdown(f"""
<div style="background-color:#06452d; padding: 10px; border-radius: 5px; margin-top: 10px;">
    <p style="margin: 0; color: white; font-size: 16px;">
        <strong>Conclusion:</strong> The portfolio's Sharpe Ratio is <strong>{optimal_sharpe:.2f}</strong>, 
        classified as <strong>{rating}</strong>.
    </p>
</div>
""", unsafe_allow_html=True)
    
    # --- SHARPE RATIO BENCHMARKS ---
    st.markdown("<hr style='border-color:#34495e; margin:2rem 0;'>", unsafe_allow_html=True)

    col_rule1, col_rule2, col_rule3, col_rule4 = st.columns(4)
    with col_rule1:
        st.markdown("<div class='metric-box-rule'>🔻 < 1<br>Poor</div>", unsafe_allow_html=True)
    with col_rule2:
        st.markdown("<div class='metric-box-rule'>⚖️ 1 to 2<br>Acceptable</div>", unsafe_allow_html=True)
    with col_rule3:
        st.markdown("<div class='metric-box-rule'>👍 2 to 3<br>Good</div>", unsafe_allow_html=True)
    with col_rule4:
        st.markdown("<div class='metric-box-rule'>🌟 > 3<br>Excellent</div>", unsafe_allow_html=True)
    
    # --- FOOTER ---
    st.markdown("""<hr style="border-top: 2px solid #2c3e50;">""", unsafe_allow_html=True)
    st.markdown("<div class='footer-text'>Created by Abida Massi</div>", unsafe_allow_html=True)
    
else:
    st.info("Please enter stock tickers and click 'Submit' in the sidebar to run the analysis.")
