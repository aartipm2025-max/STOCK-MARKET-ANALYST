import streamlit as st
import requests
import json
import time
import pandas as pd

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Market Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Premium UI ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Dark Mode Theme */
[data-testid="stAppViewContainer"] {
    background-color: #0f172a !important;
    color: #f8fafc !important;
}

[data-testid="stHeader"] {
    background-color: rgba(15, 23, 42, 0.8) !important;
}

/* Sidebar Styling - Higher Contrast */
[data-testid="stSidebar"] {
    background-color: #1e293b !important;
    border-right: 1px solid #334155 !important;
    color: #f8fafc !important;
}

.sidebar-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #38bdf8 !important; /* Bright Blue */
    display: flex;
    align-items: center;
    gap: 10px;
}

.sidebar-subtitle {
    font-size: 0.85rem;
    color: #cbd5e1 !important;
    margin-bottom: 2rem;
}

/* Main Content Header - Bright White */
.stock-header {
    font-size: 2.5rem;
    font-weight: 800;
    color: #ffffff !important;
    margin-bottom: 1.5rem;
    text-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
}

/* Score Cards - High Contrast */
.score-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}

.score-card:hover {
    transform: translateY(-5px);
    border-color: #38bdf8;
}

.score-label {
    font-size: 0.85rem;
    color: #cbd5e1 !important;
    text-transform: uppercase;
    font-weight: 600;
}

.score-value {
    font-size: 2rem;
    font-weight: 800;
    color: #facc15 !important; /* Bright Yellow */
}

.score-denom {
    font-size: 1rem;
    color: #94a3b8 !important;
}

/* Analysis Content - Bright White */
.analysis-text {
    font-size: 1.15rem;
    line-height: 1.7;
    color: #ffffff !important;
    background: rgba(30, 41, 59, 0.5);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #38bdf8;
}

/* Metric Items - High Visibility */
.metric-item-label {
    font-size: 0.9rem;
    color: #94a3b8 !important;
    font-weight: 500;
}

.metric-item-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff !important;
}

h1, h2, h3, h4, h5, h6, label, p {
    color: #ffffff !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #94a3b8 !important;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
}

/* Chat Input styling at bottom */
.stTextInput input {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-size: 1.1rem !important;
}

.stTextInput input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}

</style>
""", unsafe_allow_html=True)

# ── Sidebar Setup ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ Market Analyst AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Multi-agent AI system for Indian stock market analysis</div>', unsafe_allow_html=True)
    
    st.markdown("### Analysis Mode")
    analysis_mode = st.radio(
        "Select Mode",
        ["Chat", "Single Stock", "Compare Stocks", "Portfolio"],
        label_visibility="collapsed"
    )
    
    st.markdown("### Settings")
    api_url_setting = st.text_input("API URL", "http://localhost:8000")
    
    st.markdown('<div class="footer-text">Built with LangGraph + Gemini + FastAPI</div>', unsafe_allow_html=True)

# ── Main UI State ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ── Helper: Call API ─────────────────────────────────────────────────────────
def call_market_api(query, mode=None):
    # If mode is explicitly selected in UI, hint it in the query
    hinted_query = query
    if mode == "Single Stock": hinted_query = f"[MODE: Single Stock Analysis] {query}"
    elif mode == "Compare Stocks": hinted_query = f"[MODE: Stock Comparison] {query}"
    elif mode == "Portfolio": hinted_query = f"[MODE: Portfolio Analysis] {query}"
    
    endpoint = f"{api_url_setting}/analyze"
    try:
        response = requests.post(endpoint, json={"query": hinted_query}, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"API Connection Error: {e}")
        return None

# ── Content Logic ─────────────────────────────────────────────────────────────
if st.session_state.results:
    res = st.session_state.results
    intent = res.get("intent", "unknown")
    tickers = res.get("tickers", [])
    agg_data_list = res.get("aggregated_data", [])
    
    # Header
    main_title = tickers[0] if len(tickers) == 1 else ("Comparison" if intent == "comparison" else "Market Analysis")
    st.markdown(f'<div class="stock-header">📈 {main_title}</div>', unsafe_allow_html=True)
    
    if agg_data_list:
        if intent == "comparison":
            st.subheader("Comparison Ranking")
            comp_df = pd.DataFrame([
                {
                    "Ticker": item['ticker'],
                    "Fund.": item['scores']['fundamental'],
                    "Tech.": item['scores']['technical'],
                    "Sent.": item['scores']['sentiment'],
                    "Total": item['final_score'],
                    "Rec.": item['recommendation']
                } for item in agg_data_list
            ])
            st.table(comp_df)
            
        elif intent == "portfolio":
            st.subheader("Portfolio Health")
            # Usually the aggregator puts health stats in the analysis part or extra keys
            # For now, show scores of all holdings
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.write("**Holdings Analysis**")
                for item in agg_data_list:
                    st.write(f"- {item['ticker']}: {item['final_score']}/10 ({item['recommendation']})")
            
        else: # Single Stock or Unknown with data
            raw_data = agg_data_list[0]
            scores = raw_data.get("scores", {"fundamental": 0, "technical": 0, "sentiment": 0})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="score-card"><div class="score-label">Fundamental</div><div class="score-value">{scores.get("fundamental", 0)} <span class="score-denom">/ 10</span></div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="score-card"><div class="score-label">Technical</div><div class="score-value">{scores.get("technical", 0)} <span class="score-denom">/ 10</span></div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="score-card"><div class="score-label">Sentiment</div><div class="score-value">{scores.get("sentiment", 0)} <span class="score-denom">/ 10</span></div></div>', unsafe_allow_html=True)

        # Tabs for detailed metrics (only for relevant tickers)
        if intent in ["single_stock", "comparison"]:
            st.subheader("Metric Deep Dive")
            # If comparison, let user pick which one to see in detail
            target_ticker = st.selectbox("Select ticker for details", tickers) if len(tickers) > 1 else tickers[0]
            target_data = next((x for x in agg_data_list if x['ticker'] == target_ticker), None)
            
            if target_data:
                tab_fund, tab_tech, tab_sent = st.tabs(["Fundamentals", "Technicals", "Sentiment"])
                with tab_fund:
                    f_metrics = target_data.get("raw_metrics", {}).get("f", {}).get("metrics", {})
                    if f_metrics:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f'<div class="metric-item"><div class="metric-item-label">PE Ratio</div><div class="metric-item-value">{f_metrics.get("pe_ratio", "N/A")}</div></div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="metric-item"><div class="metric-item-label">EPS Growth</div><div class="metric-item-value">{f_metrics.get("eps_growth", 0)*100:.1f}%</div></div>', unsafe_allow_html=True)
                        with c2:
                            st.markdown(f'<div class="metric-item"><div class="metric-item-label">Revenue Growth</div><div class="metric-item-value">{f_metrics.get("revenue_growth", 0)*100:.1f}%</div></div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="metric-item"><div class="metric-item-label">Debt/Equity</div><div class="metric-item-value">{f_metrics.get("debt_to_equity", "N/A")}</div></div>', unsafe_allow_html=True)
                    else: st.info("No fundamental metrics.")
                with tab_tech:
                    t_metrics = target_data.get("raw_metrics", {}).get("t", {}).get("indicators", {})
                    if t_metrics:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f'<div class="metric-item"><div class="metric-item-label">Price</div><div class="metric-item-value">{t_metrics.get("current_price", "N/A")}</div></div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="metric-item"><div class="metric-item-label">RSI</div><div class="metric-item-value">{t_metrics.get("rsi", "N/A")}</div></div>', unsafe_allow_html=True)
                        with c2:
                            st.markdown(f'<div class="metric-item"><div class="metric-item-label">SMA 50</div><div class="metric-item-value">{t_metrics.get("sma_50", "N/A")}</div></div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="metric-item"><div class="metric-item-label">MACD</div><div class="metric-item-value">{t_metrics.get("macd_signal", "N/A")}</div></div>', unsafe_allow_html=True)
                    else: st.info("No technical indicators.")
                with tab_sent:
                    s_metrics = target_data.get("raw_metrics", {}).get("s", {})
                    if s_metrics:
                        st.markdown(f"**Score: {s_metrics.get('sentiment_score', 5.0)}/10**")
                        st.write(s_metrics.get("summary", "No details."))
                    else: st.info("No sentiment data.")

    # Narrative Summary
    st.subheader("AI Recommendation Report")
    st.markdown(f'<div class="analysis-text">{res.get("analysis", "")}</div>', unsafe_allow_html=True)

else:
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #ffffff;'>Ready to analyze the market?</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Select a mode on the left and enter your query below.</p>", unsafe_allow_html=True)

# ── Footer / Input ────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)

with st.form("query_form", clear_on_submit=True):
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_query = st.text_input("Enter your query", placeholder="e.g. Analyze Reliance Industries", label_visibility="collapsed")
    with col_btn:
        submit_btn = st.form_submit_button("Analyze")

if submit_btn and user_query:
    st.session_state.results = None  # Clear previous results
    with st.spinner("🤖 Analyzing Market Data..."):
        st.session_state.results = call_market_api(user_query, mode=analysis_mode)
        st.rerun()
