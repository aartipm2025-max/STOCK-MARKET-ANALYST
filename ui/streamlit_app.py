import streamlit as st
import requests
import json
import time
import pandas as pd
import os
import sys
from dotenv import load_dotenv

# Ensure the project root is in sys.path for cloud deployment
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

# Try to load local env, but don't fail if not present (Cloud)
try:
    load_dotenv()
except:
    pass

from backend.langgraph_workflow.graph import build_graph

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Market Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Premium NVST-Style UI ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Main Container */
.main {
    background-color: #f1f5f9;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-color: #f1f5f9;
}

/* Custom Header Bar */
.top-nav {
    background-color: #7c3aed; /* Deep Lavender */
    padding: 0.75rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-radius: 0 0 15px 15px;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.brand {
    font-size: 1.8rem;
    font-weight: 900;
    letter-spacing: -1px;
}

/* Global Content Alignment */
.content-container {
    max-width: 1000px;
    margin: 0 auto;
    padding: 0 1rem;
    display: flex;
    flex-direction: column;
    align-items: center;
}

div[data-testid="stVerticalBlock"] > div:has(.content-container) {
    width: 100% !important;
    display: flex;
    justify-content: center;
}

.analysis-text {
    font-size: 1.1rem;
    line-height: 1.7;
    color: #334155 !important;
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #7c3aed;
    max-width: 1000px;
    margin: 0 auto;
}

.metric-item-label {
    font-size: 0.85rem;
    color: #64748b !important;
    font-weight: 500;
}

.metric-item-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1e293b !important;
}

/* Hide Sidebar elements */
[data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

# Sidebar removed as requested
use_integrated = True
api_url_setting = "http://localhost:8000"

# ── Main UI State ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ── Helper: Call API ─────────────────────────────────────────────────────────
def call_market_api(query, mode=None, force_integrated=False):
    # If mode is explicitly selected in UI, hint it in the query
    hinted_query = query
    if mode == "Single Stock": hinted_query = f"[MODE: Single Stock Analysis] {query}"
    elif mode == "Compare Stocks": hinted_query = f"[MODE: Stock Comparison] {query}"
    elif mode == "Portfolio": hinted_query = f"[MODE: Portfolio Analysis] {query}"
    
    if not force_integrated:
        endpoint = f"{api_url_setting}/analyze"
        try:
            response = requests.post(endpoint, json={"query": hinted_query}, timeout=5) # Short timeout for check
            if response.status_code == 200:
                return response.json()
        except:
            pass # Fallback to integrated
            
    # Integrated Mode Logic
    try:
        graph = build_graph()
        initial_state = {
            "query": hinted_query,
            "intent": "",
            "tickers": [],
            "fundamental_data": {},
            "technical_data": {},
            "sentiment_data": {},
            "portfolio_data": {},
            "final_analysis": "",
            "aggregated_data": []
        }
        result_state = graph.invoke(initial_state)
        return {
            "intent": result_state.get("intent", "unknown"),
            "tickers": result_state.get("tickers", []),
            "analysis": result_state.get("final_analysis", "Analysis failed."),
            "aggregated_data": result_state.get("aggregated_data", [])
        }
    except Exception as e:
        st.error(f"Integrated Agent Error: {e}")
        return None

# ── Global Central Container ──────────────────────────────────────────────────
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# ── Header & Branding ────────────────────────────────────────────────────────
st.markdown("""
<div class="welcome-container" style="margin-top: 2rem;">
    <h1 style="color: #7c3aed; font-weight: 800; font-size: 3.5rem; margin-bottom: 0.5rem;">Market Analyst AI</h1>
    <p style="color: #64748b; font-size: 1.1rem;">Multi-agent AI system for Indian stock market analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Analysis Mode Bar (Reinforced Center) ───────────────────────────────────
mode_c1, mode_c2, mode_c3 = st.columns([1, 3, 1])
with mode_c2:
    analysis_mode = st.radio(
        "Select Mode",
        ["Chat", "Single Stock", "Compare Stocks", "Portfolio"],
        horizontal=True,
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Content Logic ─────────────────────────────────────────────────────────────
if st.session_state.results:
    res = st.session_state.results
    intent = res.get("intent", "unknown")
    tickers = res.get("tickers", [])
    agg_data_list = res.get("aggregated_data", [])
    
    st.markdown(f"### 📈 {tickers[0] if len(tickers) == 1 else ('Comparison Dashboard' if intent == 'comparison' else 'Analysis Report')}")
    
    if agg_data_list:
        if intent == "comparison":
            st.markdown("#### Market Comparison Table")
            comp_df = pd.DataFrame([
                {
                    "Ticker": item['ticker'],
                    "Fund.": item['scores']['fundamental'],
                    "Tech.": item['scores']['technical'],
                    "Sent.": item['scores']['sentiment'],
                    "Final": item['final_score'],
                    "Recommendation": item['recommendation']
                } for item in agg_data_list
            ])
            st.table(comp_df)
            st.markdown("<br>", unsafe_allow_html=True)
            
        elif intent == "portfolio":
            st.subheader("Portfolio Breakdown")
            for item in agg_data_list:
                st.markdown(f"**{item['ticker']}**: Score {item['final_score']}/10 - *{item['recommendation']}*")
            st.markdown("<br>", unsafe_allow_html=True)

        if len(agg_data_list) > 0:
            # Main Score Overview (for first or selected ticker)
            target_ticker = st.selectbox("Select ticker for deep dive", tickers) if len(tickers) > 1 else tickers[0]
            target_data = next((x for x in agg_data_list if x['ticker'] == target_ticker), agg_data_list[0])
            
            scores = target_data.get("scores", {"fundamental": 0, "technical": 0, "sentiment": 0})
            
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center;"><div style="color: #64748b; font-size: 0.8rem; font-weight: 700;">FUNDAMENTAL</div><div style="font-size: 2.2rem; font-weight: 800; color: #7c3aed;">{scores.get("fundamental", 0)}</div><div style="color: #94a3b8; font-size: 0.8rem;">/ 10</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center;"><div style="color: #64748b; font-size: 0.8rem; font-weight: 700;">TECHNICAL</div><div style="font-size: 2.2rem; font-weight: 800; color: #7c3aed;">{scores.get("technical", 0)}</div><div style="color: #94a3b8; font-size: 0.8rem;">/ 10</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center;"><div style="color: #64748b; font-size: 0.8rem; font-weight: 700;">SENTIMENT</div><div style="font-size: 2.2rem; font-weight: 800; color: #7c3aed;">{scores.get("sentiment", 0)}</div><div style="color: #94a3b8; font-size: 0.8rem;">/ 10</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Metric Deep Dive Tabs
            tab_fund, tab_tech, tab_sent = st.tabs(["📊 Fundamentals", "📈 Technicals", "💬 Sentiment"])
            
            with tab_fund:
                f_metrics = target_data.get("raw_metrics", {}).get("f", {}).get("metrics", {})
                if f_metrics:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f'<div class="metric-item-label">PE Ratio</div><div class="metric-item-value">{f_metrics.get("pe_ratio", "N/A")}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-item-label">EPS Growth</div><div class="metric-item-value">{f_metrics.get("eps_growth", 0)*100:.1f}%</div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="metric-item-label">Revenue Growth</div><div class="metric-item-value">{f_metrics.get("revenue_growth", 0)*100:.1f}%</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-item-label">Debt/Equity</div><div class="metric-item-value">{f_metrics.get("debt_to_equity", "N/A")}</div>', unsafe_allow_html=True)
                else: st.info("Fundamental metrics not available for this ticker.")
                
            with tab_tech:
                t_metrics = target_data.get("raw_metrics", {}).get("t", {}).get("indicators", {})
                if t_metrics:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f'<div class="metric-item-label">Current Price</div><div class="metric-item-value">{t_metrics.get("current_price", "N/A")}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-item-label">RSI (14)</div><div class="metric-item-value">{t_metrics.get("rsi", "N/A")}</div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="metric-item-label">SMA 50</div><div class="metric-item-value">{t_metrics.get("sma_50", "N/A")}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-item-label">MACD Signal</div><div class="metric-item-value">{t_metrics.get("macd_signal", "N/A")}</div>', unsafe_allow_html=True)
                else: st.info("Technical indicators not available.")

            with tab_sent:
                s_metrics = target_data.get("raw_metrics", {}).get("s", {})
                if s_metrics:
                    st.markdown(f"**Sentiment Analysis Score: {s_metrics.get('sentiment_score', 5.0)}/10**")
                    st.write(s_metrics.get("summary", "No detailed sentiment analysis available."))
                else: st.info("Sentiment data not available.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### 🤖 AI Recommendation Report")
    st.markdown(f'<div class="analysis-text">{res.get("analysis", "")}</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align: center; margin-top: 1rem;'>
        <h2 style='color: #475569; font-size: 2rem;'>Welcome back, Aarti</h2>
        <p style='color: #64748b; font-size: 1.1rem; margin-bottom: 2.5rem;'>What would you like to analyze today?</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Perfectly Centered Search Container
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        user_query = st.text_input("Search or Ask AI...", key="main_input", placeholder="e.g. Compare TCS and Infosys", label_visibility="collapsed")
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        # Center the button within this column
        b_c1, b_c2, b_c3 = st.columns([1, 1, 1])
        with b_c2:
            submit_btn = st.button("Generate Analysis", use_container_width=True)

# ── Metrics Bar ──────────────────────────────────────────────────────────────
# (Optional: Only show if results exist or keep as personal dash)
# Moving metrics below if needed, but keeping for now as it gives it a dash feel.

if submit_btn and user_query:
    st.session_state.results = None
    with st.spinner("🤖 AI Agents at work..."):
        st.session_state.results = call_market_api(user_query, mode=analysis_mode, force_integrated=use_integrated)
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True) # End Global Container
