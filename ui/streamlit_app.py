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

/* Cards & Tables */
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    height: 100%;
    border: 1px solid #e2e8f0;
}

.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #475569;
    margin-bottom: 1rem;
    display: flex;
    justify-content: space-between;
}

/* Sentiment Feed (Right Side) */
.feed-item {
    border-bottom: 1px solid #f1f5f9;
    padding: 1rem 0;
}

.user-tag {
    font-size: 0.85rem;
    font-weight: 700;
    color: #1e293b;
}

.tweet-text {
    font-size: 0.9rem;
    color: #64748b;
    line-height: 1.4;
}

/* Recommendation Cards */
.rec-card {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    min-width: 200px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

/* Inputs */
.stTextInput input {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
}

/* Hide Sidebar elements for cleaner look */
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

# ── Header & Branding ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; margin-bottom: 1.5rem;">
    <h1 style="color: #7c3aed; font-weight: 800; font-size: 3.5rem; margin-bottom: 0.5rem;">Market Analyst AI</h1>
    <p style="color: #64748b; font-size: 1.1rem;">Multi-agent AI system for Indian stock market analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Analysis Mode Bar (Centered) ─────────────────────────────────────────────
mode_col1, mode_col2, mode_col3 = st.columns([1, 2, 1])
with mode_col2:
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
    
    # Dash Layout
    main_col, feed_col = st.columns([2.5, 1])
    
    with main_col:
        st.markdown(f"### 📈 {tickers[0] if tickers else 'Analysis Dashboard'}")
    
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
            
        else: 
            raw_data = agg_data_list[0]
            scores = raw_data.get("scores", {"fundamental": 0, "technical": 0, "sentiment": 0})
            st.markdown("""<div style="font-size: 0.9rem; font-weight: 700; margin-bottom: 0.5rem; color: #475569;">Performance Overview</div>""", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="card"><div style="color: #64748b; font-size: 0.8rem;">FUNDAMENTAL</div><div style="font-size: 1.5rem; font-weight: 700;">{scores.get("fundamental", 0)}/10</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="card"><div style="color: #64748b; font-size: 0.8rem;">TECHNICAL</div><div style="font-size: 1.5rem; font-weight: 700;">{scores.get("technical", 0)}/10</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="card"><div style="color: #64748b; font-size: 0.8rem;">SENTIMENT</div><div style="font-size: 1.5rem; font-weight: 700;">{scores.get("sentiment", 0)}/10</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### AI Narrative Summary")
        st.markdown(f"""<div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0; color: #334155; line-height: 1.6;">{res.get("analysis", "")}</div>""", unsafe_allow_html=True)

    with feed_col:
        st.markdown("""<div class="card"><div class="card-title">Market Sentiment</div>
        <div class="feed-item"><span class="user-tag">@MarketWatch</span><br><span class="tweet-text">Nifty shows strong resistance at 22,500. Bulls looking for a breakout.</span></div>
        <div class="feed-item"><span class="user-tag">@FinIntel</span><br><span class="tweet-text">Tech stocks leading the rally today after positive global cues.</span></div>
        <div class="feed-item"><span class="user-tag">@AnalystLisa</span><br><span class="tweet-text">Keep an eye on banking sector results tomorrow. Expecting volatility.</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Recommendations for you")
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1: st.markdown('<div class="rec-card"><div style="font-weight:700;">TITAN</div><div style="color:#10b981;">INR 3,450 (+2.1%)</div><div style="font-size:0.75rem; color:#64748b;">91% Analysts Buy</div></div>', unsafe_allow_html=True)
    with rc2: st.markdown('<div class="rec-card"><div style="font-weight:700;">RELIANCE</div><div style="color:#10b981;">INR 2,980 (+0.8%)</div><div style="font-size:0.75rem; color:#64748b;">84% Analysts Buy</div></div>', unsafe_allow_html=True)
    with rc3: st.markdown('<div class="rec-card"><div style="font-weight:700;">HDFCBANK</div><div style="color:#ef4444;">INR 1,420 (-0.5%)</div><div style="font-size:0.75rem; color:#64748b;">76% Analysts Buy</div></div>', unsafe_allow_html=True)
    with rc4: st.markdown('<div class="rec-card"><div style="font-weight:700;">ASIANPAINT</div><div style="color:#10b981;">INR 3,120 (+1.2%)</div><div style="font-size:0.75rem; color:#64748b;">92% Analysts Buy</div></div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align: center; margin-top: 2rem;'>
        <h2 style='color: #475569;'>Welcome back, Aarti</h2>
        <p style='color: #64748b; font-size: 1.1rem;'>Enter a stock name or a question to start your analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Place search in the center for welcome screen
    search_col_1, search_col_2, search_col_3 = st.columns([1, 4, 1])
    with search_col_2:
        user_query = st.text_input("Search or Ask AI...", key="main_input", placeholder="e.g. Compare TCS and Infosys", label_visibility="collapsed")
        sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])
        with sub_col2:
            submit_btn = st.button("Generate Analysis", use_container_width=True)

# ── Metrics Bar ──────────────────────────────────────────────────────────────
# (Optional: Only show if results exist or keep as personal dash)
# Moving metrics below if needed, but keeping for now as it gives it a dash feel.

if submit_btn and user_query:
    st.session_state.results = None
    with st.spinner("🤖 AI Agents at work..."):
        st.session_state.results = call_market_api(user_query, mode=analysis_mode, force_integrated=use_integrated)
        st.rerun()
