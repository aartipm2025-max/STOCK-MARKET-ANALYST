import yfinance as yf
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

/* Main area text adjustments */
.analysis-card {
    background: white;
    padding: 2rem;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

.section-header {
    color: #1e293b;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 1rem;
    border-bottom: 2px solid #f1f5f9;
    padding-bottom: 0.5rem;
}

.recommendation-banner {
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 800;
    margin-top: 2rem;
}

.score-card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    text-align: center;
    transition: transform 0.2s;
}

.score-card:hover {
    transform: translateY(-5px);
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Navigation")
    analysis_mode = st.radio(
        "Select Mode",
        ["Chat", "Single Stock", "Compare Stocks", "Portfolio"],
        index=0
    )
    st.markdown("---")
    st.markdown("### Stock Suggestions")
    st.markdown("- **Reliance**\n- **TCS**\n- **Infosys**\n- **HDFC Bank**\n- **Eternal (Zomato)**")

use_integrated = True
api_url_setting = "http://localhost:8000"

# Main config

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

# ── Header & Greeting ─────────────────────────────────────────────────────────
st.markdown("""
<div class="welcome-container" style="margin-top: 2rem; text-align: center;">
    <h1 style="color: #7c3aed; font-weight: 800; font-size: 3.5rem; margin-bottom: 0.5rem;">Market Analyst AI</h1>
    <h2 style="color: #475569; font-size: 1.8rem; margin: 0; padding-top: 10px;">Welcome back, Aarti</h2>
    <p style="color: #64748b; font-size: 1.1rem; margin-top: 0.5rem;">Multi-agent AI system for Indian stock market analysis</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 1rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

# ── Content Logic ─────────────────────────────────────────────────────────────
user_query = None
submit_btn = False

if st.session_state.results:
    res = st.session_state.results
    intent = res.get("intent", "unknown")
    tickers = res.get("tickers", [])
    agg_data_list = res.get("aggregated_data", [])
    
    # Clean Ticker Display
    display_ticker = tickers[0] if tickers else "Unknown"
    
    # Extract Date from the first valid record
    analysis_date = "N/A"
    if agg_data_list:
        analysis_date = agg_data_list[0].get("analysis_date", "N/A")

    st.markdown(f"""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='color: #1e293b; font-size: 3rem; margin-bottom: 0px;'>{display_ticker}</h1>
            <p style='color: #64748b; font-size: 1.2rem; font-weight: 500; margin-bottom: 5px;'>Comprehensive Market Analysis Report</p>
            <p style='color: #475569; font-size: 1rem; font-weight: 600;'>Data Analyzed Up To: {analysis_date}</p>
        </div>
    """, unsafe_allow_html=True)

    # Upgrade 3: Stock Price Chart
    try:
        if tickers:
            symbol = tickers[0]
            ticker_obj = yf.Ticker(symbol)
            hist = ticker_obj.history(period="6mo")
            if not hist.empty:
                st.markdown(f"### Price Performance: {symbol} (6 Months)")
                st.line_chart(hist["Close"])
            else:
                st.warning(f"No historical price data found for {symbol}")
    except Exception as e:
        st.error(f"Error rendering price chart: {e}")

    if agg_data_list:
        if intent == "comparison":
            st.markdown("#### Market Comparison Table")
            comp_df = pd.DataFrame([
                {
                    "Ticker": item['ticker'],
                    "Fund.": item['scores']['fundamental'],
                    "Tech.": item['scores']['technical'],
                    "Sent.": item['scores']['sentiment'],
                    "Context": item['scores'].get('market_context', 0.0),
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
            # Main Score Overview (Horizontal Metrics Row)
            target_ticker = st.selectbox("Select ticker for deep dive", tickers) if len(tickers) > 1 else tickers[0]
            target_data = next((x for x in agg_data_list if x['ticker'] == target_ticker), agg_data_list[0])
            scores = target_data.get("scores", {"fundamental": 0, "technical": 0, "sentiment": 0})
            
            # Color logic for metric cards
            def get_score_color(val):
                if val >= 7: return "#059669" # Green
                if val >= 4: return "#d97706" # Yellow/Orange
                return "#dc2626" # Red

            f_color = get_score_color(scores.get("fundamental", 0))
            t_color = get_score_color(scores.get("technical", 0))
            s_color = get_score_color(scores.get("sentiment", 0))
            m_color = get_score_color(scores.get("market_context", 0))

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f'<div class="score-card"><div style="color: #64748b; font-size: 0.8rem; font-weight: 700;">FUNDAMENTAL</div><div style="font-size: 2.2rem; font-weight: 800; color: {f_color};">{scores.get("fundamental", 0)}</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="score-card"><div style="color: #64748b; font-size: 0.8rem; font-weight: 700;">TECHNICAL</div><div style="font-size: 2.2rem; font-weight: 800; color: {t_color};">{scores.get("technical", 0)}</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="score-card"><div style="color: #64748b; font-size: 0.8rem; font-weight: 700;">SENTIMENT</div><div style="font-size: 2.2rem; font-weight: 800; color: {s_color};">{scores.get("sentiment", 0)}</div></div>', unsafe_allow_html=True)
            with c4: st.markdown(f'<div class="score-card"><div style="color: #64748b; font-size: 0.8rem; font-weight: 700;">CONTEXT</div><div style="font-size: 2.2rem; font-weight: 800; color: {m_color};">{scores.get("market_context", 0)}</div></div>', unsafe_allow_html=True)

            # Upgrade 5: Rating Summary Panel
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### RATING SUMMARY")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Fundamental", scores.get("fundamental", 0))
            r2.metric("Technical", scores.get("technical", 0))
            r3.metric("Sentiment", scores.get("sentiment", 0))
            r4.metric("Market Context", scores.get("market_context", 0))
            st.markdown("---")

            # Parsing the LLM report into sectioned cards
            full_report = res.get("analysis", "")
            
            # Extract segments by triple hyphens
            segments = full_report.split("---")
            
            # Map sections to ensure order and avoid duplicates
            report_sections = {
                "fundamental": None,
                "technical": None,
                "sentiment": None,
                "market_context": None,
                "narrative": None,
                "risks": None,
                "recommendation": None,
                "horizon": None
            }
            
            for segment in segments:
                seg_upper = segment.upper().strip()
                if "FUNDAMENTAL ANALYSIS" in seg_upper:
                    report_sections["fundamental"] = segment.replace("**FUNDAMENTAL ANALYSIS**", "").strip()
                elif "TECHNICAL ANALYSIS" in seg_upper:
                    report_sections["technical"] = segment.replace("**TECHNICAL ANALYSIS**", "").strip()
                elif "SENTIMENT ANALYSIS" in seg_upper:
                    report_sections["sentiment"] = segment.replace("**SENTIMENT ANALYSIS**", "").strip()
                elif "MARKET CONTEXT" in seg_upper:
                    report_sections["market_context"] = segment.replace("**MARKET CONTEXT**", "").strip()
                elif "AI NARRATIVE SUMMARY" in seg_upper:
                    report_sections["narrative"] = segment.replace("**AI NARRATIVE SUMMARY**", "").strip()
                elif "RISK FACTORS" in seg_upper:
                    report_sections["risks"] = segment.replace("**RISK FACTORS**", "").strip()
                elif "FINAL RECOMMENDATION" in seg_upper:
                    report_sections["recommendation"] = segment.replace("**FINAL RECOMMENDATION**", "").strip()
                elif "INVESTMENT HORIZON" in seg_upper:
                    report_sections["horizon"] = segment.replace("**INVESTMENT HORIZON**", "").strip()

            # Render in fixed order
            if report_sections["fundamental"]:
                st.markdown('<div class="analysis-card"><div class="section-header">📊 Fundamental Analysis</div>', unsafe_allow_html=True)
                st.markdown(report_sections["fundamental"])
                st.markdown('</div>', unsafe_allow_html=True)

            if report_sections["technical"]:
                st.markdown('<div class="analysis-card"><div class="section-header">📈 Technical Analysis</div>', unsafe_allow_html=True)
                st.markdown(report_sections["technical"])
                st.markdown('</div>', unsafe_allow_html=True)

            if report_sections["sentiment"]:
                st.markdown('<div class="analysis-card"><div class="section-header">💬 Sentiment Analysis</div>', unsafe_allow_html=True)
                st.markdown(report_sections["sentiment"])
                st.markdown('</div>', unsafe_allow_html=True)

            if report_sections["market_context"]:
                st.markdown('<div class="analysis-card"><div class="section-header">🌐 Market Context</div>', unsafe_allow_html=True)
                st.markdown(report_sections["market_context"])
                st.markdown('</div>', unsafe_allow_html=True)

            if report_sections["narrative"]:
                st.markdown('<div class="analysis-card"><div class="section-header">🤖 AI Narrative Summary</div>', unsafe_allow_html=True)
                st.markdown(report_sections["narrative"])
                st.markdown('</div>', unsafe_allow_html=True)

            if report_sections["risks"]:
                st.markdown('<div class="analysis-card"><div class="section-header">⚠️ Risk Factors</div>', unsafe_allow_html=True)
                st.markdown(report_sections["risks"])
                st.markdown('</div>', unsafe_allow_html=True)

            if report_sections["recommendation"]:
                # Parse recommendation and confidence
                rec_lines = report_sections["recommendation"].split("\n")
                rec_call = rec_lines[0].replace("**", "").strip()
                conf_level = ""
                for line in rec_lines:
                    if "Confidence Level:" in line:
                        conf_level = line.replace("Confidence Level:", "").strip()
                        break
                
                # Recommendation color logic
                rec_bg = "#fef2f2" # Light Red
                rec_border = "#fecaca"
                rec_text_color = "#dc2626"
                
                if "STRONG BUY" in rec_call or "BUY" in rec_call:
                    rec_bg = "#ecfdf5" # Light Green
                    rec_border = "#d1fae5"
                    rec_text_color = "#059669"
                elif "HOLD" in rec_call:
                    rec_bg = "#fffbeb" # Light Yellow
                    rec_border = "#fef3c7"
                    rec_text_color = "#d97706"
                
                st.markdown(f"""
                    <div class="recommendation-banner" style="background: {rec_bg}; border: 2px solid {rec_border}; color: {rec_text_color}; padding: 2.5rem; border-radius: 20px;">
                        <div style="font-size: 1rem; font-weight: 700; opacity: 0.9; margin-bottom: 0.75rem; letter-spacing: 1px;">INSTITUTIONAL CALL</div>
                        <div style="font-size: 3.5rem; font-weight: 950; letter-spacing: -2px; margin-bottom: 1rem; line-height: 1;">{rec_call}</div>
                        <div style="font-size: 1.2rem; font-weight: 700; background: rgba(255,255,255,0.6); display: inline-block; padding: 6px 16px; border-radius: 50px; border: 1px solid rgba(0,0,0,0.05);">
                            Confidence Level: {conf_level}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            if report_sections["horizon"]:
                st.markdown('<br>', unsafe_allow_html=True)
                st.markdown('<div class="analysis-card"><div class="section-header">⏳ Investment Horizon</div>', unsafe_allow_html=True)
                st.markdown(report_sections["horizon"])
                st.markdown('</div>', unsafe_allow_html=True)


else:
    st.markdown("""
    <div style='text-align: center;'>
        <p style='color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;'>What would you like to analyze today?</p>
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
