import yfinance as yf
import pandas as pd
from utils.logger import get_logger

logger = get_logger("market_context_agent")

def analyze_market_context(ticker: str) -> dict:
    """Agent to analyze broader market and sector context with 5-day Nifty trend."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        sector = info.get("sector", "Unknown Sector")
        
        # 1. Nifty 50 Trend (last 5 sessions as requested)
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="5d")
        
        nifty_trend = "Data not available"
        change_val = 0
        
        if not hist.empty and len(hist) >= 2:
            first_close = hist['Close'].iloc[0]
            last_close = hist['Close'].iloc[-1]
            change_val = ((last_close - first_close) / first_close) * 100
            
            trend_dir = "Bullish" if change_val > 0.5 else ("Bearish" if change_val < -0.5 else "Neutral")
            nifty_trend = f"{trend_dir} ({change_val:.2f}% over 5 days)"
        
        # 2. Sector Performance (Simplified for reliability)
        sector_perf = f"Sector: {sector} analysis relies on broader index trend ({trend_dir if 'trend_dir' in locals() else 'Neutral'})"
        
        # 3. Peer Comparison
        price_hist = stock.history(period="5d")
        stock_perf = "N/A"
        if not price_hist.empty and len(price_hist) >= 2:
            s_first = price_hist['Close'].iloc[0]
            s_last = price_hist['Close'].iloc[-1]
            s_change = ((s_last - s_first) / s_first) * 100
            rel_perf = "Outperforming" if s_change > change_val else "Underperforming"
            stock_perf = f"{rel_perf} Nifty 50 (Stock: {s_change:.2f}% vs Market: {change_val:.2f}%)"
            
        # Context Score calculation
        score = 5.0
        if change_val > 0: score += 2.0
        if 's_change' in locals() and s_change > change_val: score += 3.0

        return {
            "agent": "market_context",
            "ticker": ticker,
            "sector": sector,
            "nifty_trend": nifty_trend,
            "sector_performance": sector_perf,
            "peer_comparison": stock_perf,
            "context_score": round(min(score, 10.0), 1)
        }
    except Exception as e:
        logger.error(f"Market context analysis failed: {e}")
        return {
            "agent": "market_context",
            "ticker": ticker,
            "nifty_trend": "Unavailable",
            "sector_performance": "Unavailable",
            "peer_comparison": "Unavailable",
            "context_score": 0.0
        }
