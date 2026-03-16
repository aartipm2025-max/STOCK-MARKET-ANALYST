import yfinance as yf
import pandas as pd
from utils.logger import get_logger

logger = get_logger("market_context_agent")

def analyze_market_context(ticker: str) -> dict:
    """Agent to analyze broader market and sector context."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        sector = info.get("sector", "Unknown Sector")
        industry = info.get("industry", "Unknown Industry")
        
        # 1. Nifty 50 Trend (last 5 sessions)
        nifty = yf.Ticker("^NSEI")
        nifty_hist = nifty.history(period="1mo")
        nifty_trend = "Stable"
        change = 0
        if not nifty_hist.empty and len(nifty_hist) >= 5:
            last_5 = nifty_hist['Close'].tail(5)
            change = (last_5.iloc[-1] - last_5.iloc[0]) / last_5.iloc[0]
            if change > 0.005: nifty_trend = "Bullish (Up " + f"{change*100:.2f}%)"
            elif change < -0.005: nifty_trend = "Bearish (Down " + f"{change*100:.2f}%)"
            else: nifty_trend = "Side-ways / Neutral"
        else:
            # Fallback if 1mo fails
            nifty_hist = nifty.history(period="5d")
            if not nifty_hist.empty and len(nifty_hist) >= 2:
                last_prices = nifty_hist['Close']
                change = (last_prices.iloc[-1] - last_prices.iloc[0]) / last_prices.iloc[0]
                if change > 0.005: nifty_trend = "Bullish (Up " + f"{change*100:.2f}%)"
                elif change < -0.005: nifty_trend = "Bearish (Down " + f"{change*100:.2f}%)"
                else: nifty_trend = "Side-ways / Neutral"

        # 2. Sector Performance
        sector_map = {
            "Technology": "^CNXIT",
            "Financial Services": "^CNXBANK",
            "Auto": "^CNXAUTO",
            "FMCG": "^CNXFMCG",
            "Healthcare": "^CNXPHARMA"
        }
        sector_index = sector_map.get(sector, "^NSEI")
        s_ticker = yf.Ticker(sector_index)
        s_hist = s_ticker.history(period="1mo")
        sector_perf = "Stable"
        s_change = 0
        if not s_hist.empty and len(s_hist) >= 5:
            s_last_5 = s_hist['Close'].tail(5)
            s_change = (s_last_5.iloc[-1] - s_last_5.iloc[0]) / s_last_5.iloc[0]
            comparison = "Outperforming market" if s_change > change else "Underperforming market"
            sector_perf = f"{sector}: {comparison} ({s_change*100:.2f}% change)"

        # 3. Peer Comparison
        returns_1m = 0
        stock_hist = stock.history(period="1mo")
        if not stock_hist.empty:
            returns_1m = (stock_hist['Close'].iloc[-1] - stock_hist['Close'].iloc[0]) / stock_hist['Close'].iloc[0]
        
        peer_comp = f"Stock 1-month return ({returns_1m*100:.2f}%) is " + ("above" if returns_1m > s_change else "below") + " sector/market average."

        # Scoring (0-10)
        context_score = 5.0
        if change > 0: context_score += 1.5
        if s_change > change: context_score += 1.5
        if returns_1m > s_change: context_score += 2.0
        
        return {
            "agent": "market_context",
            "ticker": ticker,
            "sector": sector,
            "nifty_trend": nifty_trend,
            "sector_performance": sector_perf,
            "peer_comparison": peer_comp,
            "context_score": round(min(context_score, 10.0), 1)
        }
    except Exception as e:
        logger.error(f"Market context analysis failed for {ticker}: {e}")
        return {"error": f"DATA_NOT_AVAILABLE: {str(e)}"}
