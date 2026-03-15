import yfinance as yf
from utils.logger import get_logger
import pandas as pd

logger = get_logger("fundamental_agent")

def analyze_fundamentals(ticker: str) -> dict:
    """Production Fundamental Analyst Agent."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # More robust ticker verification
        is_valid = info and (info.get("longName") or info.get("shortName") or info.get("symbol"))
        if not is_valid:
            # Last resort: check if price exists
            hist = stock.history(period="1d")
            if hist.empty:
                logger.warning(f"No fundamental data for {ticker}")
                return {"error": "DATA_NOT_AVAILABLE", "details": "Ticker not found on Yahoo Finance"}

        # Required metrics per prompt
        revenue_growth = info.get("revenueGrowth") or info.get("earningsQuarterlyGrowth")
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        roe = info.get("returnOnEquity")
        debt_to_equity = info.get("debtToEquity")
        eps_growth = info.get("earningsGrowth")
        op_margin = info.get("operatingMargins") or info.get("profitMargins")

        # Normalize Debt to Equity (Yahoo often returns it as a percentage like 35 for 0.35)
        norm_debt_equity = debt_to_equity / 100.0 if debt_to_equity is not None else None

        # Scoring Logic (Max 10.0)
        score = 0.0
        if revenue_growth is not None and revenue_growth > 0.08: score += 2 # Adjusted threshold to be slightly more lenient
        if roe is not None and roe > 0.12: score += 2
        if norm_debt_equity is not None and norm_debt_equity < 1.0: score += 2 # Standard debt limit
        if pe_ratio is not None and 0 < pe_ratio < 30: score += 2
        if eps_growth is not None and eps_growth > 0: score += 2
        
        # If score is 0 but we have some data, give a base score to differentiate from "No Data"
        if score == 0: score = 1.0

        return {
            "agent": "fundamental",
            "ticker": ticker,
            "metrics": {
                "long_name": info.get("longName") or info.get("shortName") or ticker,
                "revenue_growth": round(revenue_growth, 4) if revenue_growth is not None else None,
                "pe_ratio": round(pe_ratio, 2) if pe_ratio is not None else None,
                "roe": round(roe, 4) if roe is not None else None,
                "debt_to_equity": round(norm_debt_equity, 4) if norm_debt_equity is not None else None,
                "eps_growth": round(eps_growth, 4) if eps_growth is not None else None,
                "operating_margin": round(op_margin, 4) if op_margin is not None else None
            },
            "fundamental_score": score
        }
    except Exception as e:
        logger.error(f"Fundamental analysis failed for {ticker}: {e}")
        return {"error": f"DATA_NOT_AVAILABLE: {str(e)}"}
