import yfinance as yf
from utils.logger import get_logger
import pandas as pd
import numpy as np

logger = get_logger("fundamental_agent")

def analyze_fundamentals(ticker: str) -> dict:
    """Enhanced Fundamental Analyst Agent prioritizing statement data over ticker.info."""
    try:
        stock = yf.Ticker(ticker)
        
        # Pre-fetch financial statements
        inc = stock.financials
        bal = stock.balance_sheet
        if inc.empty: inc = stock.income_stmt
        info = stock.info or {}
        
        metrics = {
            "revenue_growth": None,
            "pe_ratio": None,
            "roe": None,
            "debt_to_equity": None,
            "operating_margin": None
        }

        def get_from_stmt(df, keys):
            if df is None or df.empty: return None
            for k in keys:
                if k in df.index:
                    series = df.loc[k]
                    if not series.empty: return series
            return None

        # 1. Revenue Growth Extraction
        try:
            rev_series = get_from_stmt(inc, ["Total Revenue", "TotalRevenue", "OperatingRevenue"])
            if rev_series is not None and len(rev_series) >= 2:
                metrics["revenue_growth"] = (rev_series.iloc[0] - rev_series.iloc[1]) / rev_series.iloc[1]
        except Exception as e:
            logger.warning(f"Error calculating Revenue Growth for {ticker}: {e}")

        # 2. Operating Margin Extraction
        try:
            op_inc = get_from_stmt(inc, ["Operating Income", "OperatingIncome"])
            total_rev = get_from_stmt(inc, ["Total Revenue", "TotalRevenue"])
            if op_inc is not None and total_rev is not None and total_rev.iloc[0] != 0:
                metrics["operating_margin"] = op_inc.iloc[0] / total_rev.iloc[0]
        except Exception as e:
            logger.warning(f"Error calculating Operating Margin for {ticker}: {e}")

        # 3. ROE Extraction
        try:
            equity = get_from_stmt(bal, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity", "Total Equity"])
            net_inc = get_from_stmt(inc, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
            if equity is not None and net_inc is not None and equity.iloc[0] != 0:
                metrics["roe"] = net_inc.iloc[0] / equity.iloc[0]
        except Exception as e:
            logger.warning(f"Error calculating ROE for {ticker}: {e}")

        # 4. Debt to Equity Extraction
        try:
            equity_val = equity.iloc[0] if equity is not None else None
            debt = get_from_stmt(bal, ["Total Debt", "Net Debt", "Total Liabilities Net Minority Interest"])
            if debt is not None and equity_val and equity_val != 0:
                metrics["debt_to_equity"] = debt.iloc[0] / equity_val
        except Exception as e:
            logger.warning(f"Error calculating Debt to Equity for {ticker}: {e}")

        # 5. PE Ratio (Price / EPS)
        try:
            eps = info.get("trailingEps")
            if eps is None:
                ni = get_from_stmt(inc, ["Net Income", "NetIncome"])
                shares = get_from_stmt(bal, ["Ordinary Shares Number", "Share Cap", "Total Shares Outstanding"])
                if ni is not None and shares is not None and shares.iloc[0] != 0:
                    eps = ni.iloc[0] / shares.iloc[0]
            
            price = info.get("currentPrice") or info.get("previousClose")
            if not price:
                hist = stock.history(period="1d")
                if not hist.empty: price = hist['Close'].iloc[0]
            
            if price and eps and eps != 0:
                metrics["pe_ratio"] = price / eps
            elif info.get("trailingPE"):
                metrics["pe_ratio"] = info.get("trailingPE")
        except Exception as e:
            logger.warning(f"Error calculating PE Ratio for {ticker}: {e}")

        # Fallback to info for any missing metrics
        if metrics["revenue_growth"] is None: metrics["revenue_growth"] = info.get("revenueGrowth")
        if metrics["roe"] is None: metrics["roe"] = info.get("returnOnEquity")
        if metrics["debt_to_equity"] is None: metrics["debt_to_equity"] = info.get("debtToEquity")
        if metrics["operating_margin"] is None: metrics["operating_margin"] = info.get("operatingMargins")

        # Data Validation Layer: Handle None values and prevent all empty
        validated_metrics = {}
        for key, val in metrics.items():
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                try:
                    num_val = float(val)
                    metrics[key] = num_val # Ensure it's a number for scoring
                    
                    # Formatting logic
                    if key in ["revenue_growth", "roe", "operating_margin"]:
                        # Convert to percentage format (e.g., 0.032 -> 3.20%)
                        validated_metrics[key] = f"{round(num_val * 100, 2):.2f}%"
                    elif key == "pe_ratio":
                        validated_metrics[key] = str(round(num_val, 2))
                    else:
                        validated_metrics[key] = str(round(num_val, 4))
                except:
                    validated_metrics[key] = "Unavailable from source"
                    metrics[key] = None
            else:
                validated_metrics[key] = "Unavailable from source"
                metrics[key] = None

        # Scoring Logic (Max 10.0)
        score = 0.0
        
        # Growth Strength (max 2.5)
        rev_g = metrics["revenue_growth"]
        if isinstance(rev_g, (int, float)):
            if rev_g > 0.15: score += 2.5
            elif rev_g > 0.05: score += 1.5
            elif rev_g > 0: score += 0.5
            
        # Profitability: ROE + Op Margin (max 2.5)
        p_score = 0.0
        roe_val = metrics["roe"]
        if isinstance(roe_val, (int, float)):
            if roe_val > 0.15: p_score += 1.25
            elif roe_val > 0.08: p_score += 0.75
        
        op_m = metrics["operating_margin"]
        if isinstance(op_m, (int, float)):
            if op_m > 0.20: p_score += 1.25
            elif op_m > 0.10: p_score += 0.75
        score += p_score
            
        # Financial Stability: Debt-to-Equity (max 2.5)
        de = metrics["debt_to_equity"]
        if isinstance(de, (int, float)):
            # Normalize if it's already in percentage (some API sources return it > 1 for 100%)
            if de > 10: de = de / 100.0
            if de < 0.5: score += 2.5
            elif de < 1.0: score += 1.5
            elif de < 2.0: score += 0.5
            
        # Valuation: P/E Ratio (max 2.5)
        pe = metrics["pe_ratio"]
        if isinstance(pe, (int, float)):
            if 0 < pe < 20: score += 2.5
            elif 20 <= pe < 40: score += 1.5
            elif 40 <= pe < 60: score += 0.5

        return {
            "agent": "fundamental",
            "ticker": ticker,
            "metrics": {
                "long_name": info.get("longName") or info.get("shortName") or ticker,
                **validated_metrics
            },
            "fundamental_score": round(min(score, 10.0), 1)
        }
    except Exception as e:
        logger.error(f"Fundamental analysis failed for {ticker}: {e}")
        return {"error": f"DATA_NOT_AVAILABLE: {str(e)}"}
