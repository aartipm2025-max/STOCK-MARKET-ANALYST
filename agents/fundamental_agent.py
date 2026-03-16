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
        # income_stmt and financials are often the same in newer yfinance versions
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        info = stock.info or {}
        
        metrics = {
            "revenue_growth": None,
            "pe_ratio": None,
            "roe": None,
            "debt_to_equity": None,
            "operating_margin": None
        }

        # 1. Revenue Growth Extraction (from Income Statement)
        try:
            if not income_stmt.empty and "Total Revenue" in income_stmt.index:
                revs = income_stmt.loc["Total Revenue"]
                if len(revs) >= 2:
                    # Calculate year-over-year growth
                    metrics["revenue_growth"] = (revs.iloc[0] - revs.iloc[1]) / revs.iloc[1]
        except Exception as e:
            logger.warning(f"Error calculating Revenue Growth for {ticker}: {e}")

        # 2. Operating Margin Extraction (from Income Statement)
        try:
            if not income_stmt.empty and "Operating Income" in income_stmt.index and "Total Revenue" in income_stmt.index:
                op_inc = income_stmt.loc["Operating Income"]
                total_rev = income_stmt.loc["Total Revenue"]
                if not op_inc.empty and not total_rev.empty and total_rev.iloc[0] != 0:
                    metrics["operating_margin"] = op_inc.iloc[0] / total_rev.iloc[0]
        except Exception as e:
            logger.warning(f"Error calculating Operating Margin for {ticker}: {e}")

        # 3. ROE Extraction (Net Income / Total Equity)
        try:
            equity = None
            for k in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
                if not balance_sheet.empty and k in balance_sheet.index:
                    equity = balance_sheet.loc[k].iloc[0]
                    break
            
            if equity and not income_stmt.empty and "Net Income" in income_stmt.index:
                net_inc = income_stmt.loc["Net Income"].iloc[0]
                if equity != 0:
                    metrics["roe"] = net_inc / equity
        except Exception as e:
            logger.warning(f"Error calculating ROE for {ticker}: {e}")

        # 4. Debt to Equity Extraction
        try:
            if equity:
                total_debt = None
                for k in ["Total Debt", "Net Debt"]:
                    if not balance_sheet.empty and k in balance_sheet.index:
                        total_debt = balance_sheet.loc[k].iloc[0]
                        break
                if total_debt is not None and equity != 0:
                    metrics["debt_to_equity"] = total_debt / equity
        except Exception as e:
            logger.warning(f"Error calculating Debt to Equity for {ticker}: {e}")

        # 5. PE Ratio (Price / EPS)
        try:
            # Try to get EPS from info first as it's more standard, fallback to calculation
            eps = info.get("trailingEps")
            if eps is None and not income_stmt.empty and "Net Income" in income_stmt.index:
                ni = income_stmt.loc["Net Income"].iloc[0]
                shares = None
                for k in ["Ordinary Shares Number", "Share Cap", "Total Shares Outstanding"]:
                    if not balance_sheet.empty and k in balance_sheet.index:
                        shares = balance_sheet.loc[k].iloc[0]
                        break
                if shares and shares != 0:
                    eps = ni / shares
            
            price = info.get("currentPrice") or info.get("previousClose")
            if not price:
                hist = stock.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[0]
            
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
