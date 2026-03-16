import yfinance as yf
from utils.logger import get_logger
import pandas as pd
import numpy as np

logger = get_logger("fundamental_agent")

def analyze_fundamentals(ticker: str) -> dict:
    """Enhanced Fundamental Analyst Agent with debug logs and hybrid fallback."""
    print(f"--- Running Fundamental Agent for {ticker} ---")
    try:
        stock = yf.Ticker(ticker)
        
        # Pre-fetch financial statements
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        info = stock.info or {}
        
        print(f"DEBUG: Retrieved info for {ticker}")
        if financials.empty:
            print(f"DEBUG: Financials empty for {ticker}, using income_stmt")
            financials = income_stmt
        
        metrics = {
            "revenue_growth": None,
            "pe_ratio": None,
            "roe": None,
            "debt_to_equity": None,
            "operating_margin": None
        }

        def get_metric_from_df(df, possible_keys):
            if df is None or df.empty: return None
            for key in possible_keys:
                if key in df.index:
                    return df.loc[key]
            return None

        # 1. Revenue Growth (Direct Calculation & Info Fallback)
        try:
            rev_data = get_metric_from_df(financials, ["Total Revenue", "TotalRevenue", "Operating Revenue"])
            if rev_data is not None and len(rev_data) >= 2:
                current_rev = rev_data.iloc[0]
                prev_rev = rev_data.iloc[1]
                if prev_rev and prev_rev != 0:
                    metrics["revenue_growth"] = (current_rev - prev_rev) / prev_rev
            
            if metrics["revenue_growth"] is None:
                metrics["revenue_growth"] = info.get("revenueGrowth")
        except: pass

        # 2. Operating Margin
        try:
            op_inc = get_metric_from_df(financials, ["Operating Income", "OperatingIncome"])
            total_rev = get_metric_from_df(financials, ["Total Revenue", "TotalRevenue"])
            if op_inc is not None and total_rev is not None and total_rev.iloc[0] != 0:
                metrics["operating_margin"] = op_inc.iloc[0] / total_rev.iloc[0]
            
            if metrics["operating_margin"] is None:
                metrics["operating_margin"] = info.get("operatingMargins")
        except: pass

        # 3. ROE
        try:
            net_inc = get_metric_from_df(financials, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
            equity = get_metric_from_df(balance_sheet, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity", "Total Equity"])
            if net_inc is not None and equity is not None and equity.iloc[0] != 0:
                metrics["roe"] = net_inc.iloc[0] / equity.iloc[0]
            
            if metrics["roe"] is None:
                metrics["roe"] = info.get("returnOnEquity")
        except: pass

        # 4. Debt to Equity
        try:
            equity_val = get_metric_from_df(balance_sheet, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity", "Total Equity"])
            debt = get_metric_from_df(balance_sheet, ["Total Debt", "Net Debt"])
            if debt is not None and equity_val is not None and equity_val.iloc[0] != 0:
                metrics["debt_to_equity"] = debt.iloc[0] / equity_val.iloc[0]
            
            if metrics["debt_to_equity"] is None:
                # info returns debtToEquity usually as a percentage or raw number
                de = info.get("debtToEquity")
                if de is not None:
                    metrics["debt_to_equity"] = de / 100.0 if de > 5 else de
        except: pass

        # 5. PE Ratio
        try:
            metrics["pe_ratio"] = info.get("trailingPE") or info.get("forwardPE")
            if metrics["pe_ratio"] is None:
                price = info.get("currentPrice") or info.get("previousClose")
                eps = info.get("trailingEps")
                if price and eps and eps != 0:
                    metrics["pe_ratio"] = price / eps
        except: pass

        print(f"DEBUG: Calculated metrics for {ticker}: {metrics}")

        # Validate and Format
        validated_metrics = {}
        processed_metrics = {}
        for k, v in metrics.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                num_val = float(v)
                processed_metrics[k] = num_val
                if k in ["revenue_growth", "roe", "operating_margin"]:
                    validated_metrics[k] = f"{round(num_val * 100, 2):.2f}%"
                else:
                    validated_metrics[k] = str(round(num_val, 2))
            else:
                validated_metrics[k] = "Unavailable"
                processed_metrics[k] = None

        # Determine Score (0-10)
        score = 0.0
        if processed_metrics.get("revenue_growth", 0) and processed_metrics["revenue_growth"] > 0.1: score += 2.5
        if processed_metrics.get("roe", 0) and processed_metrics["roe"] > 0.15: score += 2.5
        if processed_metrics.get("operating_margin", 0) and processed_metrics["operating_margin"] > 0.15: score += 2.5
        de = processed_metrics.get("debt_to_equity", 3.0) or 3.0
        if de < 1.0: score += 2.5
        elif de < 2.0: score += 1.0

        print(f"DEBUG: Fundamental Score for {ticker}: {score}")

        return {
            "agent": "fundamental",
            "ticker": ticker,
            "metrics": {
                "long_name": info.get("longName") or ticker,
                **validated_metrics
            },
            "fundamental_score": round(min(score, 10.0), 1),
            "revenue_growth": metrics["revenue_growth"],
            "pe_ratio": metrics["pe_ratio"],
            "roe": metrics["roe"],
            "debt_to_equity": metrics["debt_to_equity"],
            "operating_margin": metrics["operating_margin"]
        }
    except Exception as e:
        print(f"ERROR: Fundamental Agent failed for {ticker}: {e}")
        logger.error(f"Fundamental analysis failed for {ticker}: {e}")
        return {
            "agent": "fundamental",
            "ticker": ticker,
            "metrics": {"error": "DATA_NOT_AVAILABLE"},
            "fundamental_score": 0.0
        }
