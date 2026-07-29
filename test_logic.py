import yfinance as yf
import pandas as pd
import numpy as np
import json
import traceback

def test_fundamental_logic(ticker):
    print(f"--- Testing Fundamental Logic for {ticker} ---")
    try:
        stock = yf.Ticker(ticker)
        
        # Pre-fetch separately
        try: financials = stock.financials
        except: 
            print("Failed to fetch financials")
            financials = pd.DataFrame()
        try: balance_sheet = stock.balance_sheet
        except: 
            print("Failed to fetch balance sheet")
            balance_sheet = pd.DataFrame()
        try: income_stmt = stock.income_stmt
        except: 
            print("Failed to fetch income statement")
            income_stmt = pd.DataFrame()
        try: info = stock.info or {}
        except:
            print("Failed to fetch info")
            info = {}
        
        print(f"DEBUG: Info keys count: {len(info)}")
        if financials.empty:
            print("DEBUG: Financials empty, using income_stmt")
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
            normalized_index = {str(k).lower().replace(" ", ""): k for k in df.index}
            for key in possible_keys:
                normalized_key = key.lower().replace(" ", "")
                if normalized_key in normalized_index:
                    return df.loc[normalized_index[normalized_key]]
            return None

        # 1. Revenue Growth
        try:
            rev_data = get_metric_from_df(financials, ["Total Revenue", "Operating Revenue", "TotalRevenue", "Total Interest Income", "Interest Income"])
            if rev_data is not None and len(rev_data) >= 2:
                current_rev = rev_data.iloc[0]
                prev_rev = rev_data.iloc[1]
                if prev_rev and prev_rev != 0:
                    metrics["revenue_growth"] = (current_rev - prev_rev) / prev_rev
            
            if metrics["revenue_growth"] is None:
                metrics["revenue_growth"] = info.get("revenueGrowth")
        except Exception as e:
            print(f"DEBUG: Revenue Growth error: {e}")

        # 2. Operating Margin
        try:
            op_inc = get_metric_from_df(financials, ["Operating Income", "Net Income From Continuing Operations", "Pretax Income"])
            total_rev = get_metric_from_df(financials, ["Total Revenue", "Operating Revenue", "Total Interest Income", "Interest Income"])
            if op_inc is not None and total_rev is not None and not total_rev.empty and total_rev.iloc[0] != 0:
                metrics["operating_margin"] = op_inc.iloc[0] / total_rev.iloc[0]
            
            if metrics["operating_margin"] is None:
                metrics["operating_margin"] = info.get("operatingMargins")
        except Exception as e:
            print(f"DEBUG: Op Margin error: {e}")

        # 3. ROE
        try:
            net_inc = get_metric_from_df(financials, ["Net Income", "Net Income Common Stockholders"])
            equity = get_metric_from_df(balance_sheet, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity", "Total Stockholders Equity"])
            if net_inc is not None and equity is not None and not equity.empty and equity.iloc[0] != 0:
                metrics["roe"] = net_inc.iloc[0] / equity.iloc[0]
            
            if metrics["roe"] is None:
                metrics["roe"] = info.get("returnOnEquity")
        except Exception as e:
            print(f"DEBUG: ROE error: {e}")

        # 4. Debt to Equity
        try:
            equity_val = get_metric_from_df(balance_sheet, ["Stockholders Equity", "Common Stock Equity", "Total Equity", "Total Stockholders Equity"])
            debt = get_metric_from_df(balance_sheet, ["Total Debt", "Net Debt"])
            if debt is not None and equity_val is not None and not equity_val.empty and equity_val.iloc[0] != 0:
                metrics["debt_to_equity"] = debt.iloc[0] / equity_val.iloc[0]
            
            if metrics["debt_to_equity"] is None:
                de = info.get("debtToEquity")
                if de is not None:
                    metrics["debt_to_equity"] = de / 100.0 if de > 5 else de
        except Exception as e:
            print(f"DEBUG: D/E error: {e}")

        # 5. PE Ratio
        try:
            metrics["pe_ratio"] = info.get("trailingPE") or info.get("forwardPE")
            if metrics["pe_ratio"] is None:
                price = info.get("currentPrice") or info.get("previousClose")
                eps = info.get("trailingEps")
                if price and eps and eps != 0:
                    metrics["pe_ratio"] = price / eps
        except Exception as e:
            print(f"DEBUG: PE error: {e}")

        print(f"Final metrics: {metrics}")
        return metrics

    except Exception as e:
        print(f"FAILED CRITICALLY: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_fundamental_logic("ICICIBANK.NS")
