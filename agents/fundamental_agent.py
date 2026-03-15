import yfinance as yf
from utils.logger import get_logger
import pandas as pd
import numpy as np

logger = get_logger("fundamental_agent")

def analyze_fundamentals(ticker: str) -> dict:
    """Production Fundamental Analyst Agent with multi-source statement extraction."""
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Primary Source: Ticker Info
        info = stock.info or {}
        
        # Pre-fetch financial statements to avoid multiple network calls
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        
        # Metric Extraction Logic
        revenue_growth = info.get("revenueGrowth")
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        roe = info.get("returnOnEquity")
        debt_to_equity = info.get("debtToEquity")
        op_margin = info.get("operatingMargins") or info.get("profitMargins")

        # 2. Fallback to Financial Statements if Info is missing
        if any(v is None for v in [revenue_growth, roe, debt_to_equity, op_margin]):
            try:
                if not income_stmt.empty and not balance_sheet.empty:
                    # Revenue Growth Calculation
                    if revenue_growth is None and "Total Revenue" in income_stmt.index:
                        revs = income_stmt.loc["Total Revenue"]
                        if len(revs) >= 2:
                            revenue_growth = (revs.iloc[0] - revs.iloc[1]) / revs.iloc[1]
                    
                    # Operating Margin Calculation
                    if op_margin is None and "Operating Income" in income_stmt.index and "Total Revenue" in income_stmt.index:
                        op_inc = income_stmt.loc["Operating Income"]
                        total_rev = income_stmt.loc["Total Revenue"]
                        if not op_inc.empty and not total_rev.empty and total_rev.iloc[0] != 0:
                            op_margin = op_inc.iloc[0] / total_rev.iloc[0]

                    # ROE Calculation (Net Income / Shareholder Equity)
                    equity = None
                    for k in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
                        if k in balance_sheet.index:
                            equity = balance_sheet.loc[k].iloc[0]
                            break
                    
                    if roe is None and equity and "Net Income" in income_stmt.index:
                        net_inc = income_stmt.loc["Net Income"].iloc[0]
                        if equity != 0:
                            roe = net_inc / equity

                    # Debt to Equity Calculation
                    if debt_to_equity is None and equity:
                        total_debt = None
                        for k in ["Total Debt", "Net Debt"]:
                            if k in balance_sheet.index:
                                total_debt = balance_sheet.loc[k].iloc[0]
                                break
                        if total_debt is not None and equity != 0:
                            debt_to_equity = total_debt / equity

            except Exception as fe:
                logger.warning(f"Manual financial calculation failed for {ticker}: {fe}")

        # 3. Manual P/E Calculation Fallback
        if pe_ratio is None:
            try:
                price = info.get("currentPrice") or info.get("previousClose")
                if not price:
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        price = hist['Close'].iloc[0]
                
                eps = info.get("trailingEps")
                if eps is None and "Net Income" in income_stmt.index and "Ordinary Shares Number" in balance_sheet.index:
                    ni = income_stmt.loc["Net Income"].iloc[0]
                    shares = balance_sheet.loc["Ordinary Shares Number"].iloc[0]
                    if shares != 0:
                        eps = ni / shares
                
                if price and eps and eps > 0:
                    pe_ratio = price / eps
            except:
                pass

        # Normalize Debt to Equity percentage
        norm_de = debt_to_equity
        if norm_de is not None and norm_de > 10:
            norm_de = norm_de / 100.0

        # 4. Scoring Logic (Max 10.0)
        # Weights: Growth (2.5), Profitability (2.5), Stability (2.5), Valuation (2.5)
        score = 0.0
        
        # Growth Strength (max 2.5)
        if revenue_growth is not None:
            if revenue_growth > 0.15: score += 2.5
            elif revenue_growth > 0.05: score += 1.5
            elif revenue_growth > 0: score += 0.5
            
        # Profitability: ROE + Op Margin (max 2.5)
        p_score = 0.0
        if roe is not None:
            if roe > 0.15: p_score += 1.25
            elif roe > 0.08: p_score += 0.75
        if op_margin is not None:
            if op_margin > 0.20: p_score += 1.25
            elif op_margin > 0.10: p_score += 0.75
        score += p_score
            
        # Financial Stability: Debt-to-Equity (max 2.5)
        if norm_de is not None:
            if norm_de < 0.5: score += 2.5
            elif norm_de < 1.0: score += 1.5
            elif norm_de < 2.0: score += 0.5
            
        # Valuation: P/E Ratio (max 2.5)
        if pe_ratio is not None:
            if 0 < pe_ratio < 20: score += 2.5
            elif 20 <= pe_ratio < 40: score += 1.5
            elif 40 <= pe_ratio < 60: score += 0.5

        # 5. Result Formatting
        return {
            "agent": "fundamental",
            "ticker": ticker,
            "metrics": {
                "long_name": info.get("longName") or info.get("shortName") or ticker,
                "revenue_growth": round(revenue_growth, 4) if revenue_growth is not None else "Unavailable from source",
                "pe_ratio": round(pe_ratio, 2) if pe_ratio is not None else "Unavailable from source",
                "roe": round(roe, 4) if roe is not None else "Unavailable from source",
                "debt_to_equity": round(norm_de, 4) if norm_de is not None else "Unavailable from source",
                "operating_margin": round(op_margin, 4) if op_margin is not None else "Unavailable from source"
            },
            "fundamental_score": round(min(score, 10.0), 1)
        }
    except Exception as e:
        logger.error(f"Fundamental analysis failed for {ticker}: {e}")
        return {"error": f"DATA_NOT_AVAILABLE: {str(e)}"}
