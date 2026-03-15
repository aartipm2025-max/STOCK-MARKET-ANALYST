import yfinance as yf
from utils.logger import get_logger
import pandas as pd
import numpy as np

logger = get_logger("fundamental_agent")

def analyze_fundamentals(ticker: str) -> dict:
    """Production Fundamental Analyst Agent with multi-source fallback."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        
        # 1. Primary Source: Ticker Info
        revenue_growth = info.get("revenueGrowth") or info.get("earningsQuarterlyGrowth")
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        roe = info.get("returnOnEquity")
        debt_to_equity = info.get("debtToEquity")
        op_margin = info.get("operatingMargins") or info.get("profitMargins")
        eps_growth = info.get("earningsGrowth")

        # 2. Secondary Source: Financial Statements (Fallback)
        if any(v is None for v in [revenue_growth, roe, debt_to_equity, op_margin]):
            try:
                income = stock.income_stmt
                balance = stock.balance_sheet
                
                if not income.empty and not balance.empty:
                    # Extraction from Income Statement
                    if revenue_growth is None and "Total Revenue" in income.index:
                        revs = income.loc["Total Revenue"]
                        if len(revs) >= 2:
                            revenue_growth = (revs.iloc[0] - revs.iloc[1]) / revs.iloc[1]
                    
                    if op_margin is None and "Operating Income" in income.index and "Total Revenue" in income.index:
                        op_inc = income.loc["Operating Income"]
                        total_rev = income.loc["Total Revenue"]
                        if not op_inc.empty and not total_rev.empty:
                            op_margin = op_inc.iloc[0] / total_rev.iloc[0]

                    # Extraction from Balance Sheet
                    equity = None
                    if "Stockholders Equity" in balance.index:
                        equity = balance.loc["Stockholders Equity"].iloc[0]
                    elif "Total Equity Gross Minority Interest" in balance.index:
                        equity = balance.loc["Total Equity Gross Minority Interest"].iloc[0]

                    if roe is None and equity and "Net Income" in income.index:
                        net_inc = income.loc["Net Income"].iloc[0]
                        roe = net_inc / equity if equity != 0 else None

                    if debt_to_equity is None and equity:
                        total_debt = None
                        if "Total Debt" in balance.index:
                            total_debt = balance.loc["Total Debt"].iloc[0]
                        
                        if total_debt is not None:
                            debt_to_equity = total_debt / equity if equity != 0 else None
            except Exception as fe:
                logger.warning(f"Financial statement extraction failed for {ticker}: {fe}")

        # Normalize Debt to Equity (Convert % to decimal if needed)
        norm_debt_equity = debt_to_equity
        if norm_debt_equity is not None and norm_debt_equity > 10: # Likely a percentage
            norm_debt_equity = norm_debt_equity / 100.0

        # Scoring Logic (Max 10.0)
        score = 0.0
        metrics_found = 0
        
        if revenue_growth is not None:
            metrics_found += 1
            if revenue_growth > 0.10: score += 2
            elif revenue_growth > 0: score += 1
            
        if roe is not None:
            metrics_found += 1
            if roe > 0.15: score += 2
            elif roe > 0.08: score += 1
            
        if norm_debt_equity is not None:
            metrics_found += 1
            if norm_debt_equity < 0.5: score += 2
            elif norm_debt_equity < 1.0: score += 1
            
        if pe_ratio is not None:
            metrics_found += 1
            if 5 < pe_ratio < 25: score += 2
            elif 0 < pe_ratio < 40: score += 1
            
        if op_margin is not None:
            metrics_found += 1
            if op_margin > 0.20: score += 2
            elif op_margin > 0.10: score += 1

        # Adjust score if data is missing but some metrics are positive
        if metrics_found > 0 and score == 0:
            score = 1.0 # Minimal positive presence
        elif metrics_found == 0:
            score = 0.0

        return {
            "agent": "fundamental",
            "ticker": ticker,
            "metrics": {
                "long_name": info.get("longName") or info.get("shortName") or ticker,
                "revenue_growth": round(revenue_growth, 4) if revenue_growth is not None else None,
                "pe_ratio": round(pe_ratio, 2) if pe_ratio is not None else None,
                "roe": round(roe, 4) if roe is not None else None,
                "debt_to_equity": round(norm_debt_equity, 4) if norm_debt_equity is not None else None,
                "operating_margin": round(op_margin, 4) if op_margin is not None else None
            },
            "fundamental_score": round(score, 1)
        }
    except Exception as e:
        logger.error(f"Fundamental analysis failed for {ticker}: {e}")
        return {"error": f"DATA_NOT_AVAILABLE: {str(e)}"}
