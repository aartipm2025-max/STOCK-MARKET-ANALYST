import yfinance as yf
import pandas as pd

def get_stock_info(ticker: str) -> dict:
    """Fetch basic info, current price, and financial metrics."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "eps": info.get("trailingEps"),
            "roe": info.get("returnOnEquity"),
            "profit_margin": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "revenue_growth": info.get("revenueGrowth"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow")
        }
    except Exception as e:
        return {"error": str(e)}

def get_historical_prices(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical price data."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        return hist
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def get_financial_statements(ticker: str) -> dict:
    """Fetch income statement, balance sheet, and cashflow."""
    try:
        stock = yf.Ticker(ticker)
        return {
            "income_stmt": stock.financials.to_dict() if not stock.financials.empty else {},
            "balance_sheet": stock.balance_sheet.to_dict() if not stock.balance_sheet.empty else {},
            "cash_flow": stock.cashflow.to_dict() if not stock.cashflow.empty else {}
        }
    except Exception as e:
        return {"error": str(e)}
