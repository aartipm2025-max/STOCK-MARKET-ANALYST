import yfinance as yf
import pandas as pd
import numpy as np

def analyze_portfolio(tickers: list) -> dict:
    """Production Portfolio Analyst Agent."""
    try:
        portfolio_data = {}
        total_value = 0
        returns = []
        
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if hist.empty: continue
            
            # YTD Return
            ytd_start = f"{pd.Timestamp.now().year}-01-01"
            ytd_data = hist.loc[ytd_start:]
            if not ytd_data.empty:
                ytd_ret = (ytd_data['Close'].iloc[-1] / ytd_data['Close'].iloc[0]) - 1
            else:
                ytd_ret = 0.0
                
            # Log daily returns for volatility
            daily_rets = hist['Close'].pct_change().dropna()
            returns.append(daily_rets)
            
            portfolio_data[ticker] = {
                "ytd_return": round(ytd_ret, 4),
                "sector": stock.info.get("sector", "Unknown")
            }

        # Portfolio level metrics
        combined_returns = pd.concat(returns, axis=1).mean(axis=1) if returns else pd.Series()
        ytd_port_return = combined_returns.loc[ytd_start:].sum() if not combined_returns.empty else 0
        volatility = combined_returns.std() * np.sqrt(252) if not combined_returns.empty else 0
        
        sectors = set([d["sector"] for d in portfolio_data.values()])
        div = "high" if len(sectors) > 4 else "moderate" if len(sectors) > 2 else "low"
        
        risk = "low" if volatility < 0.15 else "medium" if volatility < 0.3 else "high"

        return {
            "agent": "portfolio",
            "portfolio": tickers,
            "metrics": {
                "return_ytd": round(ytd_port_return, 4),
                "volatility": round(volatility, 4),
                "sector_diversification": div
            },
            "risk_score": risk
        }
    except Exception as e:
        return {"error": f"DATA_NOT_AVAILABLE: {str(e)}"}
