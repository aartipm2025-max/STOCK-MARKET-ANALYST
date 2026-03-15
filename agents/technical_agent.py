import yfinance as yf
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger("technical_agent")

def analyze_technicals(ticker: str) -> dict:
    """Production Technical Analyst Agent."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        if df.empty or len(df) < 20: 
            return {"error": "DATA_NOT_AVAILABLE", "details": "Insufficient historical data"}

        # Compute Indicators
        df['SMA_50'] = df['Close'].rolling(window=min(len(df), 50)).mean()
        df['SMA_200'] = df['Close'].rolling(window=min(len(df), 200)).mean()
        
        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=min(len(df), 14)).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=min(len(df), 14)).mean()
        rs = gain / (loss + 1e-9) # Avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volume Trend
        df['Vol_Avg'] = df['Volume'].rolling(window=min(len(df), 20)).mean()
        
        latest = df.iloc[-1]
        latest_date = df.index[-1].strftime("%d %B %Y")
        
        # Scoring
        score = 0.0
        if latest['SMA_50'] >= latest['SMA_200']: score += 2.5
        if latest['Close'] >= latest['SMA_50']: score += 2.5
        if 30 <= latest['RSI'] <= 70: score += 2.5 # Relaxed RSI range
        elif latest['RSI'] < 30: score += 2.5 # Oversold is good for buy
        if latest['MACD'] > latest['Signal']: score += 2.5
        
        return {
            "agent": "technical",
            "ticker": ticker,
            "latest_data_date": latest_date,
            "indicators": {
                "current_price": round(latest['Close'], 2),
                "sma_50": round(latest['SMA_50'], 2) if not pd.isna(latest['SMA_50']) else None,
                "sma_200": round(latest['SMA_200'], 2) if not pd.isna(latest['SMA_200']) else None,
                "rsi": round(latest['RSI'], 2) if not pd.isna(latest['RSI']) else None,
                "macd_signal": "bullish" if latest['MACD'] > latest['Signal'] else "bearish",
                "volume_trend": "increasing" if latest['Volume'] >= latest['Vol_Avg'] else "decreasing"
            },
            "technical_score": score
        }
    except Exception as e:
        logger.error(f"Technical analysis failed for {ticker}: {e}")
        return {"error": f"DATA_NOT_AVAILABLE: {str(e)}"}
