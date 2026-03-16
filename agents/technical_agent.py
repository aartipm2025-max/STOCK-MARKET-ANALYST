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
            return {
                "agent": "technical", "ticker": ticker,
                "latest_data_date": "Unavailable",
                "indicators": {
                    "current_price": 0.0,
                    "rsi_interpretation": "Insufficient data for RSI calculation",
                    "sma_50_interpretation": "Historical data not sufficient for moving average analysis",
                    "sma_200_interpretation": "Historical data not sufficient for long-term trend analysis",
                    "macd_signal": "neutral", "volume_trend": "Unavailable"
                },
                "technical_score": 2.5
            }

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
        
        # Prepare Interpretations
        rsi_val = latest['RSI']
        rsi_desc = "Neutral"
        if rsi_val > 70: rsi_desc = f"Overbought ({round(rsi_val, 2)})"
        elif rsi_val < 30: rsi_desc = f"Oversold ({round(rsi_val, 2)})"
        else: rsi_desc = f"Neutral ({round(rsi_val, 2)})"

        price = latest['Close']
        sma_50 = latest['SMA_50']
        sma_50_desc = f"Insufficient data for 50-day moving average — price trending at {round(price, 2)}"
        if not pd.isna(sma_50):
            status = "above" if price > sma_50 else "below"
            sentiment = "bullish" if price > sma_50 else "bearish"
            sma_50_desc = f"Price is trading {status} the 50-day moving average ({round(sma_50, 2)}), indicating {sentiment} momentum."

        sma_200 = latest['SMA_200']
        sma_200_desc = f"Insufficient data for 200-day moving average — monitoring short-term trend at {round(price, 2)}"
        if not pd.isna(sma_200):
            status = "above" if price > sma_200 else "below"
            sentiment = "long-term bullish" if price > sma_200 else "long-term bearish"
            sma_200_desc = f"Price is trading {status} the 200-day moving average ({round(sma_200, 2)}), showing {sentiment} trend."

        return {
            "agent": "technical",
            "ticker": ticker,
            "latest_data_date": latest_date,
            "indicators": {
                "current_price": round(price, 2),
                "rsi_interpretation": rsi_desc,
                "sma_50_interpretation": sma_50_desc,
                "sma_200_interpretation": sma_200_desc,
                "macd_signal": "bullish" if latest['MACD'] > latest['Signal'] else "bearish",
                "volume_trend": "Increasing" if latest['Volume'] >= latest['Vol_Avg'] else "Decreasing"
            },
            "technical_score": score
        }
    except Exception as e:
        logger.error(f"Technical analysis failed for {ticker}: {e}")
        return {
            "agent": "technical", "ticker": ticker,
            "latest_data_date": "Unavailable",
            "indicators": {
                "current_price": 0.0,
                "rsi_interpretation": "Technical data temporarily unavailable — markets may be closed or data delayed",
                "sma_50_interpretation": "Moving average data temporarily unavailable — unable to confirm current trend direction",
                "sma_200_interpretation": "Long-term moving average data temporarily unavailable",
                "macd_signal": "neutral",
                "volume_trend": "Unavailable"
            },
            "technical_score": 2.5
        }
