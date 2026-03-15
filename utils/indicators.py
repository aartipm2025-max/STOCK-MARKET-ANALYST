import pandas as pd
import numpy as np

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic technical indicators using pandas."""
    if df.empty or len(df) < 50:
        return df

    try:
        # Moving Averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        if len(df) >= 200:
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        
    return df
