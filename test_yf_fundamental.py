import yfinance as yf
import json

def test_yf(ticker):
    print(f"Testing {ticker}...")
    stock = yf.Ticker(ticker)
    
    try:
        print("Fetching info...")
        info = stock.info
        print(f"Info keys: {list(info.keys())[:10]}...")
        print(f"PE: {info.get('trailingPE')}")
        print(f"ROE: {info.get('returnOnEquity')}")
    except Exception as e:
        print(f"Info failed: {e}")

    try:
        print("\nFetching financials...")
        fin = stock.financials
        print(f"Financials empty: {fin.empty}")
        if not fin.empty:
            print(f"Financials rows: {fin.index.tolist()[:10]}")
    except Exception as e:
        print(f"Financials failed: {e}")

    try:
        print("\nFetching balance sheet...")
        bs = stock.balance_sheet
        print(f"Balance sheet empty: {bs.empty}")
        if not bs.empty:
            print(f"Balance sheet rows: {bs.index.tolist()[:10]}")
    except Exception as e:
        print(f"Balance sheet failed: {e}")

    try:
        print("\nFetching income stmt...")
        is_df = stock.income_stmt
        print(f"Income stmt empty: {is_df.empty}")
    except Exception as e:
        print(f"Income stmt failed: {e}")

if __name__ == "__main__":
    test_yf("ICICIBANK.NS")
