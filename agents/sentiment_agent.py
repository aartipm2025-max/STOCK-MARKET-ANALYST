from tools.duckduckgo_tool import get_recent_news
from models.sentiment_model import analyze_news_sentiment

def analyze_sentiment(ticker: str) -> dict:
    """Production Sentiment Analyst Agent."""
    # Try specific ticker search first
    query = f"{ticker} stock news last 7 days"
    news_items = get_recent_news(query, max_results=8)
    
    # If no results and it's an Indian ticker, try without the .NS suffix
    if not news_items and ticker.endswith(".NS"):
        clean_ticker = ticker.replace(".NS", "")
        fallback_query = f"{clean_ticker} stock news India"
        print(f"No news for {ticker}, trying fallback: {fallback_query}")
        news_items = get_recent_news(fallback_query, max_results=8)
    
    if not news_items:
        return {
            "agent": "sentiment",
            "ticker": ticker,
            "articles": 0,
            "sentiment_breakdown": {"positive": 0, "neutral": 1, "negative": 0},
            "sentiment_score": 5.0,
            "summary": f"Market sentiment for {ticker.replace('.NS','')} appears neutral. No significant news signals detected in the past 7 days. Absence of negative headlines is a cautiously positive indicator."
        }
        
    analysis = analyze_news_sentiment(ticker, news_items)
    
    return {
        "agent": "sentiment",
        "ticker": ticker,
        "articles": len(news_items),
        "sentiment_breakdown": {
            "positive": analysis.get("positive", 0),
            "neutral": analysis.get("neutral", 0),
            "negative": analysis.get("negative", 0)
        },
        "sentiment_score": analysis.get("sentiment_score", 5.0),
        "summary": analysis.get("summary", "")
    }
