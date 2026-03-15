from duckduckgo_search import DDGS

def get_recent_news(query: str, max_results: int = 15) -> list:
    """Fetch recent news articles based on a search query using DuckDuckGo."""
    try:
        results = []
        with DDGS() as ddgs:
            # Try news search first
            try:
                news_results = list(ddgs.news(query, max_results=max_results))
                if news_results:
                    return news_results
            except Exception:
                pass
            
            # Fallback to general text search if news is empty or fails
            text_results = list(ddgs.text(query, max_results=max_results))
            for r in text_results:
                results.append({
                    "title": r.get("title"),
                    "body": r.get("body"),
                    "url": r.get("href")
                })
        return results
    except Exception as e:
        print(f"Error fetching news for {query}: {e}")
        return []
