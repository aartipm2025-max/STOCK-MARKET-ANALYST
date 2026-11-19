try:
    # The package was renamed from `duckduckgo_search` to `ddgs`
    from ddgs import DDGS
except ImportError:  # pragma: no cover - legacy fallback
    from duckduckgo_search import DDGS


def get_recent_news(query: str, max_results: int = 15) -> list:
    """Fetch recent news articles based on a search query using DuckDuckGo."""
    try:
        ddgs = DDGS()
        # Try news search first
        try:
            news_results = list(ddgs.news(query, max_results=max_results) or [])
            if news_results:
                return news_results
        except Exception:
            pass

        # Fallback to general text search if news is empty or fails
        results = []
        for r in ddgs.text(query, max_results=max_results) or []:
            results.append({
                "title": r.get("title"),
                "body": r.get("body"),
                "url": r.get("href"),
            })
        return results
    except Exception as e:
        print(f"Error fetching news for {query}: {e}")
        return []
