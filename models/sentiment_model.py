from langchain_core.prompts import PromptTemplate
from utils.llm_utils import invoke_with_failover, extract_json
import json
from utils.logger import get_logger

logger = get_logger("sentiment_model")

def analyze_news_sentiment(ticker: str, news_items: list) -> dict:
    """Analyze the sentiment of recent news articles using an LLM with classification counts."""
    if not news_items:
        return {"sentiment_score": 5.0, "positive": 0, "neutral": 0, "negative": 0, "summary": "No news found."}

    news_text = "\n\n".join([f"Headline: {item.get('title', '')}\nSnippet: {item.get('body', '')}" for item in news_items])
    
    template = """
### SENTIMENT ANALYST AGENT PROMPT
Read the following news articles regarding {ticker}.
Classify each headline as either: positive, neutral, or negative.

Articles:
{news_text}

Output MUST be a single JSON object (STRICT JSON ONLY, NO PREAMBLE):
{{
    "positive": <count>,
    "neutral": <count>,
    "negative": <count>,
    "summary": "<2-sentence summary of the overall market sentiment>"
}}
"""
    prompt = PromptTemplate(template=template, input_variables=["ticker", "news_text"])
    
    try:
        response = invoke_with_failover(prompt, {"ticker": ticker, "news_text": news_text}, temperature=0)
        logger.info(f"Sentiment analysis response for {ticker} received.")
        result = extract_json(response.content)
        
        pos = result.get("positive", 0)
        neu = result.get("neutral", 0)
        neg = result.get("negative", 0)
        total = pos + neu + neg
        
        # sentiment_score = (positive_ratio) * 10
        score = (pos / total * 10) if total > 0 else 5.0
        
        return {
            "sentiment_score": round(score, 2),
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "summary": result.get("summary", "Neutral.")
        }
    except Exception as e:
        logger.error(f"Error during sentiment analysis for {ticker}: {e}")
        return {"sentiment_score": 5.0, "positive": 0, "neutral": 0, "negative": 0, "summary": f"Error: {e}"}
