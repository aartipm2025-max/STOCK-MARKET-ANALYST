import os
import json
from langchain_core.prompts import PromptTemplate
from utils.logger import get_logger
from utils.llm_utils import invoke_with_failover, extract_json

logger = get_logger("master_node")

def parse_query_and_intent(query: str) -> dict:
    """Production Ticker Resolver & Intent Classifier."""
    
    template = """
### MASTER ORCHESTRATOR PROMPT
You are the entry point of a financial AI system. Your task is to classify user intent and extract stock tickers.

#### INTENT RULES:
1. "single_stock": User asks about ONE specific company (e.g., "Review Reliance").
2. "comparison": User asks to compare TWO OR MORE companies (e.g., "TCS vs Infosys") or asks for "best stock among X, Y, Z".
3. "portfolio": User provides a list of stocks they OWN or asks to analyze their "portfolio" or "holdings".
4. "general_query": Query is a general financial question, definition, or chat that DOES NOT require specific stock data (e.g., "What is a PE ratio?" or "How is the market today?").
5. "unknown": Query is completely irrelevant to finance or markets.

#### TICKER RESOLUTION RULES:
- Resolve company names into valid Yahoo Finance tickers.
- DEFAULT: Use ".NS" suffix for Indian stocks (NSE).
- GLOBAL: Only use US/Global tickers if the user explicitly mentions a global market or company (e.g., "Apple", "Nvidia").
- EXTRACT ALL: For comparison/portfolio, extract every single stock mentioned.
- If it is a "general_query", leave the "tickers" list empty.

User Query: {query}

Output Format (STRICT JSON ONLY, NO PREAMBLE):
{{
    "intent": "single_stock|comparison|portfolio|general_query|unknown",
    "tickers": ["RELIANCE.NS", ...],
    "mode_hint": "optional specific instructions if detected"
}}
"""
    prompt = PromptTemplate(template=template, input_variables=["query"])
    
    try:
        logger.info(f"Invoking LLM for query: {query[:50]}...")
        response = invoke_with_failover(prompt, {"query": query}, temperature=0)
        logger.info(f"LLM Response received: {response.content[:50]}...")
        result = extract_json(response.content)
        
        # Ensure NSE suffix if not present for strings that look like Indian tickers
        resolved_tickers = []
        for t in result.get("tickers", []):
            t = t.upper()
            if not t.endswith(".NS") and "." not in t:
                # Assuming Indian stocks as default per system role
                t = f"{t}.NS"
            resolved_tickers.append(t)
            
        logger.info(f"Resolved intent: {result.get('intent', 'unknown')}, tickers: {resolved_tickers}")
        return {
            "intent": result.get("intent", "unknown"),
            "tickers": resolved_tickers
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error parsing query: {error_msg}")
        # Capture quota errors specifically to inform the user
        if "429" in error_msg or "quota" in error_msg.lower():
            intent = "error_quota"
        else:
            intent = "error_general"
        return {"intent": intent, "tickers": [], "error": error_msg}

def master_node_func(state: dict) -> dict:
    """Master node logic to process initial query and routing."""
    query = state.get("query", "")
    parsed = parse_query_and_intent(query)
    
    intent = parsed["intent"]
    tickers = parsed["tickers"]
    
    # Calculate how many agents we expect to join at the end
    if intent == "portfolio":
        total_agents = 3 # portfolio, sentiment, technical
    elif intent in ["single_stock", "comparison"] and tickers:
        total_agents = 3 # fundamental, technical, sentiment
    else:
        total_agents = 0 # unknown or no tickers goes straight to aggregator
        
    return {
        "intent": intent,
        "tickers": tickers,
        "error": parsed.get("error", "")
    }

