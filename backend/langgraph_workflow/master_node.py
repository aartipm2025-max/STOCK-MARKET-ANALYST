import os
import re
import json
from langchain_core.prompts import PromptTemplate
from utils.logger import get_logger
from utils.llm_utils import invoke_with_failover, extract_json

logger = get_logger("master_node")

# Rule-based fallback used when every LLM provider is unavailable.
COMMON_TICKERS = {
    "reliance": "RELIANCE.NS", "ril": "RELIANCE.NS",
    "tcs": "TCS.NS", "tata consultancy": "TCS.NS",
    "infosys": "INFY.NS", "infy": "INFY.NS",
    "wipro": "WIPRO.NS", "hcl": "HCLTECH.NS", "hcltech": "HCLTECH.NS",
    "tech mahindra": "TECHM.NS", "techm": "TECHM.NS",
    "hdfc bank": "HDFCBANK.NS", "hdfcbank": "HDFCBANK.NS", "hdfc": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS", "icici": "ICICIBANK.NS",
    "sbi": "SBIN.NS", "state bank": "SBIN.NS",
    "axis bank": "AXISBANK.NS", "axis": "AXISBANK.NS",
    "kotak": "KOTAKBANK.NS", "kotak mahindra": "KOTAKBANK.NS",
    "bajaj finance": "BAJFINANCE.NS", "bajaj finserv": "BAJAJFINSV.NS",
    "itc": "ITC.NS", "hindustan unilever": "HINDUNILVR.NS", "hul": "HINDUNILVR.NS",
    "larsen": "LT.NS", "l&t": "LT.NS", "maruti": "MARUTI.NS",
    "tata motors": "TATAMOTORS.NS", "tata steel": "TATASTEEL.NS",
    "mahindra": "M&M.NS", "m&m": "M&M.NS", "sun pharma": "SUNPHARMA.NS",
    "bharti airtel": "BHARTIARTL.NS", "airtel": "BHARTIARTL.NS",
    "asian paints": "ASIANPAINT.NS", "titan": "TITAN.NS", "adani enterprises": "ADANIENT.NS",
    "adani ports": "ADANIPORTS.NS", "ongc": "ONGC.NS", "ntpc": "NTPC.NS",
    "power grid": "POWERGRID.NS", "coal india": "COALINDIA.NS", "nestle": "NESTLEIND.NS",
    "ultratech": "ULTRACEMCO.NS", "zomato": "ETERNAL.NS", "eternal": "ETERNAL.NS",
    "paytm": "PAYTM.NS", "nykaa": "NYKAA.NS", "dmart": "DMART.NS",
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "tesla": "TSLA", "meta": "META",
}

_STOPWORDS = {
    "VS", "AND", "OR", "THE", "HOW", "IS", "DOING", "ANALYZE", "ANALYSE", "MY", "PORTFOLIO",
    "COMPARE", "WITH", "STOCK", "STOCKS", "SHARE", "SHARES", "OF", "FOR", "A", "AN", "TO",
    "WHAT", "SHOULD", "I", "BUY", "SELL", "HOLD", "VERSUS", "ABOUT", "TELL", "ME", "IN",
    "BETTER", "BEST", "AMONG", "HOLDINGS", "REVIEW", "LTD", "LIMITED", "INDIA", "NSE", "BSE",
    "PE", "P/E", "EPS", "ROE", "ROCE", "IPO", "ETF", "NAV", "RSI", "MACD", "SMA", "EMA", "SIP",
    "GDP", "RBI", "SEBI", "FII", "DII", "AI", "NIFTY", "SENSEX", "USA", "US", "EV", "CAGR",
}


def rule_based_parse(query: str) -> dict:
    """Resolve tickers without an LLM (fallback when all providers fail)."""
    q = query.lower()
    hits = []  # (position, ticker)
    masked = query
    # Match longest names first so "hdfc bank" wins over "hdfc"
    for name in sorted(COMMON_TICKERS, key=len, reverse=True):
        pattern = r"(?<![a-z])" + re.escape(name) + r"(?![a-z])"
        for m in re.finditer(pattern, q):
            hits.append((m.start(), COMMON_TICKERS[name]))
            # mask so shorter names / raw-symbol pass don't re-match it
            q = q[:m.start()] + " " * len(name) + q[m.end():]
            masked = masked[:m.start()] + " " * len(name) + masked[m.end():]

    # Tokens the user typed in CAPITALS (e.g. "IRCTC", "INFY.NS") are treated as raw symbols
    for m in re.finditer(r"[^\s,;/+]+", masked):
        tok = m.group().strip(".?!:()[]\"'")
        if not tok or tok != tok.upper() or tok in _STOPWORDS:
            continue
        if not re.fullmatch(r"[A-Z][A-Z0-9&\-]{1,14}(\.[A-Z]{1,3})?", tok):
            continue
        hits.append((m.start(), tok if "." in tok else f"{tok}.NS"))

    found = []
    for _, t in sorted(hits):
        if t not in found:
            found.append(t)

    if not found:
        intent = "general_query"
    elif len(found) == 1:
        intent = "single_stock"
    elif "portfolio" in query.lower() or "holding" in query.lower():
        intent = "portfolio"
    else:
        intent = "comparison"
    return {"intent": intent, "tickers": found}

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
- RECENT REBRANDINGS (CRITICAL):
    * Zomato -> Use "ETERNAL.NS" (Corporate name changed to Eternal Ltd).
    * PayTM -> Use "PAYTM.NS" (Check current status if data fails).
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
        for t in result.get("tickers", []) or []:
            if not isinstance(t, str) or not t.strip():
                continue
            t = t.strip().upper()
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
        logger.error(f"Error parsing query via LLM: {error_msg}. Falling back to rule-based resolver.")
        fallback = rule_based_parse(query)
        if fallback["tickers"]:
            logger.info(f"Rule-based fallback resolved: {fallback}")
            return fallback
        # Capture quota errors specifically to inform the user
        if "429" in error_msg or "quota" in error_msg.lower():
            intent = "error_quota"
        else:
            intent = "error_general"
        return {"intent": intent, "tickers": [], "error": error_msg}

def master_node_func(state: dict) -> dict:
    """Master node logic to process initial query and routing."""
    query = state.get("query", "")
    mode = state.get("mode", "Single Stock")
    
    parsed = parse_query_and_intent(query)
    
    intent = parsed["intent"]
    tickers = parsed["tickers"]
    
    # CRITICAL: Override intent based on explicit UI mode selection
    # Only override if we didn't hit a hard API error, AND if there are actually tickers to process.
    if not parsed.get("error") and intent not in ["error_quota", "error_general"] and len(tickers) > 0:
        if mode == "Single Stock":
            # Several stocks typed in single mode ("TCS vs Infosys") -> compare them
            intent = "single_stock" if len(tickers) == 1 else ("portfolio" if intent == "portfolio" else "comparison")
        elif mode == "Compare Stocks":
            intent = "comparison"
        elif mode == "Portfolio":
            intent = "portfolio"
    
    logger.info(f"Final resolved intent after UI mode override: {intent}")
        
    return {
        "intent": intent,
        "tickers": tickers,
        "error": parsed.get("error", "")
    }

