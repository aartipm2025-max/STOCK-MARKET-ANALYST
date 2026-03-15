import os
import json
from langchain_core.prompts import PromptTemplate
from utils.logger import get_logger
from utils.llm_utils import invoke_with_failover

logger = get_logger("aggregator_node")

def generate_fallback_report(aggregation_results: list) -> str:
    """Generates a professional text summary if the LLM fails."""
    if not aggregation_results:
        return "No ticker data available to analyze. Please check your input."
    
    report_lines = ["### Stock Market Analysis Report (Deterministic Summary)"]
    for item in aggregation_results:
        ticker = item['ticker']
        score = item['final_score']
        rec = item['recommendation']
        f = item['scores']['fundamental']
        t = item['scores']['technical']
        s = item['scores']['sentiment']
        
        report_lines.append(f"\n**{ticker}**")
        report_lines.append(f"- **Final Score:** {score}/10")
        report_lines.append(f"- **Recommendation:** {rec}")
        report_lines.append(f"- **Breakdown:** Fundamental: {f}, Technical: {t}, Sentiment: {s}")
        report_lines.append(f"- *Note: Detailed narrative unavailable due to system limit; providing score-based summary.*")
        
    return "\n".join(report_lines)

def summarize_results(state: dict) -> dict:
    """Production Aggregator & Report Generator Node."""
    intent = state.get("intent", "")
    tickers = state.get("tickers", [])
    
    # Handle explicit error cases from master_node
    if intent == "error_quota":
        return {
            "final_analysis": "### ⚠️ API Quota Reached\nYour Gemini API free-tier quota has been exhausted for today. Please wait for it to reset or switch to a different API key/model in your .env file.",
            "aggregated_data": []
        }
    if intent == "error_general":
        return {
            "final_analysis": f"### ❌ System Error\nAn unexpected error occurred while parsing your query: {state.get('error', 'Unknown error')}",
            "aggregated_data": []
        }
    
    fundamental = state.get("fundamental_data", {})
    technical = state.get("technical_data", {})
    sentiment = state.get("sentiment_data", {})
    portfolio = state.get("portfolio_data", {})
    
    aggregation_results = []
    
    # Process scores for each ticker
    for ticker in tickers:
        f_score = fundamental.get(ticker, {}).get("fundamental_score", 5.0)
        t_score = technical.get(ticker, {}).get("technical_score", 5.0)
        s_score = sentiment.get(ticker, {}).get("sentiment_score", 5.0)
        
        # Final Score = (0.4 * F) + (0.4 * T) + (0.2 * S)
        final_score = (0.4 * f_score) + (0.4 * t_score) + (0.2 * s_score)
        
        # Recommendation logic
        if final_score >= 8: rec = "STRONG BUY"
        elif final_score >= 6: rec = "BUY"
        elif final_score >= 4: rec = "HOLD"
        else: rec = "AVOID"
        
        aggregation_results.append({
            "ticker": ticker,
            "scores": {"fundamental": f_score, "technical": t_score, "sentiment": s_score},
            "final_score": round(final_score, 2),
            "recommendation": rec,
            "raw_metrics": {
                "f": fundamental.get(ticker),
                "t": technical.get(ticker),
                "s": sentiment.get(ticker)
            }
        })

    # Mode-Specific Enhancements
    if intent == "comparison":
        # Sort by final score descending
        aggregation_results.sort(key=lambda x: x['final_score'], reverse=True)
    
    portfolio_stats = {}
    if intent == "portfolio" and aggregation_results:
        avg_score = sum(x['final_score'] for x in aggregation_results) / len(aggregation_results)
        best = max(aggregation_results, key=lambda x: x['final_score'])
        worst = min(aggregation_results, key=lambda x: x['final_score'])
        portfolio_stats = {
            "overall_score": round(avg_score, 2),
            "best_performing": best['ticker'],
            "worst_performing": worst['ticker'],
            "diversification": "Good" if len(tickers) > 4 else "Low",
            "risk_level": "Moderate" if 4 <= avg_score <= 7 else ("High" if avg_score < 4 else "Low")
        }

    if not tickers and intent not in ["general_query", "error_quota", "error_general"]:
        if intent == "unknown":
            return {
                "final_analysis": "I'm sorry, I specialize in stock market and financial analysis. I can't help with that particular question. Please try asking about a specific stock, a portfolio, or a financial concept.",
                "aggregated_data": []
            }
        return {
            "final_analysis": "No stock tickers could be resolved from your query. If you're asking about a specific company, please mention its name clearly. If you have a general financial question, I'll try my best to answer it.",
            "aggregated_data": []
        }

    # Prepare data for LLM
    data_for_llm = json.dumps(aggregation_results, indent=2)
    portfolio_str = json.dumps(portfolio, indent=2) if portfolio else "N/A"
    
    template = """
### FINAL REPORT GENERATOR PROMPT
You are a senior financial analyst. Generate a structured report based on user intent: {intent}.

#### MANDATORY OUTPUT STRUCTURE (Use this exact format for each ticker):

**REPORT FOR [TICKER NAME]**

**SCORES OVERVIEW**

• **Fundamental Score:** [F_SCORE] / 10
• **Technical Score:** [T_SCORE] / 10
• **Sentiment Score:** [S_SCORE] / 10

---

**FUNDAMENTAL ANALYSIS**

Key Insights:

• **Revenue Growth:** Brief explanation
• **P/E Ratio:** Brief explanation
• **Return on Equity:** Brief explanation
• **Debt-to-Equity:** Brief explanation
• **Operating Margin:** Brief explanation

---

**TECHNICAL ANALYSIS**

Key Indicators:

• **RSI:** Brief explanation
• **MACD:** Brief explanation
• **50-Day Moving Average:** Brief explanation
• **200-Day Moving Average:** Brief explanation
• **Volume Trend:** Brief explanation

---

**SENTIMENT ANALYSIS**

Market Sentiment Insights:

• **News Sentiment:** Brief explanation
• **Industry Outlook:** Brief explanation
• **Investor Confidence:** Brief explanation

---

**AI NARRATIVE SUMMARY**

• **Growth Outlook:** Brief explanation
• **Key Strengths:** Brief explanation
• **Risks:** Brief explanation

---

**FINAL RECOMMENDATION**

**[STRONG BUY / BUY / HOLD / SELL]**

#### RULES:
1. STRICTLY use bullet points (•) for all insights.
2. BOLD all financial terms and metrics (e.g., **Revenue Growth**, **RSI**, **Growth Outlook**).
3. NEVER return long paragraphs. Keep explanations concise and readable.
4. Total length MUST be under 400 words.
5. If data is missing (DATA_NOT_AVAILABLE), still list the bullet point but state "Data not available".

Agent Data:
{data_for_llm}

Portfolio Metrics (if applicable):
{portfolio_stats}
"""
    prompt = PromptTemplate(template=template, input_variables=["data_for_llm", "portfolio_str"])
    
    try:
        logger.info(f"Attempting {intent} report generation with failover support...")
        response = invoke_with_failover(prompt, {
            "data_for_llm": data_for_llm, 
            "portfolio_stats": json.dumps(portfolio_stats),
            "intent": intent
        }, temperature=0.5)
        report = response.content.strip()
        logger.info("LLM report generation successful.")
    except Exception as e:
        logger.error(f"LLM failure occurred: {e}. Switching to deterministic fallback.")
        report = generate_fallback_report(aggregation_results)

    # Ensure we ALWAYS return final_analysis as a string and aggregated_data as a list
    return {
        "final_analysis": str(report), 
        "aggregated_data": aggregation_results
    }
