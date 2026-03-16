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
    market_context = state.get("market_context_data", {})
    portfolio = state.get("portfolio_data", {})
    
    aggregation_results = []
    
    # Process scores for each ticker
    for ticker in tickers:
        f_score = fundamental.get(ticker, {}).get("fundamental_score", 0.0)
        t_score = technical.get(ticker, {}).get("technical_score", 0.0)
        s_score = sentiment.get(ticker, {}).get("sentiment_score", 0.0)
        m_score = market_context.get(ticker, {}).get("context_score", 0.0)
        
        # Final Score = (0.4 * F) + (0.4 * T) + (0.2 * S)
        final_score = (0.4 * f_score) + (0.4 * t_score) + (0.2 * s_score)
        
        # Confidence Score = (0.4 * F) + (0.3 * T) + (0.3 * S) -> Normalized to 100%
        conf_score = (0.35 * f_score) + (0.3 * t_score) + (0.2 * s_score) + (0.15 * m_score) # Adjusted to include Market Context
        confidence_pct = round(conf_score * 10, 1)

        # Recommendation logic
        if final_score >= 8.0: rec = "STRONG BUY"
        elif final_score >= 6.5: rec = "BUY"
        elif final_score >= 4.5: rec = "HOLD"
        else: rec = "AVOID"

        # Extract latest date from technical data
        analysis_date = technical.get(ticker, {}).get("latest_data_date", "Unknown")
        
        aggregation_results.append({
            "ticker": ticker,
            "scores": {
                "fundamental": f_score, 
                "technical": t_score, 
                "sentiment": s_score,
                "market_context": m_score
            },
            "final_score": round(final_score, 2),
            "confidence_level": f"{confidence_pct}%",
            "recommendation": rec,
            "analysis_date": analysis_date,
            "raw_metrics": {
                "f": fundamental.get(ticker),
                "t": technical.get(ticker),
                "s": sentiment.get(ticker),
                "m": market_context.get(ticker)
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
    
    template = """
### INSTITUTIONAL FINANCIAL RESEARCH REPORT GENERATOR
You are a senior equity research analyst at a top-tier hedge fund. Generate a professional, highly reliable, and non-duplicated report.

#### MANDATORY REPORT STRUCTURE (Strict Order):

**REPORT FOR [TICKER NAME]**
Analysis Date: [LATEST DATE FROM DATA]

---

**FUNDAMENTAL ANALYSIS**
Analyze these core metrics with financial reasoning:
• **Revenue Growth:** [State value and explain performance]
• **P/E Ratio:** [State value and explain valuation context]
• **Return on Equity:** [State value and explain capital efficiency]
• **Debt-to-Equity:** [State value and explain risk profile]
• **Operating Margin:** [State value and explain operational efficiency]

---

**TECHNICAL ANALYSIS**
Translate these indicators into market trend interpretations:
• **RSI:** [Status and market interpretation]
• **MACD:** [Current trend momentum]
• **Moving Averages:** [Relationship between price and SMAs]

---

**SENTIMENT ANALYSIS**
Provide institutional-grade market mood analysis:
• **Market Sentiment:** [Calculated tone from news]
• **Risk/Confidence Drivers:** [What is currently moving price action]

---

**MARKET CONTEXT**
Analyze the broader environment using 'm' data:
• **Nifty 50 Trend:** [Explain index performance impact]
• **Sector Performance:** [Relative strength of the industry]
• **Peer Comparison:** [How stock compares to industry peers]

---

**AI NARRATIVE SUMMARY**
Synthesize all agent inputs into a coherent investment thesis:
• **Growth Outlook:** [Synthesized expectation]
• **Key Strengths:** [Primary competitive advantages]
• **Key Risks:** [Primary headwinds]

---

**RISK FACTORS**
Identify and explain primary investment risks:
• **Technical Risk:** [Based on momentum/volatility]
• **Fundamental Risk:** [Based on financial health/metrics]
• **Market Risk:** [Based on macro/index trends]
• **Industry Risk:** [Sector specific headwinds]

---

**FINAL RECOMMENDATION**
**[STRONG BUY / BUY / HOLD / AVOID]**
Confidence Level: {confidence_placeholder}

---

**INVESTMENT HORIZON**
• **Short-Term Outlook:** [Explanation based on technical signals and momentum]
• **Long-Term Outlook:** [Explanation based on fundamentals and macro sentiment]

---

#### CRITICAL RULES:
1. PROFESSIONAL TONE: Use formal financial terminology.
2. NO RAW DATA DUMPS: Interpret the numbers provided in the 'Agent Data'.
3. FORMATTING: Use bolding for metrics and headers. Use exactly two newlines between bullets.
4. RECOMMENDATION: Bold the final call. Include the "Confidence Level" provided.
5. NO DUPLICATION: Each section must be unique and non-repetitive.

Agent Data:
{data_for_llm}
"""
    prompt = PromptTemplate(template=template, input_variables=["data_for_llm", "confidence_placeholder"])
    
    conf_val = aggregation_results[0].get("confidence_level", "N/A") if aggregation_results else "N/A"

    try:
        logger.info(f"Attempting {intent} institutional report generation...")
        response = invoke_with_failover(prompt, {
            "data_for_llm": data_for_llm,
            "confidence_placeholder": conf_val
        }, temperature=0.3)
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
