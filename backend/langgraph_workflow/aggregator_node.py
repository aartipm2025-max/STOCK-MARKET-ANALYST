import os
import json
from langchain_core.prompts import PromptTemplate
from utils.logger import get_logger
from utils.llm_utils import invoke_with_failover

logger = get_logger("aggregator_node")


def build_deterministic_sections(ticker_data: dict, agg: dict) -> dict:
    """Build guaranteed section content from raw agent data."""
    sections = {}
    
    # Fundamental
    f_data = ticker_data.get("f") or {}
    f_metrics = f_data.get("metrics", {})
    sections["fundamental"] = (
        f"• **Revenue Growth:** {f_metrics.get('revenue_growth', 'Unavailable')}\n\n"
        f"• **P/E Ratio:** {f_metrics.get('pe_ratio', 'Unavailable')}\n\n"
        f"• **Return on Equity:** {f_metrics.get('roe', 'Unavailable')}\n\n"
        f"• **Debt-to-Equity:** {f_metrics.get('debt_to_equity', 'Unavailable')}\n\n"
        f"• **Operating Margin:** {f_metrics.get('operating_margin', 'Unavailable')}"
    )
    
    # Technical
    t_data = ticker_data.get("t") or {}
    t_ind = t_data.get("indicators", {})
    sections["technical"] = (
        f"• **RSI:** {t_ind.get('rsi_interpretation', 'Unavailable')}\n\n"
        f"• **MACD Signal:** {t_ind.get('macd_signal', 'Unavailable')}\n\n"
        f"• **50-Day SMA:** {t_ind.get('sma_50_interpretation', 'Unavailable')}\n\n"
        f"• **200-Day SMA:** {t_ind.get('sma_200_interpretation', 'Unavailable')}\n\n"
        f"• **Volume Trend:** {t_ind.get('volume_trend', 'Unavailable')}"
    )
    
    # Sentiment
    s_data = ticker_data.get("s") or {}
    s_breakdown = s_data.get("sentiment_breakdown", {})
    s_summary = s_data.get("summary", "No sentiment summary available.")
    sections["sentiment"] = (
        f"• **Articles Analyzed:** {s_data.get('articles', 0)}\n\n"
        f"• **Positive:** {s_breakdown.get('positive', 0)} | **Neutral:** {s_breakdown.get('neutral', 0)} | **Negative:** {s_breakdown.get('negative', 0)}\n\n"
        f"• **Summary:** {s_summary}"
    )
    
    # Market Context
    m_data = ticker_data.get("m") or {}
    sections["market_context"] = (
        f"• **Nifty 50 Trend:** {m_data.get('nifty_trend', 'Unavailable')}\n\n"
        f"• **Sector Performance:** {m_data.get('sector_performance', 'Unavailable')}\n\n"
        f"• **Peer Comparison:** {m_data.get('peer_comparison', 'Unavailable')}"
    )
    
    # Recommendation
    rec = agg.get("recommendation", "HOLD")
    conf = agg.get("confidence_level", "N/A")
    sections["recommendation"] = f"**{rec}**\nConfidence Level: {conf}"
    
    # Narrative, Risks, Horizon — placeholders from deterministic data
    sections["narrative"] = (
        f"• **Overall Score:** {agg.get('final_score', 'N/A')}/10\n\n"
        f"• **Key Driver:** Fundamental Score {agg.get('scores', {}).get('fundamental', 0)}/10"
    )
    sections["risks"] = (
        "• **Technical Risk:** Evaluate RSI and MACD divergence\n\n"
        "• **Fundamental Risk:** Review debt levels and margin trends\n\n"
        "• **Market Risk:** Monitor Nifty 50 correlation"
    )
    sections["horizon"] = (
        "• **Short-Term:** Based on technical momentum indicators\n\n"
        "• **Long-Term:** Based on fundamental strength and growth trajectory"
    )
    
    return sections


def parse_llm_report(report_text: str) -> dict:
    """Parse LLM report into structured sections, stripping ALL header variants."""
    import re
    sections = {}
    segments = report_text.split("---")
    
    section_map = {
        "FUNDAMENTAL ANALYSIS": "fundamental",
        "TECHNICAL ANALYSIS": "technical",
        "SENTIMENT ANALYSIS": "sentiment",
        "MARKET CONTEXT": "market_context",
        "AI NARRATIVE SUMMARY": "narrative",
        "RISK FACTORS": "risks",
        "FINAL RECOMMENDATION": "recommendation",
        "INVESTMENT HORIZON": "horizon"
    }
    
    # Headers to strip from content (all possible LLM formats)
    all_headers = [
        "FUNDAMENTAL ANALYSIS", "TECHNICAL ANALYSIS", "SENTIMENT ANALYSIS",
        "MARKET CONTEXT", "AI NARRATIVE SUMMARY", "RISK FACTORS",
        "FINAL RECOMMENDATION", "INVESTMENT HORIZON",
        "REPORT FOR", "INSTITUTIONAL FINANCIAL", "CRITICAL RULES",
        "Agent Data", "RULES:"
    ]
    
    for segment in segments:
        seg_upper = segment.upper().strip()
        for header, key in section_map.items():
            if header in seg_upper:
                cleaned = segment
                # Strip all header formats: **HEADER**, ### HEADER, HEADER, etc.
                for h in all_headers:
                    # Remove **Header** (bold)
                    cleaned = re.sub(r'\*\*\s*' + re.escape(h) + r'\s*\*\*', '', cleaned, flags=re.IGNORECASE)
                    # Remove ### Header (markdown heading)
                    cleaned = re.sub(r'#{1,4}\s*' + re.escape(h), '', cleaned, flags=re.IGNORECASE)
                    # Remove plain Header at start of line
                    cleaned = re.sub(r'^\s*' + re.escape(h) + r'\s*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
                
                # Remove any leftover ### or ** lines that are now empty
                cleaned = re.sub(r'^\s*[#*]+\s*$', '', cleaned, flags=re.MULTILINE)
                # Force newlines before every bullet point •
                cleaned = re.sub(r'(?<!\n)\s*•\s*', '\n\n• ', cleaned)
                # Collapse excessive newlines
                cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
                
                content = cleaned.strip()
                if content:
                    sections[key] = content
                break
    
    return sections


def summarize_results(state: dict) -> dict:
    """Production Aggregator & Report Generator Node — returns structured sections."""
    intent = state.get("intent", "")
    tickers = state.get("tickers", [])
    
    # Handle explicit error cases from master_node
    if intent == "error_quota":
        return {
            "final_analysis": json.dumps({
                "recommendation": "⚠️ API Quota Reached",
                "fundamental": "Your Gemini API free-tier quota has been exhausted.",
                "technical": "", "sentiment": "", "market_context": "",
                "narrative": "", "risks": "", "horizon": ""
            }),
            "aggregated_data": []
        }
    if intent == "error_general":
        return {
            "final_analysis": json.dumps({
                "recommendation": f"❌ System Error: {state.get('error', 'Unknown')}",
                "fundamental": "", "technical": "", "sentiment": "",
                "market_context": "", "narrative": "", "risks": "", "horizon": ""
            }),
            "aggregated_data": []
        }
    
    fundamental = state.get("fundamental_data", {})
    technical = state.get("technical_data", {})
    sentiment = state.get("sentiment_data", {})
    market_context = state.get("context_data", {})
    portfolio = state.get("portfolio_data", {})
    
    aggregation_results = []
    
    # Process scores for each ticker
    for ticker in tickers:
        f_score = fundamental.get(ticker, {}).get("fundamental_score", 0.0)
        t_score = technical.get(ticker, {}).get("technical_score", 0.0)
        s_score = sentiment.get(ticker, {}).get("sentiment_score", 0.0)
        m_score = market_context.get(ticker, {}).get("market_context_score", 0.0)
        
        # Final Score & Confidence (40% Fundamental, 30% Technical, 30% Sentiment)
        final_score = (0.4 * f_score) + (0.3 * t_score) + (0.3 * s_score)
        confidence_pct = round(final_score * 10, 1)

        # Recommendation logic
        if final_score >= 7.5: rec = "STRONG BUY"
        elif final_score >= 6.0: rec = "BUY"
        elif final_score >= 4.0: rec = "HOLD"
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
        aggregation_results.sort(key=lambda x: x['final_score'], reverse=True)
    
    if not tickers and intent not in ["general_query", "error_quota", "error_general"]:
        msg = "I'm sorry, I specialize in stock market and financial analysis." if intent == "unknown" else "No stock tickers could be resolved from your query."
        return {
            "final_analysis": json.dumps({
                "recommendation": msg,
                "fundamental": "", "technical": "", "sentiment": "",
                "market_context": "", "narrative": "", "risks": "", "horizon": ""
            }),
            "aggregated_data": []
        }

    # Step 1: Build DETERMINISTIC sections from raw agent data (always works)
    report_sections = {}
    if aggregation_results:
        target = aggregation_results[0]
        report_sections = build_deterministic_sections(target.get("raw_metrics", {}), target)

    # Step 2: Try LLM for enhanced narrative (optional upgrade)
    data_for_llm = json.dumps(aggregation_results, indent=2)
    conf_val = aggregation_results[0].get("confidence_level", "N/A") if aggregation_results else "N/A"
    
    template = """
### INSTITUTIONAL FINANCIAL RESEARCH REPORT GENERATOR
You are a senior equity research analyst. Generate a professional report based ONLY on the provided Agent Data.
Use the EXACT section headers below, separated by --- delimiters. Each bullet must be on its own line.

---

**FUNDAMENTAL ANALYSIS**
• **Revenue Growth:** [Insight]

• **P/E Ratio:** [Insight]

• **Return on Equity:** [Insight]

• **Debt-to-Equity:** [Insight]

• **Operating Margin:** [Insight]

---

**TECHNICAL ANALYSIS**
• **RSI:** [Insight]

• **MACD:** [Insight]

• **Moving Averages:** [Insight]

---

**SENTIMENT ANALYSIS**
• **Market Sentiment:** [Insight]

• **Risk/Confidence Drivers:** [Insight]

---

**MARKET CONTEXT**
• **Nifty 50 Trend:** [Insight]

• **Sector Performance:** [Insight]

• **Peer Comparison:** [Insight]

---

**AI NARRATIVE SUMMARY**
• **Growth Outlook:** [Insight]

• **Key Strengths:** [Insight]

• **Key Risks:** [Insight]

---

**RISK FACTORS**
• **Technical Risk:** [Insight]

• **Fundamental Risk:** [Insight]

• **Market Risk:** [Insight]

---

**FINAL RECOMMENDATION**
**[STRONG BUY / BUY / HOLD / AVOID]**
Confidence Level: {confidence_placeholder}

---

**INVESTMENT HORIZON**
• **Short-Term Outlook:** [Insight]

• **Long-Term Outlook:** [Insight]

---

RULES: Output ONLY analysis. No prompt instructions. Each bullet on a new line. Professional tone.

Agent Data:
{data_for_llm}
"""
    prompt = PromptTemplate(template=template, input_variables=["data_for_llm", "confidence_placeholder"])

    try:
        logger.info(f"Attempting {intent} institutional report generation...")
        response = invoke_with_failover(prompt, {
            "data_for_llm": data_for_llm,
            "confidence_placeholder": conf_val
        }, temperature=0.3)
        llm_report = response.content.strip()
        logger.info("LLM report generation successful.")
        
        # Parse LLM output and overlay onto deterministic sections
        llm_sections = parse_llm_report(llm_report)
        for key, value in llm_sections.items():
            if value and len(value.strip()) > 10:
                report_sections[key] = value
                
    except Exception as e:
        logger.error(f"LLM failure: {e}. Using deterministic sections only.")

    # Serialize sections as JSON string for state transport
    return {
        "final_analysis": json.dumps(report_sections),
        "aggregated_data": aggregation_results
    }
