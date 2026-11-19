"""Multi-stock decision engine for Compare Stocks and Portfolio modes.

The final decision (which stock wins / what to do with the portfolio) is made
deterministically from the agent scores so it always matches the comparison
table. The LLM is only used to explain that decision in plain language.
"""
import json
from statistics import mean

from langchain_core.prompts import PromptTemplate

from utils.llm_utils import invoke_with_failover, extract_json
from utils.logger import get_logger

logger = get_logger("comparison")

DIMENSIONS = [
    ("fundamental", "fundamentals"),
    ("technical", "technical momentum"),
    ("sentiment", "news sentiment"),
    ("market_context", "market context"),
]

PORTFOLIO_ACTIONS = [(7.5, "ADD"), (6.0, "HOLD"), (4.0, "WATCH"), (float("-inf"), "REDUCE")]


# ── helpers ─────────────────────────────────────────────────────────────────
def _num(v):
    try:
        f = float(v)
        return None if f != f else f  # NaN check
    except (TypeError, ValueError):
        return None


def short(ticker: str) -> str:
    return ticker.replace(".NS", "").replace(".BO", "")


def _f(a):
    return (a.get("raw_metrics") or {}).get("f") or {}


def _t(a):
    return ((a.get("raw_metrics") or {}).get("t") or {}).get("indicators") or {}


def _s(a):
    return (a.get("raw_metrics") or {}).get("s") or {}


def company_name(a) -> str:
    return (_f(a).get("metrics") or {}).get("long_name") or a["ticker"]


def portfolio_action(score: float) -> str:
    for threshold, action in PORTFOLIO_ACTIONS:
        if score >= threshold:
            return action
    return "REDUCE"


def stock_signals(a):
    """Return (strengths, weaknesses) as short human-readable phrases."""
    fv = _f(a).get("raw_values") or {}
    is_bank = _f(a).get("is_bank", False)
    ind = _t(a)
    strengths, weaknesses = [], []

    roe = _num(fv.get("roe"))
    if roe is not None:
        if roe >= 0.15:
            strengths.append(f"high ROE of {roe * 100:.1f}%")
        elif roe < 0.08:
            weaknesses.append(f"weak ROE of {roe * 100:.1f}%")

    growth = _num(fv.get("revenue_growth"))
    if growth is not None:
        if growth >= 0.10:
            strengths.append(f"revenue growing {growth * 100:.1f}%")
        elif growth < 0:
            weaknesses.append(f"revenue shrinking {growth * 100:.1f}%")

    de = _num(fv.get("debt_to_equity"))
    if de is not None and not is_bank:
        if de < 0.5:
            strengths.append(f"low debt (D/E {de:.2f})")
        elif de > 1.5:
            weaknesses.append(f"high debt (D/E {de:.2f})")

    margin = _num(fv.get("operating_margin"))
    if margin is not None:
        if margin >= 0.20:
            strengths.append(f"strong operating margin of {margin * 100:.1f}%")
        elif margin < 0.08:
            weaknesses.append(f"thin operating margin of {margin * 100:.1f}%")

    pe = _num(fv.get("pe_ratio"))
    if pe is not None and pe > 0:
        if pe < 15:
            strengths.append(f"attractive valuation (P/E {pe:.1f})")
        elif pe > 40:
            weaknesses.append(f"expensive valuation (P/E {pe:.1f})")

    macd = ind.get("macd_signal")
    if macd == "bullish":
        strengths.append("bullish MACD momentum")
    elif macd == "bearish":
        weaknesses.append("bearish MACD momentum")

    price, sma200 = _num(ind.get("current_price")), _num(ind.get("sma_200"))
    if price and sma200:
        if price > sma200:
            strengths.append("trading above its 200-day average (long-term uptrend)")
        else:
            weaknesses.append("trading below its 200-day average (long-term downtrend)")

    rsi = _num(ind.get("rsi"))
    if rsi is not None:
        if rsi > 70:
            weaknesses.append(f"overbought (RSI {rsi:.0f})")
        elif rsi < 30:
            strengths.append(f"oversold (RSI {rsi:.0f}) — possible rebound")

    s_score = _num(a.get("scores", {}).get("sentiment"))
    if s_score is not None and _s(a).get("articles", 0):
        if s_score >= 6.5:
            strengths.append("positive news flow")
        elif s_score <= 3.5:
            weaknesses.append("negative news flow")

    return strengths, weaknesses


def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ── deterministic verdicts ──────────────────────────────────────────────────
def comparison_verdict(ranked: list) -> dict:
    w, r = ranked[0], ranked[1]
    wn, rn = short(w["ticker"]), short(r["ticker"])
    margin = w["final_score"] - r["final_score"]

    if w["recommendation"] == "AVOID":
        headline = f"None of these look attractive right now — {wn} is the least weak"
        call = "AVOID ALL"
    elif margin < 0.3:
        headline = f"Close call — {wn} edges ahead of {rn}"
        call = f"PICK {wn}"
    else:
        headline = f"{wn} is the better pick"
        call = f"PICK {wn}"

    others = ", ".join(f"{short(x['ticker'])} {x['final_score']}" for x in ranked[1:])
    reasons = [f"Highest overall score: {wn} {w['final_score']}/10 vs {others}."]

    leads = [label for key, label in DIMENSIONS
             if w["scores"].get(key, 0) >= max(x["scores"].get(key, 0) for x in ranked)]
    if leads:
        reasons.append(f"{wn} leads (or ties) on {_join(leads)}.")

    w_str, w_weak = stock_signals(w)
    if w_str:
        reasons.append(f"Key strengths: {_join(w_str[:3])}.")

    for other in ranked[1:]:
        better = [label for key, label in DIMENSIONS
                  if other["scores"].get(key, 0) > w["scores"].get(key, 0)]
        if better:
            reasons.append(f"Where {short(other['ticker'])} is stronger: {_join(better)}.")
    if w_weak:
        reasons.append(f"Watch out: {wn} shows {_join(w_weak[:2])}.")

    return {
        "mode": "comparison",
        "pick": w["ticker"],
        "call": call,
        "headline": headline,
        "rating": w["recommendation"],
        "reasons": reasons,
    }


def portfolio_verdict(ranked: list, portfolio: dict) -> dict:
    avg = mean(x["final_score"] for x in ranked)
    best, worst = ranked[0], ranked[-1]
    reduce_list = [short(x["ticker"]) for x in ranked if x.get("action") == "REDUCE"]
    add_list = [short(x["ticker"]) for x in ranked if x.get("action") == "ADD"]

    if avg >= 6.5 and not reduce_list:
        call, headline, rating = "HOLD & ADD", "Healthy portfolio — stay invested", "BUY"
    elif avg >= 5.0:
        call, headline, rating = "HOLD", "Solid portfolio — minor rebalancing suggested", "HOLD"
    else:
        call, headline, rating = "REBALANCE", "Portfolio needs rebalancing", "AVOID"

    reasons = [f"Average holding score is {avg:.1f}/10 across {len(ranked)} stocks."]
    b_str, _ = stock_signals(best)
    reasons.append(f"Strongest holding: {short(best['ticker'])} ({best['final_score']}/10)"
                   + (f" — {_join(b_str[:2])}." if b_str else "."))
    _, w_weak = stock_signals(worst)
    reasons.append(f"Weakest holding: {short(worst['ticker'])} ({worst['final_score']}/10)"
                   + (f" — {_join(w_weak[:2])}." if w_weak else "."))
    if reduce_list:
        reasons.append(f"Consider reducing: {', '.join(reduce_list)}.")
    if add_list:
        reasons.append(f"Candidates to add to: {', '.join(add_list)}.")

    pm = (portfolio or {}).get("metrics") or {}
    vol, ytd = _num(pm.get("volatility")), _num(pm.get("return_ytd"))
    div = pm.get("sector_diversification")
    if vol is not None:
        reasons.append(f"{(portfolio.get('risk_score') or 'unknown').capitalize()} risk: "
                       f"annualised volatility {vol * 100:.1f}%"
                       + (f", YTD return ~{ytd * 100:.1f}%" if ytd is not None else "") + ".")
    if div == "low":
        sectors = pm.get("sectors") or []
        reasons.append("Low diversification" + (f" — concentrated in {_join(sectors)}" if sectors else "")
                       + ". Consider adding other sectors.")

    return {
        "mode": "portfolio",
        "pick": best["ticker"],
        "call": call,
        "headline": headline,
        "rating": rating,
        "reasons": reasons,
        "portfolio_metrics": {
            "average_score": round(avg, 2),
            "return_ytd": ytd,
            "volatility": vol,
            "diversification": div,
            "risk": (portfolio or {}).get("risk_score"),
            "sectors": pm.get("sectors") or [],
        },
    }


# ── LLM explanation (optional upgrade) ──────────────────────────────────────
def _compact(a) -> dict:
    ind = _t(a)
    return {
        "ticker": a["ticker"],
        "name": company_name(a),
        "overall_score": a["final_score"],
        "rating": a["recommendation"],
        "portfolio_action": a.get("action"),
        "scores": a["scores"],
        "fundamentals": {k: v for k, v in (_f(a).get("metrics") or {}).items() if k != "long_name"},
        "technicals": {k: ind.get(k) for k in ("current_price", "rsi", "macd_signal", "sma_50", "sma_200", "volume_trend")},
        "news_sentiment": (_s(a).get("summary") or "")[:300],
    }


def llm_explain(verdict: dict, ranked: list, portfolio: dict) -> dict:
    """Ask the LLM to justify the fixed verdict. Returns {} on any failure."""
    if verdict["mode"] == "comparison":
        task = "Compare these stocks for an investor deciding which ONE to buy."
        decision = f"{verdict['headline']} (pick: {verdict['pick']})"
    else:
        task = "Review this investor's portfolio and justify the overall action and each holding's action."
        decision = f"{verdict['headline']} — overall action {verdict['call']}"

    data = {"stocks": [_compact(a) for a in ranked]}
    if verdict["mode"] == "portfolio":
        data["portfolio_metrics"] = verdict.get("portfolio_metrics")

    template = """You are a senior equity research analyst. {task}
The quantitative model has ALREADY decided: {decision}
Do NOT contradict it and do NOT pick a different stock. Explain WHY using ONLY the data below.
Be specific and cite numbers (scores, P/E, ROE, revenue growth, RSI, etc.). Plain English, no jargon overload.

Return STRICT JSON only, no preamble:
{{
  "reasons": ["3 to 5 concise one-sentence reasons supporting the decision, most important first"],
  "stock_notes": {{"<TICKER>": "one-sentence takeaway for that stock"}}
}}

Data:
{data}
"""
    prompt = PromptTemplate(template=template, input_variables=["task", "decision", "data"])
    try:
        resp = invoke_with_failover(prompt, {"task": task, "decision": decision,
                                             "data": json.dumps(data, default=str)}, temperature=0.2)
        out = extract_json(resp.content)
        reasons = [r.strip() for r in out.get("reasons", []) if isinstance(r, str) and r.strip()]
        notes = {str(k).upper(): v for k, v in (out.get("stock_notes") or {}).items() if isinstance(v, str)}
        return {"reasons": reasons[:5] if len(reasons) >= 2 else [], "stock_notes": notes}
    except Exception as e:
        logger.error(f"LLM verdict explanation failed: {e}. Using rule-based reasons.")
        return {}


def build_multi_stock_report(intent: str, aggregation_results: list, portfolio: dict, section_builder) -> dict:
    """Build the full multi-stock report: verdict on top, comparison rows, per-stock details."""
    ranked = sorted(aggregation_results, key=lambda x: x["final_score"], reverse=True)
    for i, a in enumerate(ranked, 1):
        a["rank"] = i
        a["name"] = company_name(a)
        if intent == "portfolio":
            a["action"] = portfolio_action(a["final_score"])

    verdict = portfolio_verdict(ranked, portfolio) if intent == "portfolio" else comparison_verdict(ranked)
    verdict["key_data"] = list(verdict["reasons"])  # rule-based facts, always shown

    explained = llm_explain(verdict, ranked, portfolio)
    if explained.get("reasons"):
        verdict["reasons"] = explained["reasons"]
    notes = explained.get("stock_notes", {})

    stocks = {}
    for a in ranked:
        sections = section_builder(a.get("raw_metrics", {}), a)
        note = notes.get(a["ticker"].upper()) or notes.get(short(a["ticker"]).upper())
        if note:
            sections["ai_note"] = note
        stocks[a["ticker"]] = sections

    return {"ranked": ranked, "verdict": verdict, "stocks": stocks}
