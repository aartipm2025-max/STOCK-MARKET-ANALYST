"""Streamlit rendering for Compare Stocks / Portfolio results.

Layout (top to bottom):
  1. Final decision banner with the reasons behind it
  2. Side-by-side comparison table + score chart + relative price performance
  3. Tabs with each stock's full detail
"""
import html
import re

import pandas as pd
import streamlit as st
import yfinance as yf

GREEN = ("#ecfdf5", "#a7f3d0", "#047857")
AMBER = ("#fffbeb", "#fde68a", "#b45309")
RED = ("#fef2f2", "#fecaca", "#b91c1c")

SECTION_CONFIG = [
    ("fundamental", "📊 Fundamental Analysis"),
    ("technical", "📈 Technical Analysis"),
    ("sentiment", "💬 Sentiment Analysis"),
    ("market_context", "🌐 Market Context"),
    ("narrative", "🤖 Strengths & Concerns"),
    ("risks", "⚠️ Risk Factors"),
    ("horizon", "⏳ Investment Horizon"),
]


def short(ticker: str) -> str:
    return ticker.replace(".NS", "").replace(".BO", "")


def _palette(label: str):
    label = (label or "").upper()
    if any(k in label for k in ("AVOID", "REDUCE", "REBALANCE", "SELL")):
        return RED
    if any(k in label for k in ("BUY", "ADD", "PICK")):
        return GREEN
    return AMBER


def _score_color(v):
    v = v or 0
    return GREEN[2] if v >= 7 else AMBER[2] if v >= 4 else RED[2]


def _plain(text: str) -> str:
    """Escape for HTML and turn **bold** markdown into <b>."""
    safe = html.escape(str(text))
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)


def _num(v):
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def price_history(tickers: tuple, period: str = "6mo") -> pd.DataFrame:
    frames = {}
    for t in tickers:
        try:
            close = yf.Ticker(t).history(period=period)["Close"]
            if not close.empty:
                close.index = close.index.tz_localize(None).normalize()
                frames[short(t)] = close
        except Exception:
            continue
    return pd.DataFrame(frames)


# ── 1. Verdict ──────────────────────────────────────────────────────────────
def render_verdict(verdict: dict, agg: list):
    mode = verdict.get("mode", "comparison")
    bg, border, color = _palette(verdict.get("call", ""))
    label = "FINAL DECISION" if mode == "comparison" else "PORTFOLIO VERDICT"

    sub = ""
    if mode == "comparison":
        pick = next((a for a in agg if a["ticker"] == verdict.get("pick")), None)
        if pick:
            sub = (f"{html.escape(pick.get('name', pick['ticker']))} · overall score "
                   f"<b>{pick['final_score']}/10</b> · rated <b>{html.escape(pick['recommendation'])}</b>")
    else:
        pm = verdict.get("portfolio_metrics") or {}
        sub = f"Average holding score <b>{pm.get('average_score', 'n/a')}/10</b> across {len(agg)} holdings"

    reasons = "".join(f"<li style='margin-bottom:0.45rem;'>{_plain(r)}</li>" for r in verdict.get("reasons", []))

    st.markdown(f"""
    <div style="background:{bg}; border:2px solid {border}; border-radius:20px; padding:2rem 2.25rem; margin-bottom:1.5rem;">
      <div style="color:{color}; font-size:0.85rem; font-weight:800; letter-spacing:1.5px; opacity:0.85;">{label}</div>
      <div style="color:{color}; font-size:2.8rem; font-weight:900; letter-spacing:-1.5px; line-height:1.1; margin:0.4rem 0;">{html.escape(verdict.get('call', ''))}</div>
      <div style="color:#1e293b; font-size:1.35rem; font-weight:700; margin-bottom:0.35rem;">{html.escape(verdict.get('headline', ''))}</div>
      <div style="color:#475569; font-size:1rem; margin-bottom:1.1rem;">{sub}</div>
      <div style="color:#1e293b; font-size:0.95rem; font-weight:800; margin-bottom:0.4rem;">Why</div>
      <ul style="color:#334155; font-size:1rem; line-height:1.55; padding-left:1.2rem; margin:0;">{reasons}</ul>
    </div>
    """, unsafe_allow_html=True)

    key_data = verdict.get("key_data") or []
    if key_data and key_data != verdict.get("reasons"):
        with st.expander("Key data points behind this decision"):
            st.markdown("\n".join(f"- {k}" for k in key_data))

    if mode == "portfolio":
        pm = verdict.get("portfolio_metrics") or {}
        ytd, vol = _num(pm.get("return_ytd")), _num(pm.get("volatility"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("YTD return (equal-weight)", f"{ytd * 100:.1f}%" if ytd is not None else "n/a")
        c2.metric("Annualised volatility", f"{vol * 100:.1f}%" if vol is not None else "n/a")
        c3.metric("Risk level", str(pm.get("risk") or "n/a").capitalize())
        c4.metric("Diversification", str(pm.get("diversification") or "n/a").capitalize())
        if pm.get("sectors"):
            st.caption("Sectors held: " + ", ".join(pm["sectors"]))


# ── 2. Comparison ───────────────────────────────────────────────────────────
def _row(a: dict, mode: str) -> dict:
    f = (a.get("raw_metrics") or {}).get("f") or {}
    fv = f.get("raw_values") or {}
    ind = ((a.get("raw_metrics") or {}).get("t") or {}).get("indicators") or {}
    sc = a.get("scores", {})
    pct = lambda v: None if _num(v) is None else round(_num(v) * 100, 1)
    row = {
        "Rank": a.get("rank"),
        "Stock": short(a["ticker"]),
        "Company": a.get("name") or a["ticker"],
        "Overall": a.get("final_score"),
        "Rating": a.get("recommendation"),
    }
    if mode == "portfolio":
        row["Action"] = a.get("action")
    row.update({
        "Price": _num(ind.get("current_price")) or None,
        "Fundamental": sc.get("fundamental"),
        "Technical": sc.get("technical"),
        "Sentiment": sc.get("sentiment"),
        "Market": sc.get("market_context"),
        "P/E": None if _num(fv.get("pe_ratio")) is None else round(_num(fv["pe_ratio"]), 1),
        "ROE %": pct(fv.get("roe")),
        "Rev. growth %": pct(fv.get("revenue_growth")),
        "Op. margin %": pct(fv.get("operating_margin")),
        "Debt/Equity": None if _num(fv.get("debt_to_equity")) is None else round(_num(fv["debt_to_equity"]), 2),
        "RSI": _num(ind.get("rsi")),
        "MACD": (ind.get("macd_signal") or "").capitalize() or None,
    })
    return row


def render_comparison(agg: list, mode: str):
    st.markdown("### 📋 Side-by-side Comparison")
    df = pd.DataFrame([_row(a, mode) for a in agg])
    score_col = lambda label: st.column_config.ProgressColumn(label, min_value=0, max_value=10, format="%.1f")
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn("#", format="%d", width="small"),
            "Overall": score_col("Overall /10"),
            "Fundamental": score_col("Fundamental"),
            "Technical": score_col("Technical"),
            "Sentiment": score_col("Sentiment"),
            "Market": score_col("Market"),
            "Price": st.column_config.NumberColumn("Price", format="%.2f"),
            "RSI": st.column_config.NumberColumn("RSI", format="%.0f"),
        },
    )
    st.caption("Overall = 40% fundamental + 30% technical + 30% sentiment. "
               "Rating bands: ≥7.5 Strong Buy · ≥6 Buy · ≥4 Hold · <4 Avoid. Blank cells = data not reported by Yahoo Finance.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Score breakdown")
        scores = df.set_index("Stock")[["Fundamental", "Technical", "Sentiment", "Market"]]
        try:
            st.bar_chart(scores.T, stack=False, height=320)
        except Exception:  # grouped bars need Altair 5; fall back to one bar group per stock
            st.bar_chart(scores, height=320)
    with right:
        st.markdown("#### 6-month relative performance (rebased to 100)")
        hist = price_history(tuple(a["ticker"] for a in agg))
        if not hist.empty:
            hist = hist.ffill().dropna(how="all")
            rebased = hist / hist.bfill().iloc[0] * 100
            st.line_chart(rebased, height=320)
        else:
            st.info("Price history unavailable right now.")


# ── 3. Per-stock details ────────────────────────────────────────────────────
def render_details(agg: list, stocks: dict, mode: str, clean_fn):
    st.markdown("### 🔍 Stock-by-stock Details")
    tabs = st.tabs([f"#{a.get('rank', i + 1)} {short(a['ticker'])}" for i, a in enumerate(agg)])
    for tab, a in zip(tabs, agg):
        with tab:
            sections = stocks.get(a["ticker"], {})
            call = a.get("action") if mode == "portfolio" else a.get("recommendation")
            bg, border, color = _palette(call)
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.75rem;
                        background:{bg}; border:1px solid {border}; border-radius:14px; padding:1rem 1.25rem; margin:0.5rem 0 1rem;">
              <div>
                <div style="font-size:1.4rem; font-weight:800; color:#1e293b;">{html.escape(a.get('name') or a['ticker'])}</div>
                <div style="color:#64748b; font-weight:600;">{html.escape(a['ticker'])} · Data up to {html.escape(str(a.get('analysis_date', 'n/a')))}</div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:1.6rem; font-weight:900; color:{color};">{html.escape(call or '')}</div>
                <div style="color:#475569; font-weight:700;">Overall {a.get('final_score')}/10</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if sections.get("ai_note"):
                st.info(f"**AI take:** {sections['ai_note']}")

            sc = a.get("scores", {})
            cols = st.columns(4)
            for col, (key, label) in zip(cols, [("fundamental", "FUNDAMENTAL"), ("technical", "TECHNICAL"),
                                                ("sentiment", "SENTIMENT"), ("market_context", "CONTEXT")]):
                col.markdown(
                    f'<div class="score-card"><div style="color:#64748b; font-size:0.8rem; font-weight:700;">{label}</div>'
                    f'<div style="font-size:2rem; font-weight:800; color:{_score_color(sc.get(key))};">{sc.get(key, 0)}</div></div>',
                    unsafe_allow_html=True)

            hist = price_history((a["ticker"],))
            if not hist.empty:
                st.markdown(f"#### Price: {short(a['ticker'])} (6 months)")
                st.line_chart(hist, height=260)

            for key, title in SECTION_CONFIG:
                content = clean_fn(sections.get(key, ""))
                if content:
                    st.markdown(f"#### {title}")
                    st.markdown(content)


def render_multi_report(report: dict, agg: list, clean_fn):
    mode = report.get("mode", "comparison")
    render_verdict(report.get("verdict", {}), agg)
    st.divider()
    render_comparison(agg, mode)
    st.divider()
    render_details(agg, report.get("stocks", {}), mode, clean_fn)
