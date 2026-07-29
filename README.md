# Market Analyst AI — Multi-Agent Stock Analysis System

A production-grade, multi-agent AI system for Indian stock market analysis. Built using **LangGraph**, **LangChain**, **FastAPI**, and **Streamlit**, it orchestrates specialized AI agents in parallel to deliver institutional-quality investment reports.

---

## What It Does

Users type a natural language query — _"How is Reliance doing?"_ or _"TCS vs Infosys"_ or _"Analyze my portfolio: TCS, HDFC, Wipro"_ — and the system:

1. Classifies intent and resolves stock tickers via an LLM
2. Fans out to 4–5 specialized agents **running in parallel**
3. Aggregates scores and generates a structured investment report
4. Displays it on a premium Streamlit dashboard with BUY/HOLD/SELL recommendation

---

## System Architecture

```
User (Streamlit UI)
        │
        ▼
FastAPI Backend (/analyze)
        │
        ▼
LangGraph Master Node
(Intent Classifier + Ticker Resolver via LLM)
        │
        ├──────────────────────────────────────────┐
        ▼          ▼           ▼          ▼         ▼
 Fundamental  Technical  Sentiment  Market     Portfolio
   Agent       Agent      Agent     Context     Agent
 (yfinance)  (yfinance) (DuckDuckGo) Agent    (concurrent)
        │          │           │          │         │
        └──────────┴───────────┴──────────┴─────────┘
                              │
                              ▼
                    Aggregator Node
                 (LLM Investment Report)
                              │
                              ▼
                  Structured JSON Report
            (Fundamental | Technical | Sentiment |
             Market Context | Recommendation)
```

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-Agent Orchestration** | LangGraph state machine fans out to 5 agents simultaneously |
| **Parallel Execution** | `ThreadPoolExecutor` runs agents concurrently per ticker |
| **Three Query Modes** | Single stock, stock comparison, portfolio analysis |
| **LLM Intent Detection** | Natural language → structured intent + NSE ticker resolution |
| **LLM Failover Chain** | Groq → Google Gemini → OpenAI (automatic fallback) |
| **Sector-Aware Scoring** | Banking stocks use different debt thresholds (D/E < 2.5 vs 1.0) |
| **Daily Cache Layer** | JSON file cache invalidated per ticker per day, validated before storage |
| **Dynamic Score Weights** | Only available metrics contribute to final score (no zero penalties) |
| **Streamlit Dashboard** | Score cards, price charts, structured sections, BUY/HOLD/SELL banner |

---

## Agent Details

### 1. Fundamental Agent (`agents/fundamental_agent.py`)
A 6-step pipeline class (`FundamentalAnalyst`):
- **Fetch** — Pulls `info`, `financials`, `balance_sheet`, `fast_info` from Yahoo Finance
- **Normalize** — Maps multiple possible column names to canonical keys (handles bank vs non-bank financials)
- **Validate & Compute** — Derives ROE, Debt/Equity, Revenue Growth, Operating Margin, P/E
- **Score** — Weighted partial scoring: only available metrics contribute; minimum score floor of 3.0
- **Confidence** — Reports `high/medium/low/none` based on available metric count

**Key design decision:** If a metric is unavailable, it is excluded from score calculation rather than penalized. This prevents data gaps from unfairly downgrading stocks.

### 2. Technical Agent (`agents/technical_agent.py`)
Manually computes indicators using pandas (avoids `pandas-ta` Python 3.11 incompatibility):
- **SMA 50 / SMA 200** — Golden Cross / Death Cross detection
- **RSI (14-period)** — Overbought/Oversold with epsilon for zero-division safety
- **MACD / Signal Line** — Bullish/Bearish momentum
- **Volume Trend** — 20-day rolling average comparison

Score: 2.5 points per passing condition (max 10.0)

### 3. Sentiment Agent (`agents/sentiment_agent.py`)
- Fetches recent news via **DuckDuckGo Search** (no API key required)
- Falls back to company name search if ticker search yields no results
- Passes headlines to an LLM for Positive/Neutral/Negative classification with a 1–10 score
- Returns sentiment summary written as a human-readable paragraph

### 4. Market Context Agent (`agents/market_context_agent.py`)
Analyzes macroeconomic and sector-level context — NIFTY 50 trends, sector benchmarks, index correlation — to contextualize individual stock performance.

### 5. Portfolio Agent (`agents/portfolio_agent.py`)
Loops through multiple tickers using `ThreadPoolExecutor`, runs fundamental + technical analysis per stock, and aggregates a portfolio-level health report.

---

## Orchestration: LangGraph Workflow

**File:** `backend/langgraph_workflow/graph.py`

```python
# State flows through nodes using a TypedDict (AgentState)
START → master_node → [parallel fan-out] → aggregator_node → END
```

- `master_node` — LLM classifies intent (`single_stock | comparison | portfolio | general_query`), resolves tickers, applies UI mode override
- Conditional edge returns all 5 agent node names simultaneously (LangGraph fan-out)
- All 5 agents converge into `aggregator_node` (fan-in) which calls an LLM to generate the final report
- Final output is a structured JSON with keys: `fundamental`, `technical`, `sentiment`, `market_context`, `narrative`, `risks`, `horizon`, `recommendation`

---

## LLM Failover Strategy

**File:** `utils/llm_utils.py`

```
Primary:   Groq (llama3-70b) — fastest, free tier
Fallback 1: Google Gemini
Fallback 2: OpenAI GPT-4o-mini
```

If the primary LLM hits a rate limit (HTTP 429), the system automatically retries with the next provider — zero downtime for the user.

---

## Cache Layer

**File:** `utils/cache_utils.py`

- Cache key: `{TICKER}_{YYYY-MM-DD}.json` — automatically invalidated daily
- **Validation before write:** checks for forbidden phrases (`"data unavailable"`), score > 0, all required sections present
- **Read path in FastAPI:** cache checked _before_ invoking LangGraph — sub-second response for repeated queries
- Cache stored in `backend/cache/` (gitignored)

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit with custom CSS (Plus Jakarta Sans font, card-based layout) |
| API | FastAPI + Uvicorn (async, lifespan-managed graph initialization) |
| Orchestration | LangGraph (StateGraph with fan-out/fan-in) |
| LLM Framework | LangChain (PromptTemplate, failover chain) |
| LLM Providers | Groq / Google Gemini / OpenAI |
| Financial Data | yfinance (price history, financials, balance sheet) |
| News Data | duckduckgo-search (no API key needed) |
| Concurrency | Python `concurrent.futures.ThreadPoolExecutor` |
| Caching | File-based JSON (date-keyed, validated) |
| Language | Python 3.11 |

---

## Project Structure

```
STOCK MARKET ANALYST/
├── agents/
│   ├── fundamental_agent.py     # 6-step financial pipeline class
│   ├── technical_agent.py       # Pandas-based indicator computation
│   ├── sentiment_agent.py       # DuckDuckGo + LLM sentiment scoring
│   ├── market_context_agent.py  # Macro/sector context
│   └── portfolio_agent.py       # Multi-stock concurrent evaluator
├── backend/
│   ├── main.py                  # FastAPI app with cache-first routing
│   └── langgraph_workflow/
│       ├── graph.py             # LangGraph StateGraph builder
│       ├── master_node.py       # Intent classifier + ticker resolver
│       └── aggregator_node.py   # Final LLM report generator
├── tools/
│   ├── yahoo_finance_tool.py    # yfinance wrapper
│   └── duckduckgo_tool.py       # News fetcher
├── models/
│   └── sentiment_model.py       # LLM sentiment classifier
├── utils/
│   ├── cache_utils.py           # File-based daily cache
│   ├── llm_utils.py             # LLM failover + JSON extractor
│   ├── indicators.py            # Technical indicator helpers
│   └── logger.py                # Structured logging
├── ui/
│   └── streamlit_app.py         # Full-page dashboard UI
├── requirements.txt
└── architecture.md
```

---

## Setup & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```
Only one LLM key is required — the system will failover automatically.

### 3. Run the Streamlit UI (Integrated Mode)
```bash
streamlit run ui/streamlit_app.py
```

### 4. Run with FastAPI Backend (Optional)
```bash
# Terminal 1 — Start API
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Start UI
streamlit run ui/streamlit_app.py
```
The UI auto-detects whether the FastAPI server is running and falls back to integrated mode if not.

---

## Example Queries

| Query | Mode | What Happens |
|---|---|---|
| `Reliance` | Single Stock | Full 4-agent analysis + BUY/HOLD/SELL |
| `TCS vs Infosys` | Compare Stocks | Side-by-side score comparison |
| `TCS, Wipro, Infosys, HDFC` | Portfolio | Concurrent analysis of all 4 stocks |
| `Zomato` | Single Stock | Resolves to `ETERNAL.NS` (post-rebrand) |

---

## Design Decisions & Trade-offs

**Why LangGraph over a simple sequential chain?**
LangGraph enables true parallel fan-out — all 5 agents run concurrently. A sequential chain would take 5× longer. LangGraph's `StateGraph` also provides structured state passing, making the system easier to extend.

**Why file-based cache instead of Redis?**
Simplicity and zero infrastructure overhead. Stock analysis is only valid intraday, so date-keyed JSON files (auto-expire daily) are sufficient without the operational burden of a cache server.

**Why custom pandas indicators instead of pandas-ta?**
`pandas-ta` has Python 3.11 compatibility issues. Computing RSI, MACD, and SMAs manually with pandas gives identical results with no dependency risk.

**Why DuckDuckGo for news?**
No API key required, no rate-limit billing. Acceptable latency for batch news fetching, and sufficient for sentiment signal extraction.

**Why dynamic scoring weights?**
If a stock has only 2 of 5 metrics available, the score is calculated over 2 metrics — not penalized for the 3 missing ones. This avoids misleading low scores when data is legitimately unavailable (common with newer or less-covered Indian stocks).

---

## Key Engineering Challenges Solved

- **yfinance data inconsistency** — Different column names across sectors (`Total Revenue` vs `Interest Income` for banks). Solved with multi-key lookup with fallback chains.
- **Indian stock corporate actions** — Zomato rebranded to Eternal Ltd; hardcoded resolution rules in the master node prompt.
- **LLM hallucination on tickers** — Post-LLM processing enforces `.NS` suffix for all Indian tickers, ensuring Yahoo Finance compatibility.
- **Zero-score edge case** — Scores below 3.0 are validated against actual negative signals (negative ROE/revenue) before being kept low; otherwise floored at 3.0 neutral.
- **Cache poisoning** — Cache entries are validated before write: forbidden phrases checked, scores verified > 0, all report sections confirmed present.
