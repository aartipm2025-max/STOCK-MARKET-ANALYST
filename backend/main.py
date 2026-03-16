from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

# Load .env FIRST — before any LLM module is imported
load_dotenv()

from backend.langgraph_workflow.graph import build_graph

graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    graph = build_graph()
    yield
    graph = None

app = FastAPI(title="Multi-Agent Market Analyst API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    mode: str = "Single Stock"

from utils.cache_utils import get_from_cache, save_to_cache, is_valid_analysis

class QueryResponse(BaseModel):
    intent: str
    tickers: list[str]
    analysis: str
    aggregated_data: list[dict] = []
    is_cached: bool = False

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze", response_model=QueryResponse)
async def analyze_query(req: QueryRequest):
    global graph
    try:
        # 1. Faster Routing Check (Extract Ticker for Cache Check)
        # We assume if it's a single ticker mention, we can check cache
        potential_ticker = req.query.strip().upper()
        if req.mode == "Single Stock" and potential_ticker:
            # Simple normalization for cache key
            clean_ticker = potential_ticker.split()[0].replace(".NS", "") + ".NS"
            cached_data = get_from_cache(clean_ticker)
            if cached_data:
                return QueryResponse(
                    intent="single_stock",
                    tickers=[clean_ticker],
                    analysis=cached_data["analysis"],
                    aggregated_data=cached_data["aggregated_data"],
                    is_cached=True
                )

        if not graph:
            graph = build_graph()
            
        initial_state = {
            "query": req.query,
            "mode": req.mode,
            "intent": "",
            "tickers": [],
            "fundamental_data": {},
            "technical_data": {},
            "sentiment_data": {},
            "context_data": {},
            "portfolio_data": {},
            "final_analysis": "",
            "aggregated_data": []
        }
        
        result_state = graph.invoke(initial_state)
        
        intent = result_state.get("intent", "unknown")
        tickers = result_state.get("tickers", [])
        analysis = result_state.get("final_analysis", "Analysis failed.")
        agg_data = result_state.get("aggregated_data", [])

        # 2. Post-Execution Cache Storage
        if req.mode == "Single Stock" and tickers and is_valid_analysis(analysis, agg_data):
            save_to_cache(tickers[0], {
                "analysis": analysis,
                "aggregated_data": agg_data
            })
        
        return QueryResponse(
            intent=intent,
            tickers=tickers,
            analysis=analysis,
            aggregated_data=agg_data,
            is_cached=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
