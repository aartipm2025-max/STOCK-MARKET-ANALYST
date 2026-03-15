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

class QueryResponse(BaseModel):
    intent: str
    tickers: list[str]
    analysis: str
    aggregated_data: list[dict] = []

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze", response_model=QueryResponse)
async def analyze_query(req: QueryRequest):
    global graph
    try:
        if not graph:
            graph = build_graph()
            
        initial_state = {
            "query": req.query,
            "intent": "",
            "tickers": [],
            "fundamental_data": {},
            "technical_data": {},
            "sentiment_data": {},
            "portfolio_data": {},
            "final_analysis": "",
            "aggregated_data": []
        }
        
        result_state = graph.invoke(initial_state)
        
        return QueryResponse(
            intent=result_state.get("intent", "unknown"),
            tickers=result_state.get("tickers", []),
            analysis=result_state.get("final_analysis", "Analysis failed."),
            aggregated_data=result_state.get("aggregated_data", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
