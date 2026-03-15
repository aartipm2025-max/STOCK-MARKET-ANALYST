import operator
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

from backend.langgraph_workflow.master_node import master_node_func
from backend.langgraph_workflow.aggregator_node import summarize_results

from agents.fundamental_agent import analyze_fundamentals
from agents.technical_agent import analyze_technicals
from agents.sentiment_agent import analyze_sentiment
from agents.portfolio_agent import analyze_portfolio
from utils.logger import get_logger

logger = get_logger("graph")

class AgentState(TypedDict):
    query: str
    intent: str
    tickers: List[str]
    fundamental_data: Dict[str, Any]
    technical_data: Dict[str, Any]
    sentiment_data: Dict[str, Any]
    portfolio_data: Dict[str, Any]
    final_analysis: str
    aggregated_data: List[Dict[str, Any]]

from concurrent.futures import ThreadPoolExecutor

def run_fundamental_agent(state: AgentState) -> dict:
    intent = state.get("intent")
    tickers = state.get("tickers", [])
    if intent not in ["single_stock", "comparison"] or not tickers:
        return {"fundamental_data": {}}
    
    logger.info(f"Fundamental Agent starting in parallel for {tickers}")
    with ThreadPoolExecutor() as executor:
        results = dict(zip(tickers, executor.map(analyze_fundamentals, tickers)))
    return {"fundamental_data": results}

def run_technical_agent(state: AgentState) -> dict:
    intent = state.get("intent")
    tickers = state.get("tickers", [])
    if intent not in ["single_stock", "comparison", "portfolio"] or not tickers:
        return {"technical_data": {}}
    
    logger.info(f"Technical Agent starting in parallel for {tickers}")
    with ThreadPoolExecutor() as executor:
        results = dict(zip(tickers, executor.map(analyze_technicals, tickers)))
    return {"technical_data": results}

def run_sentiment_agent(state: AgentState) -> dict:
    intent = state.get("intent")
    tickers = state.get("tickers", [])
    if intent not in ["single_stock", "comparison", "portfolio"] or not tickers:
        return {"sentiment_data": {}}
    
    logger.info(f"Sentiment Agent starting in parallel for {tickers}")
    with ThreadPoolExecutor() as executor:
        results = dict(zip(tickers, executor.map(analyze_sentiment, tickers)))
    return {"sentiment_data": results}

def run_portfolio_agent(state: AgentState) -> dict:
    intent = state.get("intent")
    tickers = state.get("tickers", [])
    if intent != "portfolio" or not tickers:
        return {"portfolio_data": {}}
    
    logger.info(f"Portfolio Agent starting for {tickers}")
    result = analyze_portfolio(tickers)
    return {"portfolio_data": result}
    
def build_graph():
    """Builds a robust parallel graph with fan-out to all agents."""
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("master_node", master_node_func)
    builder.add_node("fundamental_agent", run_fundamental_agent)
    builder.add_node("technical_agent", run_technical_agent)
    builder.add_node("sentiment_agent", run_sentiment_agent)
    builder.add_node("portfolio_agent", run_portfolio_agent)
    builder.add_node("aggregator_node", summarize_results)
    
    # Execution Flow
    builder.add_edge(START, "master_node")
    
    # Fan-out: One conditional edge that returns all agents
    builder.add_conditional_edges(
        "master_node",
        lambda x: ["fundamental_agent", "technical_agent", "sentiment_agent", "portfolio_agent"],
        {
            "fundamental_agent": "fundamental_agent",
            "technical_agent": "technical_agent",
            "sentiment_agent": "sentiment_agent",
            "portfolio_agent": "portfolio_agent"
        }
    )
    
    # Static Fan-in: Wait for all 4 incoming edges
    builder.add_edge("fundamental_agent", "aggregator_node")
    builder.add_edge("technical_agent", "aggregator_node")
    builder.add_edge("sentiment_agent", "aggregator_node")
    builder.add_edge("portfolio_agent", "aggregator_node")
    
    builder.add_edge("aggregator_node", END)
    
    return builder.compile()
