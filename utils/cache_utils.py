import os
import json
from datetime import datetime
from utils.logger import get_logger

logger = get_logger("cache_utils")

# Resolve absolute path so cache works on both local and Streamlit Cloud
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_BASE_DIR, "backend", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(ticker: str) -> str:
    """Generates a key based on ticker and today's date."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    return f"{ticker.upper()}_{date_str}.json"

def is_valid_analysis(analysis_data: str, aggregated_data: list) -> bool:
    """Validates if the analysis is complete and high-quality."""
    try:
        report = json.loads(analysis_data)
        
        # 1. Basic Structure Checks
        if not report.get("recommendation") or not any(k in report for k in ["fundamental", "technical", "sentiment"]):
            return False
            
        # 2. Score Validation from aggregated_data
        if not aggregated_data:
            return False
            
        target = aggregated_data[0]
        scores = {
            "fundamental": target.get("f_score", 0),
            "technical": target.get("t_score", 0),
            "sentiment": target.get("s_score", 0),
            "context": target.get("m_score", 0)
        }
        
        # Fundamental Score must be > 0
        if scores["fundamental"] <= 0:
            return False
            
        # 3. Content Validation (No "unavailable" placeholders)
        forbidden_phrases = ["data unavailable", "analysis unavailable", "unable to retrieve data"]
        report_text = json.dumps(report).lower()
        
        if any(phrase in report_text for phrase in forbidden_phrases):
            return False
            
        # 4. Confidence level check
        # Usually stored in recommendation text or aggregated data
        conf = target.get("confidence_level", 0)
        if conf == 0:
            # Check if it exists as string in recommendation text
            if "Confidence Level: N/A" in report.get("recommendation", ""):
                 return False
                 
        return True
    except Exception as e:
        logger.error(f"Cache validation error: {e}")
        return False

def get_from_cache(ticker: str):
    """Retrieves analysis from JSON cache if valid for today."""
    key = get_cache_key(ticker)
    path = os.path.join(CACHE_DIR, key)
    
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                logger.info(f"Cache hit for {ticker}")
                return data
        except Exception as e:
            logger.error(f"Error reading cache for {ticker}: {e}")
    return None

def save_to_cache(ticker: str, result_data: dict):
    """Stores a validated analysis result in the cache."""
    key = get_cache_key(ticker)
    path = os.path.join(CACHE_DIR, key)
    
    try:
        with open(path, "w") as f:
            json.dump(result_data, f, indent=4)
        logger.info(f"Cached results for {ticker}")
    except Exception as e:
        logger.error(f"Error saving cache for {ticker}: {e}")
