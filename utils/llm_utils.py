import os
import json
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from utils.logger import get_logger

logger = get_logger("llm_utils")
EXHAUSTED_PROVIDERS = {} # {provider_name: expiry_timestamp}

def get_llm_instance(provider, temperature=0):
    """Factory to return an LLM instance based on provider."""
    # Circuit breaker: skip if exhausted
    if provider in EXHAUSTED_PROVIDERS:
        if time.time() < EXHAUSTED_PROVIDERS[provider]:
            return None
        else:
            del EXHAUSTED_PROVIDERS[provider]

    if provider == "gemini":
        model_name = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: return None
        # max_retries=0 ensures we fail FAST and move to next provider
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key, max_retries=0)
    
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: return None
        return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=temperature, groq_api_key=api_key, max_retries=0)
    
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: return None
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=temperature, openai_api_key=api_key, max_retries=0)
    
    return None

def invoke_with_failover(prompt, input_vars, temperature=0):
    """
    Invokes LLM with failover across providers: Gemini -> Groq -> OpenAI.
    """
    providers = ["gemini", "groq", "openai"]
    last_error = None

    for provider in providers:
        try:
            llm = get_llm_instance(provider, temperature)
            if not llm:
                logger.warning(f"Skipping {provider} (API key not set).")
                continue
            
            logger.info(f"Attempting LLM invocation with {provider}...")
            chain = prompt | llm
            response = chain.invoke(input_vars)
            logger.info(f"Success with {provider}!")
            return response
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            logger.warning(f"Error with {provider}: {e}")
            
            # Common quota/rate limit indicators
            quota_errors = ["429", "quota", "resource_exhausted", "rate_limit"]
            if any(qe in error_str for qe in quota_errors):
                logger.info(f"{provider} exhausted. Blacklisting for 5 mins. Trying next provider...")
                EXHAUSTED_PROVIDERS[provider] = time.time() + 300 # 5 min blacklist
                continue
            else:
                continue

    logger.error("All LLM providers failed or are exhausted.")
    raise last_error

def extract_json(content: str) -> dict:
    """Robustly extracts JSON from a string that might contain chatty text."""
    content = content.strip()
    # Try direct load first
    try:
        # Pre-clean known markdown blocks
        clean = content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        pass
    
    # Try finding the first '{' and last '}'
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            json_str = content[start:end+1]
            return json.loads(json_str)
    except Exception:
        pass
        
    raise ValueError(f"Could not extract valid JSON from LLM response: {content[:100]}...")
