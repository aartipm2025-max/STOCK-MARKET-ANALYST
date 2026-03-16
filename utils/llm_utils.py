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
            quota_errors = ["429", "quota", "resource_exhausted", "rate_limit", "leaked", "403", "permission_denied", "invalid_api_key"]
            if any(qe in error_str for qe in quota_errors):
                logger.info(f"{provider} exhausted/invalid. Blacklisting for 10 mins. Trying next provider...")
                EXHAUSTED_PROVIDERS[provider] = time.time() + 600 # 10 min blacklist
                continue
            else:
                # Still try the next provider for any other runtime error
                continue

    logger.error("All LLM providers failed or are exhausted.")
    raise last_error

def extract_json(content: str) -> dict:
    """Robustly extracts JSON from a string that might contain chatty text."""
    import re
    if not content:
        raise ValueError("Empty content received")
        
    content = content.strip()
    
    # Remove markdown code blocks if present
    content = re.sub(r'```(?:json)?', '', content)
    content = content.replace('```', '').strip()
    
    # Try finding the first '{' and last '}'
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            json_str = content[start:end+1]
            return json.loads(json_str)
        else:
            # Fallback to direct load if no braces
            return json.loads(content)
    except Exception as e:
        logger.error(f"Failed to extract JSON. Content start: {content[:100]}")
        raise ValueError(f"Could not extract valid JSON from LLM response: {str(e)}")
