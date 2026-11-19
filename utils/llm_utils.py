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
        # llama-3.3-70b-versatile was retired by Groq; override with GROQ_MODEL in .env if needed
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return ChatGroq(model_name=model_name, temperature=temperature, groq_api_key=api_key, max_retries=0)
    
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: return None
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=temperature, openai_api_key=api_key, max_retries=0)
    
    return None

def _retry_after_seconds(error_str: str):
    """Parse 'try again in 7.33s' / 'retry in 52.4s' / '1m30s' style hints from provider errors."""
    import re
    m = re.search(r"(?:try again|retry) in (?:(\d+)m)?\s*([\d.]+)\s*s", error_str)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2))
    m = re.search(r"(?:try again|retry) in (\d+)m(?:in)?", error_str)
    if m:
        return int(m.group(1)) * 60.0
    return None


def invoke_with_failover(prompt, input_vars, temperature=0):
    """
    Invokes LLM with failover across providers: Groq -> Gemini -> OpenAI.
    Groq is the primary provider.
    """
    providers = ["groq", "gemini", "openai"]
    last_error = None

    for provider in providers:
        llm = get_llm_instance(provider, temperature)
        if not llm:
            reason = "temporarily blacklisted after an error" if provider in EXHAUSTED_PROVIDERS else "API key not set"
            logger.warning(f"Skipping {provider} ({reason}).")
            continue

        for attempt in range(3):
            try:
                logger.info(f"Attempting LLM invocation with {provider}...")
                response = (prompt | llm).invoke(input_vars)
                logger.info(f"Success with {provider}!")
                return response
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                logger.warning(f"Error with {provider}: {e}")

                # Short per-minute rate limit (e.g. Groq free tier "try again in 7.3s"): wait and retry once
                wait = _retry_after_seconds(error_str)
                if attempt < 2 and wait is not None and wait <= 20 and "rate" in error_str and "per day" not in error_str:
                    logger.info(f"{provider} rate-limited; retrying in {wait:.1f}s...")
                    time.sleep(wait + 0.5)
                    continue

                permanent = ["leaked", "invalid_api_key", "no credits", "insufficient_quota", "permission_denied", "403", "401", "model_not_found"]
                transient = ["429", "quota", "resource_exhausted", "rate_limit", "rate limit"]
                if any(p in error_str for p in permanent):
                    logger.info(f"{provider} unusable (key/billing/model). Blacklisting for 60 mins.")
                    EXHAUSTED_PROVIDERS[provider] = time.time() + 3600
                elif any(t in error_str for t in transient):
                    cooldown = max(60, wait or 0)
                    logger.info(f"{provider} rate-limited. Blacklisting for {cooldown:.0f}s.")
                    EXHAUSTED_PROVIDERS[provider] = time.time() + cooldown
                break  # move on to the next provider

    logger.error("All LLM providers failed or are exhausted.")
    raise last_error or RuntimeError("No LLM provider available (all skipped: missing keys or temporarily blacklisted).")

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
