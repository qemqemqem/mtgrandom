# This file is deprecated. All LLM code is now in llm.py.

import time
from typing import List, Optional, Dict, Any

import litellm
from llm_config import LLM_ENDPOINTS, LLMModel

# --- Retry Decorator ---
def retry_on_exception(max_retries=3, delay=1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator

# --- LLM Model Resolver ---
def resolve_model_config(model_id: LLMModel) -> Dict[str, Any]:
    """Get the full configuration for a model."""
    if model_id not in LLM_ENDPOINTS:
        raise ValueError(f"Unknown LLM model identifier: {model_id}")
    return LLM_ENDPOINTS[model_id]

def resolve_model(model_id: LLMModel) -> str:
    """Get just the litellm model string for backwards compatibility."""
    config = resolve_model_config(model_id)
    return config["model"]

# --- Completion (legacy, non-chat) ---
@retry_on_exception(max_retries=3, delay=1.0)
def prompt_completion(
    prompt: str,
    model_id: LLMModel = LLMModel.BEST_WRITING_MODEL,
    max_tokens: int = 256,
    temperature: float = 1.0,
    n: int = 1,
    stop: Optional[List[str]] = None,
    **kwargs
) -> List[str]:
    """
    Call a completion-style LLM (not chat) using litellm and the configured model.
    """
    config = resolve_model_config(model_id)
    model = config["model"]
    
    # Add thinking parameters if supported
    if config.get("supports_thinking") and "reasoning_effort" not in kwargs:
        if "default_thinking" in config:
            kwargs.update(config["default_thinking"])
    
    response = litellm.completion(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        stop=stop,
        **kwargs
    )
    return [choice["text"].strip() for choice in response["choices"]]

# --- Chat Completion ---
@retry_on_exception(max_retries=3, delay=1.0)
def prompt_completion_chat(
    messages: List[Dict[str, str]],
    model_id: LLMModel = LLMModel.BEST_WRITING_MODEL,
    max_tokens: int = 256,
    temperature: float = 1.0,
    n: int = 1,
    **kwargs
) -> str | List[str]:
    """
    Call a chat-style LLM using litellm and the configured model.
    """
    config = resolve_model_config(model_id)
    model = config["model"]
    
    # Add thinking parameters if supported
    if config.get("supports_thinking") and "reasoning_effort" not in kwargs:
        if "default_thinking" in config:
            kwargs.update(config["default_thinking"])
            # For Anthropic models with thinking enabled, temperature must be 1.0
            # and max_tokens must be high enough to account for thinking budget
            if "anthropic" in model.lower():
                temperature = 1.0
                # Ensure max_tokens is high enough for thinking budget using config values
                min_tokens = config["thinking_min_tokens"]
                token_buffer = config["thinking_token_buffer"]
                if max_tokens < min_tokens:
                    max_tokens = max(max_tokens + token_buffer, min_tokens)
    
    response = litellm.completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        **kwargs
    )
    if n == 1:
        return response["choices"][0]["message"]["content"].strip()
    else:
        return [choice["message"]["content"].strip() for choice in response["choices"]]

# --- Convenience: Return single string if n==1 ---
def prompt_completion_one(*args, **kwargs) -> str:
    results = prompt_completion(*args, **kwargs)
    return results[0] if results else ""

def prompt_completion_chat_one(*args, **kwargs) -> str:
    results = prompt_completion_chat(*args, **kwargs)
    return results[0] if results else ""
