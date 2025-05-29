# This file is deprecated. All LLM code is now in llm.py.

import time
from typing import List, Optional, Dict, Any

import litellm
from content_utils.llm_config import LLM_ENDPOINTS, LLMModel

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

# --- Structured Chat Completion ---
@retry_on_exception(max_retries=3, delay=1.0)
def prompt_completion_chat_structured(
    messages: List[Dict[str, str]],
    response_format,
    model_id: LLMModel = LLMModel.BEST_WRITING_MODEL,
    max_tokens: int = 256,
    temperature: float = 1.0,
    n: int = 1,
    **kwargs
):
    """
    Call a chat-style LLM using litellm with structured output (Pydantic models).
    
    Args:
        messages: List of message dictionaries
        response_format: Pydantic model class for structured output
        model_id: LLM model to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        n: Number of responses to generate
        **kwargs: Additional parameters for litellm
        
    Returns:
        Pydantic model instance(s) or fallback to parsed JSON
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
        response_format=response_format,
        **kwargs
    )
    
    # Handle structured output
    if n == 1:
        # Try to get parsed object first, fallback to manual parsing
        if hasattr(response.choices[0].message, 'parsed') and response.choices[0].message.parsed is not None:
            return response.choices[0].message.parsed
        else:
            # Manually parse the JSON content - raise exception if it fails
            content = response.choices[0].message.content.strip()
            try:
                return response_format.parse_raw(content)
            except Exception as e:
                raise ValueError(f"Failed to parse structured output. Content was: {content}") from e
    else:
        results = []
        for i, choice in enumerate(response.choices):
            if hasattr(choice.message, 'parsed') and choice.message.parsed is not None:
                results.append(choice.message.parsed)
            else:
                # Manually parse the JSON content - raise exception if it fails
                content = choice.message.content.strip()
                try:
                    results.append(response_format.parse_raw(content))
                except Exception as e:
                    raise ValueError(f"Failed to parse structured output for choice {i}. Content was: {content}") from e
        return results

# --- Convenience: Return single string if n==1 ---
def prompt_completion_one(*args, **kwargs) -> str:
    results = prompt_completion(*args, **kwargs)
    return results[0] if results else ""

def prompt_completion_chat_one(*args, **kwargs) -> str:
    results = prompt_completion_chat(*args, **kwargs)
    return results[0] if results else ""
