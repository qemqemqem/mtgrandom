from enum import Enum

class LLMModel(Enum):
    """
    Enum for internal LLM identifiers. Update this as you add more model roles.
    """
    BEST_WRITING_MODEL = "BEST_WRITING_MODEL"
    CHEAP_WRITING_MODEL = "CHEAP_WRITING_MODEL"
    BEST_THINKING_MODEL = "BEST_THINKING_MODEL"
    CHEAP_THINKING_MODEL = "CHEAP_THINKING_MODEL"

# Updated May 28, 2025
LLM_ENDPOINTS = {
    LLMModel.BEST_WRITING_MODEL: {
        "model": "anthropic/claude-sonnet-4-20250514",
        "supports_thinking": True,
        "default_thinking": {
            "reasoning_effort": "medium"
        },
        "thinking_min_tokens": 8000,  # Minimum tokens required when thinking is enabled
        "thinking_token_buffer": 5000,  # Extra tokens to add for thinking budget
    },
    
    LLMModel.CHEAP_WRITING_MODEL: {
        "model": "deepseek-ai/deepseek-v3",
        "supports_thinking": False,
    },
    
    LLMModel.BEST_THINKING_MODEL: {
        "model": "openai/o3",
        "supports_thinking": True,
        "default_thinking": {
            "reasoning_effort": "high"
        },
        "thinking_min_tokens": 15000,  # Higher for high reasoning effort
        "thinking_token_buffer": 10000,
    },
    
    LLMModel.CHEAP_THINKING_MODEL: {
        "model": "openai/o4-mini",
        "supports_thinking": True,
        "default_thinking": {
            "reasoning_effort": "medium"
        },
        "thinking_min_tokens": 6000,
        "thinking_token_buffer": 4000,
    },
} 