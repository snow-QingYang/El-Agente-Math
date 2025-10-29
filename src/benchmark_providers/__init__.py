"""
LLM provider factory for benchmarking.

This module provides a factory function to create the appropriate LLM provider
based on the model string format.

Model string format:
    - "openai/gpt-5" → OpenAI provider with model "gpt-5"
    - "openrouter/anthropic/claude-3.5-sonnet" → OpenRouter provider with model "anthropic/claude-3.5-sonnet"
    - "gpt-5" → Default to OpenAI provider with model "gpt-5" (backward compatibility)
"""

from typing import Optional
from .base import BaseLLMProvider
from .openai_provider import OpenAIProvider
from .openrouter_provider import OpenRouterProvider


def get_provider(
    model: str,
    temperature: float = 0.7,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Factory function to create an LLM provider based on model string.

    Args:
        model: Model string in format "provider/model" or just "model"
               Examples:
                 - "openai/gpt-5"
                 - "openai/gpt-4o"
                 - "openrouter/anthropic/claude-3.5-sonnet"
                 - "openrouter/google/gemini-pro"
                 - "gpt-5" (defaults to openai/gpt-5)
        temperature: Temperature for generation (0.0 to 1.0)
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (seconds)
        api_key: API key for the provider (optional, can use env vars)
        **kwargs: Additional provider-specific arguments

    Returns:
        Instance of BaseLLMProvider (OpenAIProvider or OpenRouterProvider)

    Raises:
        ValueError: If provider is unknown or model format is invalid

    Examples:
        >>> provider = get_provider("openai/gpt-5")
        >>> provider = get_provider("openrouter/anthropic/claude-3.5-sonnet")
        >>> provider = get_provider("gpt-5")  # defaults to openai/gpt-5
    """
    # Parse model string
    provider_name, model_name = _parse_model_string(model)

    # Create appropriate provider
    if provider_name == "openai":
        return OpenAIProvider(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    elif provider_name == "openrouter":
        return OpenRouterProvider(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Supported providers: openai, openrouter"
        )


def _parse_model_string(model: str) -> tuple[str, str]:
    """
    Parse model string to extract provider and model name.

    Format: "provider/model" or "provider/namespace/model"
    - "openai/gpt-5" → ("openai", "gpt-5")
    - "openrouter/anthropic/claude-3.5-sonnet" → ("openrouter", "anthropic/claude-3.5-sonnet")
    - "gpt-5" → ("openai", "gpt-5") [default to OpenAI]

    Args:
        model: Model string

    Returns:
        Tuple of (provider_name, model_name)

    Raises:
        ValueError: If model string format is invalid
    """
    if not model or not isinstance(model, str):
        raise ValueError("Model string must be a non-empty string")

    # Check if model has provider prefix
    if "/" not in model:
        # No provider specified, default to OpenAI for backward compatibility
        return "openai", model

    # Split by first "/"
    parts = model.split("/", 1)

    if len(parts) != 2:
        raise ValueError(
            f"Invalid model string format: '{model}'. "
            f"Expected format: 'provider/model' or just 'model'"
        )

    provider_name = parts[0].lower().strip()
    model_name = parts[1].strip()

    if not provider_name or not model_name:
        raise ValueError(
            f"Invalid model string format: '{model}'. "
            f"Provider and model name cannot be empty"
        )

    return provider_name, model_name


__all__ = [
    "get_provider",
    "BaseLLMProvider",
    "OpenAIProvider",
    "OpenRouterProvider"
]
