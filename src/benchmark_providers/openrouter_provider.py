"""
OpenRouter provider for benchmarking LLM error detection.

This module provides an OpenRouter-based implementation of the BaseLLMProvider
for detecting errors in mathematical formulas.

OpenRouter provides access to multiple LLM providers through a unified API.
"""

import os
import requests
from typing import Optional
from dotenv import load_dotenv

from .base import BaseLLMProvider


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter implementation of the LLM provider.

    Uses OpenRouter's API to access multiple LLM providers for error detection.
    OpenRouter supports models from OpenAI, Anthropic, Google, Meta, and more.
    """

    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        app_name: str = "El-Agente-Math",
        app_url: Optional[str] = None
    ):
        """
        Initialize the OpenRouter provider.

        Args:
            model: Model identifier (e.g., "anthropic/claude-3.5-sonnet", "google/gemini-pro")
            api_key: OpenRouter API key (if None, loads from OPENROUTER_API_KEY env var)
            temperature: Temperature for generation (0.0 to 1.0)
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
            app_name: Application name for OpenRouter tracking (optional)
            app_url: Application URL for OpenRouter tracking (optional)
        """
        super().__init__(model, max_retries, retry_delay)

        # Load environment variables
        load_dotenv()

        # Get API key
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter. Get your key at https://openrouter.ai/keys"
            )

        # Model configuration
        self.temperature = temperature
        self.app_name = app_name
        self.app_url = app_url

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call OpenRouter's API to get a response.

        Args:
            system_prompt: System message for the LLM
            user_prompt: User message for the LLM

        Returns:
            Raw text response from the LLM

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Add optional tracking headers
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        # Prepare request body
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature
        }

        # Make API request
        response = requests.post(
            self.OPENROUTER_API_URL,
            headers=headers,
            json=data,
            timeout=60
        )

        # Check for errors
        if response.status_code != 200:
            error_message = f"OpenRouter API error (status {response.status_code})"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_message += f": {error_data['error']}"
            except:
                error_message += f": {response.text}"
            raise Exception(error_message)

        # Parse response
        response_data = response.json()

        # Extract content
        if "choices" not in response_data or len(response_data["choices"]) == 0:
            raise Exception("OpenRouter API returned no choices")

        content = response_data["choices"][0]["message"]["content"]
        return content.strip()
