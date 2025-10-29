"""
OpenAI provider for benchmarking LLM error detection.

This module provides an OpenAI-based implementation of the BaseLLMProvider
for detecting errors in mathematical formulas.
"""

import os
from typing import Optional
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from dotenv import load_dotenv

from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI implementation of the LLM provider.

    Uses OpenAI's API for error detection in mathematical formulas.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize the OpenAI provider.

        Args:
            model: OpenAI model name (e.g., "gpt-5", "gpt-4o", "gpt-4o-mini")
            api_key: OpenAI API key (if None, loads from OPENAI_API_KEY env var)
            temperature: Temperature for generation (ignored for GPT-5 models)
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
        """
        super().__init__(model, max_retries, retry_delay)

        # Load environment variables
        load_dotenv()

        # Get API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Model configuration
        self.temperature = temperature

        # Check if model is GPT-5 family (doesn't support temperature)
        self._is_gpt5 = self._is_gpt5_model(model)

    @staticmethod
    def _is_gpt5_model(model_name: str) -> bool:
        """
        Check if the model is a GPT-5 family model.

        GPT-5 models do not support the temperature parameter.

        Args:
            model_name: Name of the model

        Returns:
            True if model is GPT-5 family, False otherwise
        """
        return model_name.lower().startswith("gpt-5") or model_name.lower().startswith("gpt5")

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call OpenAI's API to get a response.

        Args:
            system_prompt: System message for the LLM
            user_prompt: User message for the LLM

        Returns:
            Raw text response from the LLM

        Raises:
            APIError: If API call fails
            RateLimitError: If rate limit is hit
            APIConnectionError: If connection fails
        """
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Prepare API parameters
        api_params = {
            "model": self.model,
            "messages": messages,
        }

        # Only add temperature for non-GPT-5 models
        if not self._is_gpt5:
            api_params["temperature"] = self.temperature

        # Call API
        response = self.client.chat.completions.create(**api_params)
        return response.choices[0].message.content.strip()
