"""
LLM client wrapper for explaining mathematical formulas.

This module provides a clean interface to OpenAI's API for generating
formula explanations. It handles:
- Model-specific parameter handling (GPT-5 vs others)
- Error handling and retries
- Rate limit management
"""

from typing import Optional
import os
import time
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from dotenv import load_dotenv


class LLMClient:
    """
    Wrapper around OpenAI API for formula explanation generation.

    Handles model-specific behavior (e.g., GPT-5 doesn't support temperature)
    and provides retry logic for robust API interactions.
    """

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model name (default: "gpt-5")
            api_key: OpenAI API key (if None, loads from environment)
            temperature: Temperature for generation (ignored for GPT-5 models)
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
        """
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
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Check if model is GPT-5 family
        self._is_gpt5 = self._is_gpt5_model(model)

        print(f"LLM Client initialized:")
        print(f"  Model: {self.model}")
        print(f"  Temperature: {self.temperature if not self._is_gpt5 else 'N/A (GPT-5)'}")

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

    def explain_formula(
        self,
        formula: str,
        context_before: str = "",
        context_after: str = "",
        additional_context: str = ""
    ) -> str:
        """
        Generate an explanation for a mathematical formula using LLM.

        Args:
            formula: The LaTeX formula to explain
            context_before: Text appearing before the formula in the paper
            context_after: Text appearing after the formula in the paper
            additional_context: Any additional context (e.g., paper title, section)

        Returns:
            Generated explanation as a string

        Raises:
            APIError: If API call fails after all retries
        """
        # Construct the prompt
        system_prompt = self._get_system_prompt()
        user_prompt = self._construct_user_prompt(
            formula, context_before, context_after, additional_context
        )

        # Prepare API call parameters
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Add model-specific parameters
        api_params = {
            "model": self.model,
            "messages": messages,
        }

        # Only add temperature for non-GPT-5 models
        if not self._is_gpt5:
            api_params["temperature"] = self.temperature

        # Call API with retries
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**api_params)
                explanation = response.choices[0].message.content
                return explanation.strip()

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limit hit. Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise APIError(f"Rate limit exceeded after {self.max_retries} attempts") from e

            except APIConnectionError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay
                    print(f"Connection error. Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise APIError(f"Connection failed after {self.max_retries} attempts") from e

            except APIError as e:
                if attempt < self.max_retries - 1:
                    print(f"API error: {e}. Retrying...")
                    time.sleep(self.retry_delay)
                else:
                    raise

        raise APIError(f"Failed to generate explanation after {self.max_retries} attempts")

    @staticmethod
    def _get_system_prompt() -> str:
        """
        Get the system prompt for the LLM.

        Returns:
            System prompt string
        """
        return """You are an expert mathematical explainer who helps readers understand complex mathematical formulas in scientific papers.

Your task is to provide clear, concise explanations of mathematical formulas, taking into account:
1. The formula itself (in LaTeX notation)
2. The surrounding context from the paper
3. Common mathematical conventions and notation

Your explanations should:
- Be accessible to readers with a graduate-level mathematics background
- Explain the meaning and purpose of the formula
- Define key variables and operations
- Connect the formula to the broader context of the paper
- Be concise (2-4 sentences typically)

Do not include LaTeX formatting in your explanation - use plain text."""

    @staticmethod
    def _construct_user_prompt(
        formula: str,
        context_before: str,
        context_after: str,
        additional_context: str
    ) -> str:
        """
        Construct the user prompt with formula and context.

        Args:
            formula: The LaTeX formula
            context_before: Text before the formula
            context_after: Text after the formula
            additional_context: Additional context

        Returns:
            Formatted user prompt
        """
        prompt_parts = []

        if additional_context:
            prompt_parts.append(f"Paper context: {additional_context}\n")

        prompt_parts.append("Please explain the following mathematical formula:\n")
        prompt_parts.append(f"\nFormula: {formula}\n")

        if context_before:
            prompt_parts.append(f"\nContext before: {context_before}")

        if context_after:
            prompt_parts.append(f"\nContext after: {context_after}")

        prompt_parts.append("\n\nProvide a clear, concise explanation of this formula:")

        return "\n".join(prompt_parts)
