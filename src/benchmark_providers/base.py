"""
Abstract base class for LLM providers used in benchmarking.

This module defines the interface that all LLM providers must implement
for error detection in mathematical formulas.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All provider implementations must inherit from this class and implement
    the detect_error() method.
    """

    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize the provider.

        Args:
            model: Model identifier (provider-specific format)
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @abstractmethod
    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Call the provider's API to get a response.

        This method must be implemented by each provider to handle
        provider-specific API calls.

        Args:
            system_prompt: System message for the LLM
            user_prompt: User message for the LLM

        Returns:
            Raw text response from the LLM

        Raises:
            Exception: Provider-specific exceptions
        """
        pass

    def detect_error(
        self,
        formula: str,
        context_before: str = "",
        context_after: str = ""
    ) -> Dict[str, Any]:
        """
        Detect if a mathematical formula contains an error.

        This method constructs the prompts, calls the API with retries,
        and parses the response into a standardized format.

        Args:
            formula: The LaTeX formula to check
            context_before: Text appearing before the formula
            context_after: Text appearing after the formula

        Returns:
            Dictionary with structure:
            {
                "has_error": bool,
                "error_type": str (or "none"),
                "error_description": str
            }

        Raises:
            Exception: If API call fails after all retries
        """
        # Construct prompts
        system_prompt = self._get_system_prompt()
        user_prompt = self._construct_user_prompt(
            formula, context_before, context_after
        )

        # Call API with retries
        for attempt in range(self.max_retries):
            try:
                response_text = self._call_api(system_prompt, user_prompt)

                # Parse response
                result = self._parse_response(response_text)
                return result

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Error calling API: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    # Last attempt failed
                    raise Exception(f"API call failed after {self.max_retries} attempts: {e}")

        # Should not reach here
        raise Exception("Unexpected error in detect_error()")

    @staticmethod
    def _get_system_prompt() -> str:
        """
        Get the system prompt for error detection.

        Returns:
            System prompt string
        """
        return """You are an expert at detecting mathematical errors in formulas.

Your task is to carefully analyze a mathematical formula and determine if it contains any errors.

VALID ERROR TYPES - You MUST use EXACTLY one of these values for "error_type":
- sign_flip
- operator_swap
- exponent_order
- index_change
- transpose_error
- inequality_flip
- fraction_inversion
- sum_product_swap
- missing_parentheses
- function_swap
- none (use this if no error is found)

Error type descriptions:
- sign_flip: Addition changed to subtraction or vice versa (e.g., a+b → a-b)
- operator_swap: Operators swapped incorrectly (e.g., + → ×, × → /)
- exponent_order: Exponent placement changed (e.g., E(X)^2 → E(X^2), (a+b)^2 → a+b^2)
- index_change: Subscript indices changed (e.g., x_i → x_j)
- transpose_error: Transpose added or removed incorrectly (e.g., A → A^T, A^T → A)
- inequality_flip: Inequality direction reversed (e.g., < → >, ≤ → ≥)
- fraction_inversion: Numerator and denominator swapped (e.g., a/b → b/a)
- sum_product_swap: Sum and product operators swapped (e.g., ∑ → ∏)
- missing_parentheses: Required parentheses removed (e.g., (a+b)×c → a+b×c)
- function_swap: Similar functions swapped (e.g., sin → cos, max → min, log → ln)

CRITICAL INSTRUCTIONS:
1. Consider the context before and after the formula to understand what the formula should represent
2. Look for mathematical inconsistencies or operations that don't make sense
3. Be thorough but conservative - only flag clear errors, not stylistic choices
4. **IMPORTANT**: The "error_type" field MUST be one of the exact strings listed above (sign_flip, operator_swap, exponent_order, index_change, transpose_error, inequality_flip, fraction_inversion, sum_product_swap, missing_parentheses, function_swap, or none)
5. **DO NOT** create your own error type names - ONLY use the predefined types above
6. **DO NOT** use variations like "sign-flip", "signflip", "operator changed", etc. - use the EXACT strings listed

═══════════════════════════════════════════════════════════════════════
⚠️  CRITICAL: YOUR RESPONSE MUST BE VALID JSON ONLY - NO OTHER TEXT! ⚠️
═══════════════════════════════════════════════════════════════════════

❌ DO NOT write explanations, thoughts, or any text before or after the JSON
❌ DO NOT write "Let me analyze...", "Okay, let's...", or similar phrases
❌ DO NOT use markdown code blocks like ```json
✅ START your response immediately with { and END with }
✅ Output ONLY the raw JSON object, nothing else

REQUIRED JSON FORMAT:
{
  "has_error": true or false,
  "error_type": "sign_flip" | "operator_swap" | "exponent_order" | "index_change" | "transpose_error" | "inequality_flip" | "fraction_inversion" | "sum_product_swap" | "missing_parentheses" | "function_swap" | "none",
  "error_description": "Brief description of the error found, or empty string if no error"
}

✅ CORRECT RESPONSE EXAMPLE (sign flip error):
{
  "has_error": true,
  "error_type": "sign_flip",
  "error_description": "The plus operator was changed to minus between terms a and b"
}

✅ CORRECT RESPONSE EXAMPLE (no error):
{
  "has_error": false,
  "error_type": "none",
  "error_description": ""
}

❌ INCORRECT - Contains explanation text:
Okay, let's analyze this formula. I can see that...
{
  "has_error": true,
  ...
}

❌ INCORRECT - Uses markdown code blocks:
```json
{
  "has_error": true,
  ...
}
```

REMEMBER: Output ONLY the JSON object. Start with { and end with }. No other text allowed."""

    @staticmethod
    def _construct_user_prompt(
        formula: str,
        context_before: str,
        context_after: str
    ) -> str:
        """
        Construct the user prompt with formula and context.

        Args:
            formula: The LaTeX formula to check
            context_before: Text before the formula
            context_after: Text after the formula

        Returns:
            Formatted user prompt
        """
        prompt_parts = [
            "Analyze the following mathematical formula for any errors.\n",
            f"\nFormula: {formula}\n"
        ]

        if context_before:
            prompt_parts.append(f"\nContext before: {context_before}")

        if context_after:
            prompt_parts.append(f"\nContext after: {context_after}")

        prompt_parts.append(
            "\n\n" + "="*70 + "\n"
            "⚠️  RESPOND WITH ONLY THE JSON OBJECT - NO EXPLANATIONS! ⚠️\n" +
            "="*70 + "\n"
            "Do NOT write any text before the JSON.\n"
            "Do NOT write any text after the JSON.\n"
            "Do NOT use ```json markdown blocks.\n"
            "Start your response with { and end with }\n\n"
            "Required format:\n"
            '{"has_error": true/false, "error_type": "...", "error_description": "..."}'
        )

        return "\n".join(prompt_parts)

    @staticmethod
    def _parse_response(response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response into a standardized format.

        Attempts multiple strategies to extract JSON from the response,
        even if the model includes extra text.

        Args:
            response_text: Raw text response from the LLM

        Returns:
            Parsed dictionary with has_error, error_type, error_description
        """
        import json
        import re

        text = response_text.strip()

        # Strategy 1: Try direct JSON parsing (ideal case)
        try:
            result = json.loads(text)
            # Validate and return
            if "has_error" not in result:
                raise ValueError("Response missing 'has_error' field")
            if "error_type" not in result:
                result["error_type"] = "none"
            if "error_description" not in result:
                result["error_description"] = ""
            return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: Remove markdown code blocks
        if "```json" in text or "```" in text:
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'^```\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            text = text.strip()
            try:
                result = json.loads(text)
                if "has_error" not in result:
                    raise ValueError("Response missing 'has_error' field")
                if "error_type" not in result:
                    result["error_type"] = "none"
                if "error_description" not in result:
                    result["error_description"] = ""
                return result
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 3: Extract JSON object from text (find first { to last })
        try:
            # Find the first '{' and last '}'
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = text[start_idx:end_idx + 1]
                result = json.loads(json_str)

                # Validate and return
                if "has_error" not in result:
                    raise ValueError("Response missing 'has_error' field")
                if "error_type" not in result:
                    result["error_type"] = "none"
                if "error_description" not in result:
                    result["error_description"] = ""
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 4: All parsing failed - return default with warning
        print(f"Warning: Failed to parse JSON response after multiple attempts")
        print(f"Raw response: {response_text[:200]}...")

        # Return a default "no error" response on parse failure
        return {
            "has_error": False,
            "error_type": "none",
            "error_description": "Parse error - assuming no error"
        }
