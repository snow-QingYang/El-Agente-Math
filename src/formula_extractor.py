"""
Extract mathematical formulas from LaTeX files with surrounding context.

This module provides functionality to:
- Parse consolidated LaTeX files
- Extract math formulas (inline and display math)
- Capture context before and after each formula
- Handle various LaTeX math environments
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class MathFormula:
    """
    Represents a mathematical formula with its context.

    Attributes:
        formula: The LaTeX formula content (without delimiters)
        formula_type: Type of math environment (inline, equation, align, etc.)
        context_before: Text appearing before the formula
        context_after: Text appearing after the formula
        line_number: Line number in the source file
        raw_latex: Original LaTeX including delimiters
    """
    formula: str
    formula_type: str
    context_before: str
    context_after: str
    line_number: int
    raw_latex: str


def extract_formulas(tex_file: Path, context_length: int = 200) -> List[MathFormula]:
    """
    Extract all mathematical formulas from a LaTeX file with context.

    Args:
        tex_file: Path to the consolidated .tex file
        context_length: Number of characters to capture before/after each formula

    Returns:
        List of MathFormula objects containing formulas and their contexts

    Notes:
        Extracts from various math environments:
        - Inline math: $...$ or \\(...\\)
        - Display math: $$...$$ or \\[...\\]
        - Equation: \\begin{equation}...\\end{equation}
        - Align: \\begin{align}...\\end{align}
        - And other AMS math environments

    Examples:
        >>> formulas = extract_formulas(Path("consolidated.tex"), context_length=150)
        >>> for formula in formulas:
        ...     print(f"Formula: {formula.formula}")
        ...     print(f"Context: {formula.context_before}")
    """
    # TODO: Implement formula extraction logic
    # - Read LaTeX file
    # - Use pylatexenc to parse LaTeX structure
    # - Identify all math environments
    # - Extract formula content
    # - Capture surrounding context
    # - Return list of MathFormula objects
    raise NotImplementedError("Formula extraction not yet implemented")


def extract_inline_math(tex_content: str) -> List[Dict[str, any]]:
    """
    Extract inline math formulas ($...$, \\(...\\)).

    Args:
        tex_content: LaTeX content as string

    Returns:
        List of dictionaries with formula info and positions

    Examples:
        >>> inline = extract_inline_math("The energy $E=mc^2$ is constant.")
    """
    # TODO: Implement inline math extraction
    raise NotImplementedError("Inline math extraction not yet implemented")


def extract_display_math(tex_content: str) -> List[Dict[str, any]]:
    """
    Extract display math formulas ($$...$$, \\[...\\], equation, align, etc.).

    Args:
        tex_content: LaTeX content as string

    Returns:
        List of dictionaries with formula info and positions

    Examples:
        >>> display = extract_display_math(tex_content)
    """
    # TODO: Implement display math extraction
    raise NotImplementedError("Display math extraction not yet implemented")


def get_context(tex_content: str, position: int, length: int = 200) -> tuple[str, str]:
    """
    Extract context around a specific position in the text.

    Args:
        tex_content: Full LaTeX content
        position: Character position of the formula
        length: Number of characters to extract before/after

    Returns:
        Tuple of (context_before, context_after)

    Examples:
        >>> before, after = get_context(content, 500, 150)
    """
    # TODO: Implement context extraction
    # - Handle text boundaries
    # - Clean up LaTeX commands in context (optional)
    # - Preserve readability
    raise NotImplementedError("Context extraction not yet implemented")


def formulas_to_dict(formulas: List[MathFormula]) -> List[Dict[str, any]]:
    """
    Convert list of MathFormula objects to dictionaries for serialization.

    Args:
        formulas: List of MathFormula objects

    Returns:
        List of dictionaries suitable for JSON serialization

    Examples:
        >>> formula_dicts = formulas_to_dict(formulas)
        >>> import json
        >>> json.dump(formula_dicts, open("formulas.json", "w"))
    """
    # TODO: Implement conversion to dict
    raise NotImplementedError("Formula serialization not yet implemented")
