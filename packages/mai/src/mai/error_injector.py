"""
Error injection for mathematical formulas.

This module provides functionality to intentionally inject common mathematical
errors into LaTeX formulas for testing and evaluation purposes.
"""

import re
import random
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


def inject_errors_into_formulas(
    formulas_dict: Dict[str, Dict[str, Any]],
    error_rate: float = 0.5,
    random_seed: Optional[int] = None
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Inject errors into a dictionary of formulas.

    Args:
        formulas_dict: Dictionary mapping labels to formula metadata
        error_rate: Probability (0.0 to 1.0) of injecting error into each formula
        random_seed: Random seed for reproducibility (optional)

    Returns:
        Tuple of (modified_formulas_dict, error_log_dict)
        - modified_formulas_dict: Updated formulas with some containing errors
        - error_log_dict: Documentation of all changes made
    """
    if random_seed is not None:
        random.seed(random_seed)

    modified_formulas = {}
    errors_list = []
    unmodified_list = []

    for label, metadata in formulas_dict.items():
        # Decide whether to inject error
        if random.random() < error_rate:
            # Attempt to inject an error
            result = _inject_single_error(metadata['formula'])

            if result is not None:
                modified_formula, error_type, error_desc = result

                # Update formula metadata
                modified_metadata = metadata.copy()
                modified_metadata['formula'] = modified_formula

                # Also update raw_latex if it contains the formula
                if 'raw_latex' in metadata:
                    # Try to replace the formula in raw_latex
                    modified_metadata['raw_latex'] = metadata['raw_latex'].replace(
                        metadata['formula'],
                        modified_formula
                    )

                modified_formulas[label] = modified_metadata

                # Log the error
                errors_list.append({
                    "label": label,
                    "original_formula": metadata['formula'],
                    "modified_formula": modified_formula,
                    "error_type": error_type,
                    "error_description": error_desc,
                    "line_number": metadata.get('line_number', None),
                    "formula_type": metadata.get('formula_type', None)
                })
            else:
                # Error injection failed, keep original
                modified_formulas[label] = metadata.copy()
                unmodified_list.append(label)
        else:
            # Don't inject error
            modified_formulas[label] = metadata.copy()
            unmodified_list.append(label)

    # Create error log
    error_log = {
        "metadata": {
            "error_rate": error_rate,
            "random_seed": random_seed,
            "total_formulas_processed": len(formulas_dict),
            "formulas_modified": len(errors_list),
            "formulas_unmodified": len(unmodified_list),
            "timestamp": datetime.now().isoformat()
        },
        "errors": errors_list,
        "unmodified": unmodified_list
    }

    return modified_formulas, error_log


def _inject_single_error(formula: str) -> Optional[Tuple[str, str, str]]:
    """
    Inject a single error into a formula.

    Args:
        formula: LaTeX formula string

    Returns:
        Tuple of (modified_formula, error_type, description) or None if injection failed
    """
    # List of error injection functions
    error_injectors = [
        _flip_sign,
        _swap_exponent_order,
        _swap_operator,
        _change_index,
        _flip_inequality,
        _transpose_error,
        _invert_fraction,
        _swap_sum_product,
        _remove_parentheses,
        _swap_function
    ]

    # Shuffle to try errors in random order
    random.shuffle(error_injectors)

    # Try each error injector until one succeeds
    for injector in error_injectors:
        result = injector(formula)
        if result is not None:
            return result

    # No error could be injected
    return None


def _flip_sign(formula: str) -> Optional[Tuple[str, str, str]]:
    """Flip + to - or vice versa."""
    # Find all + or - that are operators (not part of numbers like -5)
    # Look for patterns like: X + Y or X - Y (with spaces or not)
    patterns = [
        (r'(\S)\s*\+\s*(\S)', r'\1 - \2', '+', '-'),
        (r'(\S)\s*-\s*(\S)', r'\1 + \2', '-', '+'),
    ]

    for pattern, replacement, old_op, new_op in patterns:
        matches = list(re.finditer(pattern, formula))
        if matches:
            # Pick a random match
            match = random.choice(matches)
            modified = formula[:match.start()] + re.sub(pattern, replacement, formula[match.start():match.end()]) + formula[match.end():]
            return modified, "sign_flip", f"Changed '{old_op}' to '{new_op}'"

    return None


def _swap_exponent_order(formula: str) -> Optional[Tuple[str, str, str]]:
    """Change exponent order, e.g., E(X)^2 to E(X^2) or (a+b)^2 to a+b^2."""
    # Pattern 1: Function with exponent -> move exponent inside
    # E.g., \E(X)^2 -> \E(X^2)
    pattern1 = r'(\\[a-zA-Z]+)\(([^()]+)\)\^(\{[^}]+\}|\d)'
    matches1 = list(re.finditer(pattern1, formula))
    if matches1:
        match = random.choice(matches1)
        func = match.group(1)
        inner = match.group(2)
        exp = match.group(3)
        modified = formula[:match.start()] + f"{func}(({inner})^{exp})" + formula[match.end():]
        return modified, "exponent_order", f"Moved exponent inside {func}()"

    # Pattern 2: Parenthesized expression with exponent -> remove outer parens
    # E.g., (a+b)^2 -> a+b^2 (incorrect!)
    pattern2 = r'\(([^()]+)\)\^(\{[^}]+\}|\d)'
    matches2 = list(re.finditer(pattern2, formula))
    if matches2:
        match = random.choice(matches2)
        inner = match.group(1)
        exp = match.group(2)
        modified = formula[:match.start()] + f"{inner}^{exp}" + formula[match.end():]
        return modified, "exponent_order", "Removed parentheses from exponentiated expression"

    return None


def _swap_operator(formula: str) -> Optional[Tuple[str, str, str]]:
    """Swap operators like + to ×, × to /, etc."""
    swaps = [
        (r'\+', r'\\times', '+', '×'),
        (r'\\times', r'/', '×', '/'),
        (r'\\cdot', r'+', '·', '+'),
    ]

    for pattern, replacement, old_name, new_name in swaps:
        matches = list(re.finditer(pattern, formula))
        if matches:
            match = random.choice(matches)
            modified = formula[:match.start()] + replacement + formula[match.end():]
            return modified, "operator_swap", f"Changed '{old_name}' to '{new_name}'"

    return None


def _change_index(formula: str) -> Optional[Tuple[str, str, str]]:
    """Change subscript indices, e.g., x_i to x_j."""
    # Pattern: variable with subscript i, j, k, l, m, n
    indices = ['i', 'j', 'k', 'l', 'm', 'n']
    patterns = [
        (rf'_({idx})\b', random.choice([x for x in indices if x != idx]), idx)
        for idx in indices
    ]

    for pattern, new_idx, old_idx in patterns:
        matches = list(re.finditer(pattern, formula))
        if matches:
            match = random.choice(matches)
            modified = formula[:match.start()] + f"_{new_idx}" + formula[match.end():]
            return modified, "index_change", f"Changed subscript '{old_idx}' to '{new_idx}'"

    return None


def _flip_inequality(formula: str) -> Optional[Tuple[str, str, str]]:
    """Flip inequality direction."""
    flips = [
        (r'<', r'>', '<', '>'),
        (r'>', r'<', '>', '<'),
        (r'\\leq', r'\\geq', '≤', '≥'),
        (r'\\geq', r'\\leq', '≥', '≤'),
        (r'\\le', r'\\ge', '≤', '≥'),
        (r'\\ge', r'\\le', '≥', '≤'),
    ]

    for pattern, replacement, old_name, new_name in flips:
        matches = list(re.finditer(pattern, formula))
        if matches:
            match = random.choice(matches)
            modified = formula[:match.start()] + replacement + formula[match.end():]
            return modified, "inequality_flip", f"Changed '{old_name}' to '{new_name}'"

    return None


def _transpose_error(formula: str) -> Optional[Tuple[str, str, str]]:
    """Add or remove transpose markers."""
    # Pattern 1: Add transpose where there isn't one
    # Find matrix/vector variables (capital letters or \\v-prefixed)
    pattern_add = r'([A-Z]|\\v[a-z]+)(?!\^[T{])'
    matches_add = list(re.finditer(pattern_add, formula))
    if matches_add and random.random() < 0.5:
        match = random.choice(matches_add)
        var = match.group(0)
        modified = formula[:match.end()] + '^T' + formula[match.end():]
        return modified, "transpose_error", f"Added transpose to {var}"

    # Pattern 2: Remove existing transpose
    pattern_remove = r'([A-Z]|\\v[a-z]+)\^(T|\\top)'
    matches_remove = list(re.finditer(pattern_remove, formula))
    if matches_remove:
        match = random.choice(matches_remove)
        var = match.group(1)
        modified = formula[:match.start()] + var + formula[match.end():]
        return modified, "transpose_error", f"Removed transpose from {var}"

    return None


def _invert_fraction(formula: str) -> Optional[Tuple[str, str, str]]:
    """Invert a fraction: \\frac{a}{b} to \\frac{b}{a}."""
    pattern = r'\\frac\{([^{}]+)\}\{([^{}]+)\}'
    matches = list(re.finditer(pattern, formula))
    if matches:
        match = random.choice(matches)
        num = match.group(1)
        den = match.group(2)
        modified = formula[:match.start()] + f"\\frac{{{den}}}{{{num}}}" + formula[match.end():]
        return modified, "fraction_inversion", f"Inverted fraction (swapped numerator and denominator)"

    return None


def _swap_sum_product(formula: str) -> Optional[Tuple[str, str, str]]:
    """Swap \\sum with \\prod or vice versa."""
    swaps = [
        (r'\\sum', r'\\prod', '∑', '∏'),
        (r'\\prod', r'\\sum', '∏', '∑'),
    ]

    for pattern, replacement, old_name, new_name in swaps:
        matches = list(re.finditer(pattern, formula))
        if matches:
            match = random.choice(matches)
            modified = formula[:match.start()] + replacement + formula[match.end():]
            return modified, "sum_product_swap", f"Changed '{old_name}' to '{new_name}'"

    return None


def _remove_parentheses(formula: str) -> Optional[Tuple[str, str, str]]:
    """Remove parentheses from an expression that needs them."""
    # Look for patterns like (a+b)*c and change to a+b*c
    pattern = r'\(([^()]+[+\-][^()]+)\)\s*([\\times\*]|\\cdot)'
    matches = list(re.finditer(pattern, formula))
    if matches:
        match = random.choice(matches)
        inner = match.group(1)
        op = match.group(2)
        modified = formula[:match.start()] + f"{inner}{op}" + formula[match.end():]
        return modified, "missing_parentheses", "Removed parentheses (operator precedence error)"

    return None


def _swap_function(formula: str) -> Optional[Tuple[str, str, str]]:
    """Swap similar functions like sin/cos, log/ln."""
    swaps = [
        (r'\\sin', r'\\cos', 'sin', 'cos'),
        (r'\\cos', r'\\sin', 'cos', 'sin'),
        (r'\\log', r'\\ln', 'log', 'ln'),
        (r'\\ln', r'\\log', 'ln', 'log'),
        (r'\\max', r'\\min', 'max', 'min'),
        (r'\\min', r'\\max', 'min', 'max'),
    ]

    for pattern, replacement, old_name, new_name in swaps:
        matches = list(re.finditer(pattern, formula))
        if matches:
            match = random.choice(matches)
            modified = formula[:match.start()] + replacement + formula[match.end():]
            return modified, "function_swap", f"Changed '{old_name}' to '{new_name}'"

    return None
