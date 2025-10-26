"""
Extract mathematical formulas from LaTeX files with surrounding context.

This module provides functionality to:
- Parse consolidated LaTeX files
- Extract math formulas (inline and display math)
- Capture context before and after each formula
- Handle various LaTeX math environments
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re

from pylatexenc.latexwalker import LatexWalker, LatexWalkerError
from pylatexenc.latexnodes.nodes import LatexEnvironmentNode, LatexMathNode
from pylatexenc.latexnodes import parsers as latex_parsers


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


MATH_ENVIRONMENTS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "eqnarray",
    "eqnarray*",
    "aligned",
    "alignedat",
}


def extract_formulas(tex_file: Path, context_length: int = 200) -> List[MathFormula]:
    """
    Extract all mathematical formulas from a LaTeX file with context.

    Args:
        tex_file: Path to the consolidated .tex file
        context_length: Number of characters to capture before/after each formula

    Returns:
        List of MathFormula objects containing formulas and their contexts
    """
    tex_file = Path(tex_file)
    if not tex_file.exists():
        raise FileNotFoundError(f"LaTeX file not found: {tex_file}")

    tex_content = tex_file.read_text(encoding="utf-8", errors="ignore")

    inline_entries = extract_inline_math(tex_content)
    display_entries = extract_display_math(tex_content)

    formulas: List[MathFormula] = []
    seen_ranges: set[Tuple[int, int]] = set()

    for entry in inline_entries + display_entries:
        start = entry["start"]
        end = entry["end"]
        if (start, end) in seen_ranges:
            continue

        seen_ranges.add((start, end))
        before, after = get_context(
            tex_content, start, length=context_length, span=end - start
        )

        formulas.append(
            MathFormula(
                formula=entry["formula"],
                formula_type=entry["environment"],
                context_before=before,
                context_after=after,
                line_number=entry["line_number"],
                raw_latex=entry["raw_latex"],
            )
        )

    formulas.sort(key=lambda f: f.line_number)
    return formulas


def extract_inline_math(tex_content: str) -> List[Dict[str, Any]]:
    """
    Extract inline math formulas ($...$, \\(...\\)).

    Args:
        tex_content: LaTeX content as string

    Returns:
        List of dictionaries with formula info and positions
    """
    entries: List[Dict[str, Any]] = []
    parse_result = _safe_parse(tex_content)

    if parse_result:
        _, nodelist = parse_result
        for node in _iter_nodes(nodelist):
            if isinstance(node, LatexMathNode) and (node.displaytype or "inline") == "inline":
                start = node.pos
                end = node.pos + node.len
                raw = tex_content[start:end]
                formula = _strip_math_delimiters(raw).strip()
                if not formula:
                    continue
                line_number = _estimate_line_number(tex_content, start)
                entries.append(
                    {
                        "formula": formula,
                        "raw_latex": raw,
                        "start": start,
                        "end": end,
                        "environment": "inline",
                        "category": "inline",
                        "line_number": line_number,
                    }
                )
        return entries

    # Fallback to simple regex extraction
    entries.extend(_extract_inline_with_regex(tex_content))
    return entries


def extract_display_math(tex_content: str) -> List[Dict[str, Any]]:
    """
    Extract display math formulas ($$...$$, \\[...\\], equation, align, etc.).

    Args:
        tex_content: LaTeX content as string

    Returns:
        List of dictionaries with formula info and positions
    """
    entries: List[Dict[str, Any]] = []
    parse_result = _safe_parse(tex_content)

    if parse_result:
        _, nodelist = parse_result
        for node in _iter_nodes(nodelist):
            if isinstance(node, LatexMathNode) and (node.displaytype or "inline") == "display":
                start = node.pos
                end = node.pos + node.len
                raw = tex_content[start:end]
                formula = _strip_math_delimiters(raw).strip()
                if not formula:
                    continue
                line_number = _estimate_line_number(tex_content, start)
                entries.append(
                    {
                        "formula": formula,
                        "raw_latex": raw,
                        "start": start,
                        "end": end,
                        "environment": "display",
                        "category": "display",
                        "line_number": line_number,
                    }
                )
            elif isinstance(node, LatexEnvironmentNode):
                env_name = node.environmentname or ""
                if env_name not in MATH_ENVIRONMENTS:
                    continue
                start = node.pos
                end = node.pos + node.len
                raw = tex_content[start:end]
                formula = _strip_math_delimiters(raw).strip()
                if not formula:
                    continue
                line_number = _estimate_line_number(tex_content, start)
                entries.append(
                    {
                        "formula": formula,
                        "raw_latex": raw,
                        "start": start,
                        "end": end,
                        "environment": env_name,
                        "category": "display",
                        "line_number": line_number,
                    }
                )
        return entries

    entries.extend(_extract_display_with_regex(tex_content))
    return entries


def get_context(
    tex_content: str, position: int, length: int = 200, span: int = 0
) -> tuple[str, str]:
    """
    Extract context around a specific position in the text.

    Args:
        tex_content: Full LaTeX content
        position: Character offset where the formula begins
        length: Number of characters to extract before/after
        span: Length of the formula in characters (used to skip the formula itself)

    Returns:
        Tuple of (context_before, context_after)
    """
    before_start = max(0, position - length)
    before = tex_content[before_start:position]

    after_start = position + max(span, 0)
    after_end = min(len(tex_content), after_start + length)
    after = tex_content[after_start:after_end]

    before = _tail_of_last_line(before)
    after = _head_of_first_line(after)

    return _clean_context(before), _clean_context(after)


def formulas_to_dict(formulas: List[MathFormula]) -> List[Dict[str, Any]]:
    """
    Convert list of MathFormula objects to dictionaries for serialization.

    Args:
        formulas: List of MathFormula objects

    Returns:
        List of dictionaries suitable for JSON serialization
    """
    result: List[Dict[str, Any]] = []
    for formula in formulas:
        result.append(
            {
                "formula": formula.formula,
                "formula_type": formula.formula_type,
                "context_before": formula.context_before,
                "context_after": formula.context_after,
                "line_number": formula.line_number,
                "raw_latex": formula.raw_latex,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_parse(tex_content: str) -> Optional[Tuple[LatexWalker, Iterable]]:
    try:
        walker = LatexWalker(tex_content)
        parser = latex_parsers.LatexGeneralNodesParser()
        nodes, _ = walker.parse_content(parser)
        if nodes is None:
            return walker, []
        top_level = nodes.nodelist if hasattr(nodes, "nodelist") else [nodes]
        return walker, top_level
    except (LatexWalkerError, ValueError):
        return None


def _iter_nodes(nodes: Iterable) -> Iterable:
    if not nodes:
        return

    for node in nodes:
        yield node

        child_list = getattr(node, "nodelist", None)
        if child_list:
            yield from _iter_nodes(child_list)

        nodeargd = getattr(node, "nodeargd", None)
        if nodeargd and getattr(nodeargd, "argnlist", None):
            for arg in nodeargd.argnlist:
                if hasattr(arg, "nodelist") and arg.nodelist:
                    yield from _iter_nodes(arg.nodelist)


def _strip_math_delimiters(raw_latex: str) -> str:
    stripped = raw_latex.strip()

    if stripped.startswith(r"\(") and stripped.endswith(r"\)"):
        return stripped[2:-2]

    if stripped.startswith(r"\[") and stripped.endswith(r"\]"):
        return stripped[2:-2]

    if stripped.startswith("$$") and stripped.endswith("$$"):
        return stripped[2:-2]

    if stripped.startswith("$") and stripped.endswith("$"):
        return stripped[1:-1]

    env_match = re.match(
        r"\\begin\{([^\}]+)\}(.*)\\end\{\1\}\s*$", stripped, flags=re.DOTALL
    )
    if env_match:
        return env_match.group(2).strip()

    return stripped


def _clean_context(text: str) -> str:
    text = _strip_boundary_tokens(text)
    return re.sub(r"\s+", " ", text).strip()


def _estimate_line_number(tex_content: str, position: int) -> int:
    return tex_content.count("\n", 0, position) + 1


def _tail_of_last_line(fragment: str) -> str:
    if not fragment:
        return ""
    lines = fragment.splitlines()
    return lines[-1] if lines else fragment


def _head_of_first_line(fragment: str) -> str:
    if not fragment:
        return ""
    lines = fragment.splitlines()
    return lines[0] if lines else fragment


_LEFT_BOUNDARY = ("$", "$$", r"\[")
_RIGHT_BOUNDARY = ("$", "$$", r"\]",)


def _strip_boundary_tokens(text: str) -> str:
    text = text.lstrip()
    for token in _LEFT_BOUNDARY:
        if text.startswith(token):
            text = text[len(token):]
            break

    text = text.rstrip()
    for token in _RIGHT_BOUNDARY:
        if text.endswith(token):
            text = text[: -len(token)]
            break

    return text


def _extract_inline_with_regex(tex_content: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    patterns = [
        re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$", re.DOTALL),
        re.compile(r"\\\((.+?)\\\)", re.DOTALL),
    ]

    for pattern in patterns:
        for match in pattern.finditer(tex_content):
            raw = match.group(0)
            formula = match.group(1).strip()
            if not formula:
                continue
            start, end = match.span()
            entries.append(
                {
                    "formula": formula,
                    "raw_latex": raw,
                    "start": start,
                    "end": end,
                    "environment": "inline",
                    "category": "inline",
                    "line_number": _estimate_line_number(tex_content, start),
                }
            )

    return entries


def _extract_display_with_regex(tex_content: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    display_patterns = [
        (re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL), "display"),
        (re.compile(r"\\\[(.+?)\\\]", re.DOTALL), "display"),
    ]

    for pattern, env in display_patterns:
        for match in pattern.finditer(tex_content):
            raw = match.group(0)
            formula = match.group(1).strip()
            if not formula:
                continue
            start, end = match.span()
            entries.append(
                {
                    "formula": formula,
                    "raw_latex": raw,
                    "start": start,
                    "end": end,
                    "environment": env,
                    "category": "display",
                    "line_number": _estimate_line_number(tex_content, start),
                }
            )

    if MATH_ENVIRONMENTS:
        env_group = "|".join(re.escape(env) for env in sorted(MATH_ENVIRONMENTS))
        env_pattern = re.compile(
            rf"\\begin\{{({env_group})\}}(.*?)\\end\{{\1\}}", re.DOTALL
        )

        for match in env_pattern.finditer(tex_content):
            raw = match.group(0)
            formula = match.group(2).strip()
            if not formula:
                continue
            start, end = match.span()
            env_name = match.group(1)
            entries.append(
                {
                    "formula": formula,
                    "raw_latex": raw,
                    "start": start,
                    "end": end,
                    "environment": env_name,
                    "category": "display",
                    "line_number": _estimate_line_number(tex_content, start),
                }
            )

    return entries
