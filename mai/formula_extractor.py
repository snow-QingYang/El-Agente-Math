"""
Extract mathematical formulas from LaTeX files with surrounding context.

This module provides functionality to:
- Parse consolidated LaTeX files
- Extract math formulas (inline and display math)
- Capture context before and after each formula
- Handle various LaTeX math environments
"""

import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pylatexenc.latexnodes import parsers as latex_parsers
from pylatexenc.latexnodes.nodes import LatexEnvironmentNode, LatexMathNode
from pylatexenc.latexwalker import LatexWalker, LatexWalkerError


@dataclass
class MathFormula:
    """
    Represents a mathematical formula with its context.

    Attributes:
        formula: The LaTeX formula content (without delimiters)
        formula_type: Type of math environment (inline, equation, align, etc.)
        line_number: Line number in the source file
        raw_latex: Original LaTeX including delimiters
        start_pos: Character offset where the formula begins
        end_pos: Character offset where the formula ends
        source_path: Path to the source document (if available)
    """
    formula: str
    formula_type: str
    line_number: int
    raw_latex: str
    start_pos: int
    end_pos: int
    source_path: Optional[Path] = None


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

FONT_MACRO_PREFIXES = (
    "mathbb",
    "mathbf",
    "mathrm",
    "mathcal",
    "mathfrak",
    "mathsf",
    "mathit",
    "boldsymbol",
)

COMPLEX_MACROS = {
    "frac",
    "sqrt",
    "sum",
    "prod",
    "int",
    "iint",
    "iiint",
    "oint",
    "lim",
    "sin",
    "cos",
    "tan",
    "cot",
    "log",
    "ln",
    "exp",
    "sup",
    "inf",
    "max",
    "min",
    "det",
    "dim",
    "gcd",
    "operatorname",
}

OPERATOR_MACROS = {
    "times",
    "cdot",
    "pm",
    "mp",
    "leq",
    "geq",
    "neq",
    "approx",
    "sim",
    "to",
    "leftrightarrow",
    "rightarrow",
    "leftarrow",
    "longrightarrow",
    "longleftarrow",
    "longleftrightarrow",
    "iff",
    "land",
    "lor",
    "oplus",
    "otimes",
}

DECORATION_MACROS = {
    "quad",
    "qquad",
    "hspace",
    "vspace",
}

NUMERIC_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
TOKEN_PATTERN = re.compile(r"\\[A-Za-z]+|[A-Za-z]+|\d+|[^\s]")
OPERATOR_CHARS = set("=+-*/<>")
DECORATION_TOKENS = {"(", ")", "[", "]", ",", ";", ":"}


def extract_formulas(tex_file: Path) -> List[MathFormula]:
    """
    Extract all mathematical formulas from a LaTeX file.

    Args:
        tex_file: Path to the consolidated .tex file

    Returns:
        List of MathFormula objects describing detected formulas
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
        if _is_trivial_formula(entry["formula"]):
            continue

        seen_ranges.add((start, end))

        formulas.append(
            MathFormula(
                formula=entry["formula"],
                formula_type=entry["environment"],
                line_number=entry["line_number"],
                raw_latex=entry["raw_latex"],
                start_pos=start,
                end_pos=end,
                source_path=tex_file,
            )
        )

    formulas.sort(key=lambda f: (f.line_number, f.start_pos))
    return formulas


def generate_formula_label(index: int) -> str:
    """
    Generate a unique label for a formula.

    Args:
        index: Zero-based index of the formula

    Returns:
        Label string like <<FORMULA_0001>>

    Examples:
        >>> generate_formula_label(0)
        '<<FORMULA_0001>>'
        >>> generate_formula_label(42)
        '<<FORMULA_0043>>'
    """
    # Use 1-based indexing for human readability
    return f"<<FORMULA_{index + 1:04d}>>"


def extract_and_label_formulas(
    tex_file: Path,
    output_dir: Optional[Path] = None
) -> Tuple[Path, Path]:
    """
    Extract formulas and generate labeled output files.

    This function creates two output files:
    1. A JSON file mapping labels to formula metadata (no context stored)
    2. A modified .tex file with formulas replaced by labels

    Args:
        tex_file: Path to the consolidated .tex file
        output_dir: Directory for output files (default: same as input file)

    Returns:
        Tuple of (json_path, labeled_tex_path)

    Examples:
        >>> json_path, tex_path = extract_and_label_formulas(
        ...     Path("paper_consolidated.tex"),
        ...     Path("./output")
        ... )
    """
    import json

    tex_file = Path(tex_file)
    if not tex_file.exists():
        raise FileNotFoundError(f"LaTeX file not found: {tex_file}")

    # Determine output directory
    if output_dir is None:
        output_dir = tex_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output file names based on input file
    base_name = tex_file.stem  # e.g., "1706.03762_consolidated"
    json_path = output_dir / f"{base_name}_formulas.json"
    labeled_tex_path = output_dir / f"{base_name}_labeled.tex"

    print(f"\n{'='*70}")
    print(f"Extracting and Labeling Formulas")
    print(f"{'='*70}")
    print(f"Input:  {tex_file}")
    print(f"Output: {json_path}")
    print(f"        {labeled_tex_path}")

    # Extract formulas
    formulas = extract_formulas(tex_file)
    print(f"\nExtracted {len(formulas)} formulas")

    # Filter out overlapping/nested formulas (keep outermost ones only)
    # Sort by start position first
    formulas_by_start = sorted(formulas, key=lambda f: f.start_pos)

    non_overlapping_formulas = []
    max_end_pos = -1

    for formula in formulas_by_start:
        # If this formula starts before the previous one ended, it's nested/overlapping
        if formula.start_pos < max_end_pos:
            # Skip nested formula
            continue

        # This formula doesn't overlap, keep it
        non_overlapping_formulas.append(formula)
        max_end_pos = formula.end_pos

    skipped_count = len(formulas) - len(non_overlapping_formulas)
    if skipped_count > 0:
        print(f"Filtered out {skipped_count} nested/overlapping formulas")

    formulas = non_overlapping_formulas
    print(f"Processing {len(formulas)} non-overlapping formulas")

    # Read original content
    tex_content = tex_file.read_text(encoding="utf-8", errors="ignore")

    # Create label mapping
    label_to_metadata: Dict[str, Dict[str, Any]] = {}

    # Sort formulas by position (reverse order for safe replacement)
    formulas_sorted = sorted(formulas, key=lambda f: f.start_pos, reverse=True)

    # Replace formulas with labels (from end to start)
    labeled_content = tex_content
    for i, formula in enumerate(formulas_sorted):
        # Generate label (use original index from sorted list)
        original_index = len(formulas) - 1 - i
        label = generate_formula_label(original_index)

        # Store metadata
        label_to_metadata[label] = {
            "formula": formula.formula,
            "raw_latex": formula.raw_latex,
            "formula_type": formula.formula_type,
            "line_number": formula.line_number,
            "start_pos": formula.start_pos,
            "end_pos": formula.end_pos,
        }

        # Replace in content
        labeled_content = (
            labeled_content[:formula.start_pos] +
            label +
            labeled_content[formula.end_pos:]
        )

    # Write JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(label_to_metadata, f, indent=2, ensure_ascii=False)

    # Write labeled tex file
    labeled_tex_path.write_text(labeled_content, encoding='utf-8')

    print(f"\n{'='*70}")
    print(f"Output Files Created:")
    print(f"{'='*70}")
    print(f"JSON:        {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"Labeled TEX: {labeled_tex_path} ({labeled_tex_path.stat().st_size / 1024:.1f} KB)")
    print(f"{'='*70}\n")

    return json_path, labeled_tex_path


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

        # If parser succeeded but found nothing, fall back to regex
        if entries:
            return entries

    # Fallback to regex extraction when parser fails or returns empty results
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

        # If parser succeeded but found nothing, fall back to regex
        if entries:
            return entries

    # Fallback to regex extraction when parser fails or returns empty results
    entries.extend(_extract_display_with_regex(tex_content))
    return entries


def get_context(
    tex_content: str, position: int, words: int = 200, span: int = 0
) -> tuple[str, str]:
    """
    Extract context around a specific position in the text.

    Args:
        tex_content: Full LaTeX content
        position: Character offset where the formula begins
        words: Number of words to extract before/after the formula
        span: Length of the formula in characters (used to skip the formula itself)

    Returns:
        Tuple of (context_before, context_after)
    """
    words = max(words, 0)
    before_segment = tex_content[:position]
    after_segment = tex_content[position + max(span, 0):]

    before = _last_words(before_segment, words)
    after = _first_words(after_segment, words)

    return _clean_context(before), _clean_context(after)


def formulas_to_dict(
    formulas: List[MathFormula],
    include_context: bool = False,
    context_words: int = 200,
) -> List[Dict[str, Any]]:
    """
    Convert list of MathFormula objects to dictionaries for serialization.

    Args:
        formulas: List of MathFormula objects
        include_context: Whether to include context snippets in the result
        context_words: Number of words to use when computing context

    Returns:
        List of dictionaries suitable for JSON serialization
    """
    result: List[Dict[str, Any]] = []
    content_cache: Dict[Path, str] = {}

    for formula in formulas:
        source_path = str(formula.source_path) if formula.source_path else None

        entry: Dict[str, Any] = {
            "formula": formula.formula,
            "formula_type": formula.formula_type,
            "line_number": formula.line_number,
            "raw_latex": formula.raw_latex,
            "start_pos": formula.start_pos,
            "end_pos": formula.end_pos,
            "source_path": source_path,
        }

        if include_context:
            before = after = ""
            if formula.source_path:
                path = Path(formula.source_path)
                if path not in content_cache:
                    content_cache[path] = path.read_text(
                        encoding="utf-8", errors="ignore"
                    )
                before, after = get_context(
                    content_cache[path],
                    position=formula.start_pos,
                    words=context_words,
                    span=formula.end_pos - formula.start_pos,
                )
            entry["context_before"] = before
            entry["context_after"] = after

        result.append(entry)
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


def _is_trivial_formula(formula: str) -> bool:
    if not formula:
        return True

    stripped = formula.strip()
    if not stripped:
        return True

    prepared = _prepare_formula_for_analysis(stripped)
    if not prepared:
        return True

    tokens = [
        tok for tok in TOKEN_PATTERN.findall(prepared) if tok and not tok.isspace()
    ]

    if not tokens:
        return True

    symbol_count = 0
    distinct_symbols: set[str] = set()
    numeric_tokens = 0
    has_operator = False
    has_complex_macro = False

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in DECORATION_TOKENS:
            i += 1
            continue

        if token.startswith("\\"):
            name = token[1:]
            lower_name = name.lower()

            if lower_name in OPERATOR_MACROS:
                has_operator = True
                i += 1
                continue

            if lower_name in DECORATION_MACROS:
                i += 1
                continue

            if lower_name in COMPLEX_MACROS:
                has_complex_macro = True

            if any(lower_name.startswith(prefix) for prefix in FONT_MACRO_PREFIXES):
                if (
                    i + 1 < len(tokens)
                    and len(tokens[i + 1]) == 1
                    and tokens[i + 1].isalpha()
                ):
                    combined = f"{lower_name}:{tokens[i + 1]}"
                    symbol_count += 1
                    distinct_symbols.add(combined)
                    i += 2
                    continue

            symbol_count += 1
            distinct_symbols.add(lower_name)
            i += 1
            continue

        if token.isdigit():
            numeric_tokens += 1
            i += 1
            continue

        if token.isalpha():
            symbol_count += len(token)
            distinct_symbols.update(token)
            i += 1
            continue

        if any(ch in OPERATOR_CHARS for ch in token):
            has_operator = True
        i += 1

    if symbol_count == 0:
        return True

    if numeric_tokens > 0:
        meaningful_density = symbol_count / (symbol_count + numeric_tokens)
        if meaningful_density < 0.25:
            return True

    if has_operator or has_complex_macro:
        return False

    if symbol_count <= 1 and len(distinct_symbols) <= 1:
        return True

    return False


def _prepare_formula_for_analysis(formula: str) -> str:
    content = re.sub(r"\\(left|right)\b", "", formula)
    content = _remove_sub_supers(content)
    content = content.replace("{", "").replace("}", "")
    return content.strip()


def _remove_sub_supers(expr: str) -> str:
    result: list[str] = []
    i = 0
    length = len(expr)
    while i < length:
        ch = expr[i]
        if ch in ("_", "^"):
            i += 1
            if i < length and expr[i] == "{":
                closing = _find_matching_brace(expr, i)
                if closing is None:
                    break
                i = closing + 1
            else:
                i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def _find_matching_brace(expr: str, start: int) -> Optional[int]:
    if start >= len(expr) or expr[start] != "{":
        return None

    depth = 0
    for idx in range(start, len(expr)):
        ch = expr[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
            if depth < 0:
                return None
    return None


def _last_words(fragment: str, count: int) -> str:
    if count <= 0 or not fragment:
        return ""
    tokens = re.findall(r"\S+", fragment)
    if not tokens:
        return ""
    selected = tokens[-count:]
    return " ".join(selected)


def _first_words(fragment: str, count: int) -> str:
    if count <= 0 or not fragment:
        return ""
    tokens = re.findall(r"\S+", fragment)
    if not tokens:
        return ""
    selected = tokens[:count]
    return " ".join(selected)


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
