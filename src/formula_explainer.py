"""
Explain mathematical formulas extracted from papers.

This module provides placeholder functionality for generating explanations
of mathematical formulas. Future implementations could use:
- LLM APIs (Claude, GPT-4, etc.)
- Rule-based pattern matching
- Mathematical knowledge databases
"""

from typing import Optional, Dict
from pathlib import Path


def explain_formula(
    formula: str,
    context_before: str = "",
    context_after: str = "",
    formula_type: str = "equation"
) -> str:
    """
    Generate an explanation for a mathematical formula.

    Args:
        formula: The LaTeX formula to explain (without delimiters)
        context_before: Text appearing before the formula (for context)
        context_after: Text appearing after the formula (for context)
        formula_type: Type of math environment (inline, equation, align, etc.)

    Returns:
        Human-readable explanation of the formula

    Examples:
        >>> explanation = explain_formula(
        ...     "E = mc^2",
        ...     context_before="Einstein's famous equation",
        ...     context_after="relates energy and mass"
        ... )
        >>> print(explanation)
        "This formula represents the mass-energy equivalence..."

    Note:
        This is currently a placeholder implementation.
        Future versions will implement actual explanation generation.
    """
    # TODO: Implement formula explanation logic
    # Options for implementation:
    # 1. LLM API integration (Claude, GPT-4, etc.)
    # 2. Rule-based pattern matching for common formulas
    # 3. Mathematical knowledge database lookup
    # 4. Symbolic math libraries (SymPy) for analysis

    # Placeholder: return basic info
    return f"[Placeholder] Formula to explain: {formula}"


def explain_formulas_batch(formulas: list[Dict[str, any]]) -> list[Dict[str, any]]:
    """
    Generate explanations for multiple formulas.

    Args:
        formulas: List of formula dictionaries from formula_extractor

    Returns:
        List of formula dictionaries with added 'explanation' field

    Examples:
        >>> from formula_extractor import extract_formulas, formulas_to_dict
        >>> formulas = extract_formulas(Path("paper.tex"))
        >>> formula_dicts = formulas_to_dict(formulas)
        >>> explained = explain_formulas_batch(formula_dicts)
    """
    # TODO: Implement batch explanation logic
    # - Process multiple formulas efficiently
    # - Handle rate limiting for API calls
    # - Cache explanations
    # - Add error handling
    raise NotImplementedError("Batch explanation not yet implemented")


def save_explanations(
    formulas_with_explanations: list[Dict[str, any]],
    output_file: Path,
    format: str = "json"
) -> None:
    """
    Save formula explanations to a file.

    Args:
        formulas_with_explanations: List of formulas with explanations
        output_file: Path to output file
        format: Output format ('json', 'markdown', 'html')

    Examples:
        >>> save_explanations(
        ...     explained_formulas,
        ...     Path("./output/explanations.json"),
        ...     format="json"
        ... )
    """
    # TODO: Implement saving logic
    # - Support multiple output formats
    # - Format nicely for readability
    # - Include metadata (source paper, extraction time, etc.)
    raise NotImplementedError("Save explanations not yet implemented")


def explain_paper_pipeline(
    consolidated_tex: Path,
    output_file: Path,
    context_length: int = 200
) -> Path:
    """
    Complete pipeline: extract formulas and generate explanations.

    Args:
        consolidated_tex: Path to consolidated .tex file
        output_file: Path for output file with explanations
        context_length: Characters of context to use for explanations

    Returns:
        Path to the output file

    Examples:
        >>> result = explain_paper_pipeline(
        ...     Path("./consolidated.tex"),
        ...     Path("./output/explained.json")
        ... )
    """
    # TODO: Implement complete pipeline
    # - Extract formulas with context
    # - Generate explanations
    # - Save results
    # - Return output path
    raise NotImplementedError("Explanation pipeline not yet implemented")


# ============================================================================
# New: Formula Explanation with LLM
# ============================================================================

def explain_formulas_from_labeled_files(
    formulas_json_path: Path,
    labeled_tex_path: Path,
    output_path: Path,
    model: str = "gpt-5",
    context_words: int = 500,
    api_key: Optional[str] = None,
    temperature: float = 0.7
) -> Path:
    """
    Generate explanations for all formulas using LLM.

    This function:
    1. Loads the formula mapping from JSON
    2. Reads the labeled TeX file
    3. For each formula, extracts context dynamically
    4. Calls LLM to generate explanation
    5. Saves results with progress tracking

    Args:
        formulas_json_path: Path to formulas JSON (label -> metadata mapping)
        labeled_tex_path: Path to labeled TeX file (formulas replaced with labels)
        output_path: Path to save explanations JSON
        model: LLM model to use (default: "gpt-5")
        context_words: Number of words of context to extract (default: 500)
        api_key: OpenAI API key (optional, uses environment if not provided)
        temperature: Temperature for generation (ignored for GPT-5)

    Returns:
        Path to the output file with explanations

    Examples:
        >>> result = explain_formulas_from_labeled_files(
        ...     Path("paper_formulas.json"),
        ...     Path("paper_labeled.tex"),
        ...     Path("paper_explained.json"),
        ...     model="gpt-5",
        ...     context_words=500
        ... )
    """
    import json
    from tqdm import tqdm
    from datetime import datetime
    from .llm_client import LLMClient

    print(f"\n{'='*70}")
    print("Generating Formula Explanations with LLM")
    print(f"{'='*70}")

    # Load formula mapping
    print(f"\nLoading formulas from: {formulas_json_path}")
    with open(formulas_json_path, 'r', encoding='utf-8') as f:
        formulas_dict = json.load(f)

    print(f"Loaded {len(formulas_dict)} formulas")

    # Load labeled TeX content
    print(f"Loading labeled TeX from: {labeled_tex_path}")
    labeled_content = labeled_tex_path.read_text(encoding='utf-8', errors='ignore')

    # Initialize LLM client
    print(f"\nInitializing LLM client (model: {model})...")
    llm_client = LLMClient(
        model=model,
        api_key=api_key,
        temperature=temperature
    )

    # Prepare output structure
    explanations = {}
    failed_formulas = []

    # Process each formula with progress bar
    print(f"\nGenerating explanations...")
    for label, metadata in tqdm(formulas_dict.items(), desc="Explaining formulas"):
        try:
            # Find label position in labeled tex
            label_pos = labeled_content.find(label)
            if label_pos == -1:
                print(f"\nWarning: Label {label} not found in labeled TeX")
                failed_formulas.append({
                    "label": label,
                    "error": "Label not found in text"
                })
                continue

            # Extract context around the label
            context_before, context_after = get_context(
                labeled_content,
                label_pos,
                words=context_words,
                span=len(label)
            )

            # Generate explanation using LLM
            explanation = llm_client.explain_formula(
                formula=metadata['formula'],
                context_before=context_before,
                context_after=context_after,
                additional_context=f"Formula from scientific paper (Line {metadata['line_number']})"
            )

            # Store result
            explanations[label] = {
                **metadata,  # Include original metadata
                "explanation": explanation,
                "model_used": model,
                "timestamp": datetime.now().isoformat(),
                "context_words": context_words
            }

        except Exception as e:
            error_msg = str(e)
            print(f"\nError explaining {label}: {error_msg}")
            failed_formulas.append({
                "label": label,
                "formula": metadata.get('formula', ''),
                "error": error_msg
            })

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "explanations": explanations,
        "failed": failed_formulas,
        "metadata": {
            "model": model,
            "context_words": context_words,
            "timestamp": datetime.now().isoformat(),
            "total_formulas": len(formulas_dict),
            "successful": len(explanations),
            "failed": len(failed_formulas)
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*70}")
    print("Explanation Generation Complete")
    print(f"{'='*70}")
    print(f"Total formulas: {len(formulas_dict)}")
    print(f"Successful:     {len(explanations)}")
    print(f"Failed:         {len(failed_formulas)}")
    print(f"\nOutput saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"{'='*70}\n")

    return output_path
