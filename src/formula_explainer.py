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
    from .formula_extractor import get_context

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

    # Process each formula
    explanations_list = []
    skipped_notations = []
    failed_formulas = []

    # Process each formula with progress bar
    print(f"\nGenerating explanations...")
    for label, metadata in tqdm(formulas_dict.items(), desc="Analyzing formulas"):
        try:
            # Find label position in labeled tex
            label_pos = labeled_content.find(label)
            if label_pos == -1:
                print(f"\nWarning: Label {label} not found in labeled TeX")
                failed_formulas.append({
                    "label": label,
                    "formula": metadata.get('formula', ''),
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

            # Generate structured explanation using LLM
            explanation_result = llm_client.explain_formula_structured(
                formula=metadata['formula'],
                context_before=context_before,
                context_after=context_after,
                additional_context=f"Formula from scientific paper (Line {metadata['line_number']})"
            )

            # Check if it's a formula or just notation
            is_formula = explanation_result.get('is_formula', False)

            if not is_formula:
                # Skip notations - don't include in output
                skipped_notations.append({
                    "label": label,
                    "formula": metadata['formula'],
                    "reason": "Classified as notation, not formula"
                })
                continue

            # Build output entry for formula
            explanation_entry = {
                "label": label,
                "formula": metadata['formula'],
                "raw_latex": metadata['raw_latex'],
                "formula_type": metadata['formula_type'],
                "line_number": metadata['line_number'],
                "is_formula": True,
                "high_level_explanation": explanation_result.get('high_level_explanation', ''),
                "notations": explanation_result.get('notations', {}),
                "model_used": model,
                "timestamp": datetime.now().isoformat()
            }

            explanations_list.append(explanation_entry)

        except Exception as e:
            error_msg = str(e)
            print(f"\nError explaining {label}: {error_msg}")
            failed_formulas.append({
                "label": label,
                "formula": metadata.get('formula', ''),
                "error": error_msg
            })

    # Save results as a list of dicts
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "formulas": explanations_list,
        "metadata": {
            "model": model,
            "context_words": context_words,
            "timestamp": datetime.now().isoformat(),
            "total_analyzed": len(formulas_dict),
            "formulas_explained": len(explanations_list),
            "notations_skipped": len(skipped_notations),
            "failed": len(failed_formulas)
        },
        "skipped_notations": skipped_notations,
        "failed": failed_formulas
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*70}")
    print("Formula Explanation Complete")
    print(f"{'='*70}")
    print(f"Total analyzed:      {len(formulas_dict)}")
    print(f"Formulas explained:  {len(explanations_list)}")
    print(f"Notations skipped:   {len(skipped_notations)}")
    print(f"Failed:              {len(failed_formulas)}")
    print(f"\nOutput saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"{'='*70}\n")

    return output_path
