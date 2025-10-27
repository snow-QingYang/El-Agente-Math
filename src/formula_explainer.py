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
    temperature: float = 0.7,
    max_workers: int = 10,
    max_formulas: int = 50
) -> Path:
    """
    Generate explanations for all formulas using LLM with concurrent processing.

    This function:
    1. Loads the formula mapping from JSON
    2. Sorts and filters formulas by length (longest first, up to max_formulas)
    3. Reads the labeled TeX file
    4. For each formula, extracts context dynamically (replacing labels with original formulas)
    5. Calls LLM to generate explanation (concurrently using ThreadPoolExecutor)
    6. Saves results with progress tracking

    Args:
        formulas_json_path: Path to formulas JSON (label -> metadata mapping)
        labeled_tex_path: Path to labeled TeX file (formulas replaced with labels)
        output_path: Path to save explanations JSON
        model: LLM model to use (default: "gpt-5")
        context_words: Number of words of context to extract (default: 500)
        api_key: OpenAI API key (optional, uses environment if not provided)
        temperature: Temperature for generation (ignored for GPT-5)
        max_workers: Maximum number of concurrent API calls (default: 10)
        max_formulas: Maximum number of formulas to explain (default: 50, prioritizes longest formulas)

    Returns:
        Path to the output file with explanations

    Examples:
        >>> result = explain_formulas_from_labeled_files(
        ...     Path("paper_formulas.json"),
        ...     Path("paper_labeled.tex"),
        ...     Path("paper_explained.json"),
        ...     model="gpt-5",
        ...     context_words=500,
        ...     max_workers=10,
        ...     max_formulas=50
        ... )
    """
    import json
    from tqdm import tqdm
    from datetime import datetime
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .llm_client import LLMClient
    from .formula_extractor import get_context

    print(f"\n{'='*70}")
    print("Generating Formula Explanations with LLM")
    print(f"{'='*70}")

    # Load formula mapping
    print(f"\nLoading formulas from: {formulas_json_path}")
    with open(formulas_json_path, 'r', encoding='utf-8') as f:
        all_formulas_dict = json.load(f)

    print(f"Loaded {len(all_formulas_dict)} formulas")

    # Sort formulas by length (descending - longest first) and limit to max_formulas
    sorted_formulas = sorted(
        all_formulas_dict.items(),
        key=lambda x: len(x[1].get('formula', '')),
        reverse=True
    )

    # Take only top max_formulas
    if len(sorted_formulas) > max_formulas:
        print(f"Limiting to {max_formulas} longest formulas (out of {len(sorted_formulas)} total)")
        formulas_to_process = dict(sorted_formulas[:max_formulas])
        skipped_by_limit = len(sorted_formulas) - max_formulas
    else:
        formulas_to_process = all_formulas_dict
        skipped_by_limit = 0

    print(f"Processing {len(formulas_to_process)} formulas")

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

    # Helper function to replace labels with original formulas in context
    def replace_labels_in_context(text: str) -> str:
        """Replace all formula labels in text with their original formulas."""
        import re
        # Find all labels like <<FORMULA_0001>>
        label_pattern = r'<<FORMULA_\d+>>'
        labels_in_text = re.findall(label_pattern, text)

        result = text
        for label in labels_in_text:
            if label in all_formulas_dict:
                # Replace label with original raw LaTeX
                original_formula = all_formulas_dict[label].get('raw_latex', label)
                result = result.replace(label, original_formula)

        return result

    # Define worker function for processing a single formula
    def process_formula(label: str, metadata: dict) -> dict:
        """Process a single formula and return result dict."""
        try:
            # Find label position in labeled tex
            label_pos = labeled_content.find(label)
            if label_pos == -1:
                return {
                    "status": "failed",
                    "label": label,
                    "formula": metadata.get('formula', ''),
                    "error": "Label not found in text"
                }

            # Extract context around the label
            context_before, context_after = get_context(
                labeled_content,
                label_pos,
                words=context_words,
                span=len(label)
            )

            # Replace formula labels in context with original formulas
            context_before = replace_labels_in_context(context_before)
            context_after = replace_labels_in_context(context_after)

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
                return {
                    "status": "skipped",
                    "label": label,
                    "formula": metadata['formula'],
                    "reason": "Classified as notation, not formula"
                }

            # Build output entry for formula
            return {
                "status": "success",
                "data": {
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
            }

        except Exception as e:
            return {
                "status": "failed",
                "label": label,
                "formula": metadata.get('formula', ''),
                "error": str(e)
            }

    # Process formulas concurrently
    explanations_list = []
    skipped_notations = []
    failed_formulas = []

    print(f"\nGenerating explanations with {max_workers} concurrent workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all formulas for processing
        future_to_label = {
            executor.submit(process_formula, label, metadata): label
            for label, metadata in formulas_to_process.items()
        }

        # Process completed futures with progress bar
        for future in tqdm(as_completed(future_to_label), total=len(formulas_to_process), desc="Analyzing formulas"):
            result = future.result()

            if result["status"] == "success":
                explanations_list.append(result["data"])
            elif result["status"] == "skipped":
                skipped_notations.append({
                    "label": result["label"],
                    "formula": result["formula"],
                    "reason": result["reason"]
                })
            elif result["status"] == "failed":
                print(f"\nError explaining {result['label']}: {result['error']}")
                failed_formulas.append({
                    "label": result["label"],
                    "formula": result["formula"],
                    "error": result["error"]
                })

    # Sort results by line number (ascending order)
    explanations_list.sort(key=lambda x: x.get('line_number', 0))
    skipped_notations.sort(key=lambda x: all_formulas_dict.get(x['label'], {}).get('line_number', 0))
    failed_formulas.sort(key=lambda x: all_formulas_dict.get(x['label'], {}).get('line_number', 0))

    # Save results as a list of dicts
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "formulas": explanations_list,
        "metadata": {
            "model": model,
            "context_words": context_words,
            "max_formulas": max_formulas,
            "timestamp": datetime.now().isoformat(),
            "total_formulas_in_paper": len(all_formulas_dict),
            "formulas_selected_for_analysis": len(formulas_to_process),
            "skipped_by_length_limit": skipped_by_limit,
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
    print(f"Total formulas in paper:     {len(all_formulas_dict)}")
    print(f"Formulas selected (longest): {len(formulas_to_process)}")
    if skipped_by_limit > 0:
        print(f"Skipped by length limit:     {skipped_by_limit}")
    print(f"Formulas explained:          {len(explanations_list)}")
    print(f"Notations skipped:           {len(skipped_notations)}")
    print(f"Failed:                      {len(failed_formulas)}")
    print(f"\nOutput saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"{'='*70}\n")

    return output_path
