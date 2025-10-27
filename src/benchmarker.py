"""
Benchmark LLM's ability to detect mathematical errors in formulas.

This module evaluates how well an LLM can identify errors in mathematical
formulas by comparing its detections against ground truth error logs.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .llm_client import LLMClient
from .formula_extractor import get_context


def run_benchmark(
    paper_dir: Path,
    model: str = "gpt-5",
    context_words: int = 300,
    max_workers: int = 10
) -> Tuple[Path, Optional[Path]]:
    """
    Benchmark LLM's error detection ability on a processed paper.

    Args:
        paper_dir: Directory containing paper outputs (e.g., output/1706.03762)
        model: LLM model to use for detection
        context_words: Number of words of context to extract
        max_workers: Number of concurrent API calls

    Returns:
        Tuple of (detection_results_path, benchmark_report_path)
        benchmark_report_path is None if no error_log exists
    """
    paper_dir = Path(paper_dir)
    paper_id = paper_dir.name

    print(f"\n{'='*70}")
    print("Benchmarking LLM Error Detection")
    print(f"{'='*70}")
    print(f"Paper directory: {paper_dir}")
    print(f"Model: {model}")
    print(f"Context words: {context_words}")
    print(f"Max workers: {max_workers}")

    # Load required files
    print(f"\nLoading files...")
    explained_path = paper_dir / f"{paper_id}_explained.json"
    if not explained_path.exists():
        raise FileNotFoundError(f"Required file not found: {explained_path}")

    with open(explained_path, 'r', encoding='utf-8') as f:
        explained_data = json.load(f)
    formulas_to_check = explained_data.get('formulas', [])
    print(f"  Loaded {len(formulas_to_check)} formulas from explained.json")

    # Load formulas JSON (with or without errors)
    formulas_with_errors_path = paper_dir / f"{paper_id}_formulas_with_errors.json"
    formulas_original_path = paper_dir / f"{paper_id}_formulas.json"

    if formulas_with_errors_path.exists():
        formulas_json_path = formulas_with_errors_path
        print(f"  Using formulas_with_errors.json (error injection mode)")
    elif formulas_original_path.exists():
        formulas_json_path = formulas_original_path
        print(f"  Using formulas.json (no error injection)")
    else:
        raise FileNotFoundError(f"No formulas JSON found in {paper_dir}")

    with open(formulas_json_path, 'r', encoding='utf-8') as f:
        all_formulas_dict = json.load(f)

    # Load labeled TeX for context
    labeled_tex_path = paper_dir / f"{paper_id}_consolidated_labeled.tex"
    if not labeled_tex_path.exists():
        raise FileNotFoundError(f"Required file not found: {labeled_tex_path}")

    labeled_content = labeled_tex_path.read_text(encoding='utf-8', errors='ignore')
    print(f"  Loaded labeled TeX file")

    # Load error log if exists
    error_log_path = paper_dir / f"{paper_id}_error_log.json"
    error_log = None
    if error_log_path.exists():
        with open(error_log_path, 'r', encoding='utf-8') as f:
            error_log = json.load(f)
        print(f"  Loaded error log ({error_log['metadata']['formulas_modified']} errors injected)")
    else:
        print(f"  No error log found (benchmarking without ground truth)")

    # Initialize LLM client
    print(f"\nInitializing LLM client (model: {model})...")
    llm_client = LLMClient(model=model)

    # Helper function to replace labels in context
    def replace_labels_in_context(text: str) -> str:
        """Replace all formula labels in text with their original formulas."""
        import re
        label_pattern = r'<<FORMULA_\d+>>'
        labels_in_text = re.findall(label_pattern, text)

        result = text
        for label in labels_in_text:
            if label in all_formulas_dict:
                original_formula = all_formulas_dict[label].get('raw_latex', label)
                result = result.replace(label, original_formula)

        return result

    # Worker function to check a single formula
    def check_formula(formula_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Check a single formula for errors."""
        label = formula_entry['label']
        formula = formula_entry['formula']

        try:
            # Find label position in labeled tex
            label_pos = labeled_content.find(label)
            if label_pos == -1:
                return {
                    "label": label,
                    "formula": formula,
                    "error": "Label not found in labeled TeX",
                    "has_error": None
                }

            # Extract context
            context_before, context_after = get_context(
                labeled_content,
                label_pos,
                words=context_words,
                span=len(label)
            )

            # Replace labels in context
            context_before = replace_labels_in_context(context_before)
            context_after = replace_labels_in_context(context_after)

            # Check formula for errors
            start_time = time.time()
            result = _check_formula_for_error(
                formula,
                context_before,
                context_after,
                llm_client
            )
            response_time = time.time() - start_time

            return {
                "label": label,
                "formula": formula,
                "has_error": result.get('has_error', False),
                "error_type": result.get('error_type', 'none'),
                "error_description": result.get('error_description', ''),
                "llm_response_time": round(response_time, 2)
            }

        except Exception as e:
            return {
                "label": label,
                "formula": formula,
                "error": str(e),
                "has_error": None
            }

    # Run error detection concurrently
    print(f"\nRunning error detection with {max_workers} workers...")
    detections = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_formula = {
            executor.submit(check_formula, formula): formula
            for formula in formulas_to_check
        }

        for future in tqdm(as_completed(future_to_formula), total=len(formulas_to_check), desc="Checking formulas"):
            result = future.result()
            detections.append(result)

    # Save detection results
    detection_results = {
        "metadata": {
            "paper_id": paper_id,
            "model": model,
            "context_words": context_words,
            "total_formulas_checked": len(detections),
            "timestamp": datetime.now().isoformat(),
            "benchmark_mode": True
        },
        "detections": detections
    }

    detection_path = paper_dir / f"{paper_id}_error_detection.json"
    with open(detection_path, 'w', encoding='utf-8') as f:
        json.dump(detection_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Detection results saved: {detection_path.name}")
    print(f"  Total formulas checked: {len(detections)}")
    errors_detected = sum(1 for d in detections if d.get('has_error'))
    print(f"  Errors detected by LLM: {errors_detected}")

    # Calculate metrics if error log exists
    benchmark_report_path = None
    if error_log:
        print(f"\nCalculating benchmark metrics...")
        report = _calculate_metrics(detections, error_log, paper_id, model)

        benchmark_report_path = paper_dir / f"{paper_id}_benchmark_report.json"
        with open(benchmark_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Benchmark report saved: {benchmark_report_path.name}")
        _print_metrics_summary(report)

    print(f"\n{'='*70}\n")
    return detection_path, benchmark_report_path


def _check_formula_for_error(
    formula: str,
    context_before: str,
    context_after: str,
    llm_client: LLMClient
) -> Dict[str, Any]:
    """
    Use LLM to check if a formula contains mathematical errors.

    Args:
        formula: The LaTeX formula to check
        context_before: Text before the formula
        context_after: Text after the formula
        llm_client: LLM client for API calls

    Returns:
        Dictionary with has_error, error_type, error_description
    """
    system_prompt = """You are an expert at detecting mathematical errors in formulas.

Your task is to check if a given formula contains any mathematical errors based on the surrounding context.

Common error types to watch for:
- sign_flip: Wrong sign (+ should be -, etc.)
- operator_swap: Wrong operator (× should be +, etc.)
- exponent_order: Exponent in wrong place (E(X)^2 vs E(X^2))
- index_change: Wrong subscript/index (x_i should be x_j)
- transpose_error: Missing or extra transpose (^T)
- inequality_flip: Wrong inequality direction (<= vs >=)
- fraction_inversion: Numerator and denominator swapped
- sum_product_swap: Sum (∑) should be product (∏) or vice versa
- missing_parentheses: Parentheses missing causing precedence error
- function_swap: Wrong function (sin vs cos, max vs min, log vs ln)

Respond in JSON format:
{
  "has_error": true or false,
  "error_type": "sign_flip" | "operator_swap" | ... | "none",
  "error_description": "Detailed description of the error if found, or empty string if no error"
}"""

    user_prompt = f"""Check if this formula contains a mathematical error:

Formula: {formula}

Context before: {context_before}

Context after: {context_after}

Analyze the formula carefully. Does it contain any mathematical error? Respond in JSON format."""

    # Call LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Use the same approach as explain_formula
    api_params = {
        "model": llm_client.model,
        "messages": messages,
    }

    if not llm_client._is_gpt5:
        api_params["temperature"] = llm_client.temperature

    response = llm_client.client.chat.completions.create(**api_params)
    response_text = response.choices[0].message.content.strip()

    # Parse JSON response
    try:
        # Handle markdown code blocks
        text = response_text
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        result = json.loads(text)

        # Validate required fields
        if "has_error" not in result:
            result["has_error"] = False
        if "error_type" not in result:
            result["error_type"] = "none"
        if "error_description" not in result:
            result["error_description"] = ""

        return result

    except json.JSONDecodeError:
        # If parsing fails, return conservative response
        return {
            "has_error": False,
            "error_type": "none",
            "error_description": "",
            "parse_error": response_text[:200]
        }


def _calculate_metrics(
    detections: List[Dict[str, Any]],
    error_log: Dict[str, Any],
    paper_id: str,
    model: str
) -> Dict[str, Any]:
    """
    Calculate benchmark metrics by comparing detections to ground truth.

    Args:
        detections: List of detection results from LLM
        error_log: Ground truth error log
        paper_id: Paper identifier
        model: Model name used

    Returns:
        Complete benchmark report with metrics
    """
    # Build ground truth lookup
    ground_truth = {}

    # Formulas with errors (from error_log.errors)
    for error_entry in error_log.get('errors', []):
        label = error_entry['label']
        ground_truth[label] = {
            "has_error": True,
            "error_type": error_entry['error_type']
        }

    # Unmodified formulas (from error_log.unmodified)
    for label in error_log.get('unmodified', []):
        ground_truth[label] = {
            "has_error": False,
            "error_type": "none"
        }

    # Calculate classification metrics
    tp = 0  # True positive: detected error, error exists
    fp = 0  # False positive: detected error, no error exists
    tn = 0  # True negative: no error detected, no error exists
    fn = 0  # False negative: no error detected, error exists

    type_correct = 0  # Error type correctly identified
    type_incorrect = 0  # Error type incorrectly identified

    detailed_results = []
    per_type_stats = {}

    for detection in detections:
        label = detection['label']

        # Skip if error during detection
        if detection.get('error') or detection.get('has_error') is None:
            continue

        # Get ground truth for this formula
        gt = ground_truth.get(label, {"has_error": False, "error_type": "none"})

        detected_has_error = detection.get('has_error', False)
        detected_type = detection.get('error_type', 'none')
        gt_has_error = gt['has_error']
        gt_type = gt['error_type']

        # Classification
        if detected_has_error and gt_has_error:
            result = "TP"
            tp += 1
            # Check error type match
            if _match_error_type(detected_type, gt_type):
                type_correct += 1
            else:
                type_incorrect += 1
        elif detected_has_error and not gt_has_error:
            result = "FP"
            fp += 1
        elif not detected_has_error and not gt_has_error:
            result = "TN"
            tn += 1
        elif not detected_has_error and gt_has_error:
            result = "FN"
            fn += 1
        else:
            result = "UNKNOWN"

        # Per-error-type stats
        if gt_has_error:
            if gt_type not in per_type_stats:
                per_type_stats[gt_type] = {"detected": 0, "total": 0}
            per_type_stats[gt_type]["total"] += 1
            if detected_has_error:
                per_type_stats[gt_type]["detected"] += 1

        detailed_results.append({
            "label": label,
            "ground_truth_has_error": gt_has_error,
            "ground_truth_error_type": gt_type,
            "detected_has_error": detected_has_error,
            "detected_error_type": detected_type,
            "result": result,
            "type_match": _match_error_type(detected_type, gt_type) if gt_has_error and detected_has_error else None
        })

    # Calculate aggregate metrics
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    type_accuracy = type_correct / (type_correct + type_incorrect) if (type_correct + type_incorrect) > 0 else 0

    # Calculate per-type recall
    per_type_performance = {}
    for error_type, stats in per_type_stats.items():
        recall_for_type = stats["detected"] / stats["total"] if stats["total"] > 0 else 0
        per_type_performance[error_type] = {
            "detected": stats["detected"],
            "total": stats["total"],
            "recall": round(recall_for_type, 3)
        }

    # Build report
    report = {
        "metadata": {
            "paper_id": paper_id,
            "model": model,
            "total_formulas": total,
            "formulas_with_errors_injected": error_log['metadata']['formulas_modified'],
            "formulas_unmodified": error_log['metadata']['formulas_unmodified'],
            "timestamp": datetime.now().isoformat()
        },
        "binary_classification": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3)
        },
        "error_type_matching": {
            "total_errors_detected": tp,
            "correct_type_identified": type_correct,
            "incorrect_type_identified": type_incorrect,
            "type_accuracy": round(type_accuracy, 3)
        },
        "per_error_type_performance": per_type_performance,
        "detailed_results": detailed_results
    }

    return report


def _match_error_type(detected: str, actual: str) -> bool:
    """
    Check if detected error type matches actual error type.

    Allows for some flexibility in naming.
    """
    if detected == actual:
        return True

    # Normalize names
    detected_norm = detected.lower().replace('_', '').replace('-', '')
    actual_norm = actual.lower().replace('_', '').replace('-', '')

    return detected_norm == actual_norm


def _print_metrics_summary(report: Dict[str, Any]) -> None:
    """Print a summary of benchmark metrics to console."""
    bc = report['binary_classification']
    et = report['error_type_matching']

    print(f"\n{'='*70}")
    print("Benchmark Metrics Summary")
    print(f"{'='*70}")

    print(f"\nBinary Classification:")
    print(f"  Accuracy:  {bc['accuracy']:.1%}")
    print(f"  Precision: {bc['precision']:.1%}")
    print(f"  Recall:    {bc['recall']:.1%}")
    print(f"  F1 Score:  {bc['f1_score']:.1%}")

    print(f"\nConfusion Matrix:")
    print(f"  TP: {bc['true_positives']:<4}  FP: {bc['false_positives']}")
    print(f"  FN: {bc['false_negatives']:<4}  TN: {bc['true_negatives']}")

    print(f"\nError Type Identification:")
    print(f"  Correct types: {et['correct_type_identified']}/{et['total_errors_detected']} ({et['type_accuracy']:.1%})")

    if report.get('per_error_type_performance'):
        print(f"\nPer-Error-Type Recall:")
        for error_type, stats in sorted(report['per_error_type_performance'].items()):
            print(f"  {error_type:<20} {stats['detected']}/{stats['total']} ({stats['recall']:.1%})")
