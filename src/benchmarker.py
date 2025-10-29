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

from .benchmark_providers import get_provider
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

    # Create benchmark directory structure
    # Convert model name to safe directory name (replace / with _)
    model_dir_name = model.replace('/', '_')
    benchmark_dir = paper_dir / "benchmarks" / model_dir_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Benchmarking LLM Error Detection")
    print(f"{'='*70}")
    print(f"Paper directory: {paper_dir}")
    print(f"Model: {model}")
    print(f"Benchmark output: {benchmark_dir}")
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

    # Initialize LLM provider
    print(f"\nInitializing LLM provider (model: {model})...")
    provider = get_provider(model=model)

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
                provider
            )
            response_time = time.time() - start_time

            return {
                "label": label,
                "formula": formula,
                "has_error": result.get('has_error', False),
                "error_type": result.get('error_type', 'none'),
                "error_description": result.get('error_description', ''),
                "parse_success": result.get('parse_success', True),
                "parse_strategy": result.get('parse_strategy', 'unknown'),
                "raw_response": result.get('raw_response', ''),
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

    # Separate detections for error_detection.json (without raw responses)
    # and raw_responses.json (with raw responses)
    detections_clean = []
    raw_responses_data = []
    parsing_failures = []

    for detection in detections:
        # Clean detection (without raw_response for main file)
        clean_det = {k: v for k, v in detection.items() if k != 'raw_response'}
        detections_clean.append(clean_det)

        # Raw response data
        raw_responses_data.append({
            "label": detection.get('label'),
            "raw_response": detection.get('raw_response', ''),
            "response_length": len(detection.get('raw_response', '')),
            "parse_strategy": detection.get('parse_strategy'),
            "parse_success": detection.get('parse_success')
        })

        # Log parsing failures
        if detection.get('parse_strategy') == 'failed':
            parsing_failures.append({
                "label": detection.get('label'),
                "raw_response": detection.get('raw_response', ''),
                "timestamp": datetime.now().isoformat()
            })

    # Save detection results (without raw responses)
    detection_results = {
        "metadata": {
            "paper_id": paper_id,
            "model": model,
            "context_words": context_words,
            "total_formulas_checked": len(detections),
            "timestamp": datetime.now().isoformat(),
            "benchmark_mode": True
        },
        "detections": detections_clean
    }

    detection_path = benchmark_dir / "error_detection.json"
    with open(detection_path, 'w', encoding='utf-8') as f:
        json.dump(detection_results, f, indent=2, ensure_ascii=False)

    # Save raw responses
    raw_responses_file = {
        "metadata": {
            "paper_id": paper_id,
            "model": model,
            "total_responses": len(raw_responses_data),
            "timestamp": datetime.now().isoformat()
        },
        "responses": raw_responses_data
    }

    raw_responses_path = benchmark_dir / "raw_responses.json"
    with open(raw_responses_path, 'w', encoding='utf-8') as f:
        json.dump(raw_responses_file, f, indent=2, ensure_ascii=False)

    # Save parsing failures log
    if parsing_failures:
        failures_log_path = benchmark_dir / "parsing_failures.log"
        with open(failures_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Parsing Failures Log\n")
            f.write(f"Model: {model}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total failures: {len(parsing_failures)}\n")
            f.write("="*70 + "\n\n")

            for i, failure in enumerate(parsing_failures, 1):
                f.write(f"[Failure #{i}] {failure['label']}\n")
                f.write(f"Timestamp: {failure['timestamp']}\n")
                f.write(f"Raw response (first 500 chars):\n")
                f.write(failure['raw_response'][:500] + "...\n" if len(failure['raw_response']) > 500 else failure['raw_response'] + "\n")
                f.write("-"*70 + "\n\n")

    # Calculate parsing statistics
    parse_stats = _calculate_parse_statistics(detections)

    print(f"\n✓ Detection results saved: {detection_path.relative_to(paper_dir)}")
    print(f"✓ Raw responses saved: {raw_responses_path.relative_to(paper_dir)}")
    if parsing_failures:
        print(f"✓ Parsing failures logged: {failures_log_path.relative_to(paper_dir)} ({len(parsing_failures)} failures)")
    print(f"  Total formulas checked: {len(detections)}")
    errors_detected = sum(1 for d in detections if d.get('has_error'))
    print(f"  Errors detected by LLM: {errors_detected}")
    print(f"\n📊 Instruction Following (JSON Format Compliance):")
    print(f"  Perfect JSON responses: {parse_stats['perfect_json']}/{parse_stats['total']} ({parse_stats['perfect_json_rate']:.1%})")
    print(f"  Required fallback parsing: {parse_stats['fallback_used']}/{parse_stats['total']} ({parse_stats['fallback_rate']:.1%})")
    print(f"  Complete parsing failures: {parse_stats['failed']}/{parse_stats['total']} ({parse_stats['failure_rate']:.1%})")

    # Calculate metrics if error log exists
    benchmark_report_path = None
    summary_lines = []  # Collect summary output for saving

    if error_log:
        print(f"\nCalculating benchmark metrics...")
        report = _calculate_metrics(detections, error_log, paper_id, model)

        benchmark_report_path = benchmark_dir / "benchmark_report.json"
        with open(benchmark_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Benchmark report saved: {benchmark_report_path.relative_to(paper_dir)}")

        # Print and capture summary
        summary_lines = _print_and_capture_metrics_summary(report)
    else:
        # No ground truth, just basic summary
        summary_lines = [
            "="*70,
            "Benchmark Summary (No Ground Truth)",
            "="*70,
            f"Model: {model}",
            f"Total formulas checked: {len(detections)}",
            f"Errors detected: {sum(1 for d in detections if d.get('has_error'))}",
            "",
            "Instruction Following:",
            f"  Perfect JSON: {parse_stats['perfect_json']}/{parse_stats['total']} ({parse_stats['perfect_json_rate']:.1%})",
            f"  Fallback required: {parse_stats['fallback_used']}/{parse_stats['total']} ({parse_stats['fallback_rate']:.1%})",
            f"  Parse failures: {parse_stats['failed']}/{parse_stats['total']} ({parse_stats['failure_rate']:.1%})",
            "="*70
        ]
        for line in summary_lines:
            print(line)

    # Save summary to file
    summary_path = benchmark_dir / "summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Benchmark Summary\n")
        f.write(f"Model: {model}\n")
        f.write(f"Paper: {paper_id}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("\n")
        f.write("\n".join(summary_lines))

    print(f"✓ Summary saved: {summary_path.relative_to(paper_dir)}")
    print(f"\n{'='*70}\n")
    return detection_path, benchmark_report_path


def _check_formula_for_error(
    formula: str,
    context_before: str,
    context_after: str,
    provider
) -> Dict[str, Any]:
    """
    Use LLM to check if a formula contains mathematical errors.

    Args:
        formula: The LaTeX formula to check
        context_before: Text before the formula
        context_after: Text after the formula
        provider: LLM provider instance (from benchmark_providers)

    Returns:
        Dictionary with has_error, error_type, error_description
    """
    # Use the provider's detect_error method
    return provider.detect_error(formula, context_before, context_after)


def _calculate_parse_statistics(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate JSON parsing statistics to measure instruction following ability.

    Args:
        detections: List of detection results

    Returns:
        Dictionary with parsing statistics
    """
    total = len(detections)
    perfect_json = 0  # Direct parse success
    fallback_used = 0  # Required fallback strategy
    failed = 0  # Complete failure

    strategy_counts = {}

    for detection in detections:
        parse_success = detection.get('parse_success', True)
        parse_strategy = detection.get('parse_strategy', 'unknown')

        # Count by strategy
        strategy_counts[parse_strategy] = strategy_counts.get(parse_strategy, 0) + 1

        # Categorize
        if parse_strategy == 'direct':
            perfect_json += 1
        elif parse_strategy == 'failed':
            failed += 1
        else:
            fallback_used += 1

    return {
        "total": total,
        "perfect_json": perfect_json,
        "fallback_used": fallback_used,
        "failed": failed,
        "perfect_json_rate": perfect_json / total if total > 0 else 0,
        "fallback_rate": fallback_used / total if total > 0 else 0,
        "failure_rate": failed / total if total > 0 else 0,
        "strategy_breakdown": strategy_counts
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

        # Skip if error during detection or parsing failed
        if detection.get('error') or detection.get('has_error') is None:
            continue

        # Skip if parsing completely failed (no valid prediction)
        if detection.get('parse_strategy') == 'failed':
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

    # Calculate parsing statistics
    parse_stats = _calculate_parse_statistics(detections)

    # Calculate total detections attempted and valid evaluations
    total_detections = len(detections)
    parsing_failures = parse_stats['failed']
    valid_evaluations = total

    # Build report
    report = {
        "metadata": {
            "paper_id": paper_id,
            "model": model,
            "total_detections_attempted": total_detections,
            "parsing_failures": parsing_failures,
            "valid_evaluations": valid_evaluations,
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
        "instruction_following": {
            "perfect_json_responses": parse_stats["perfect_json"],
            "fallback_parsing_required": parse_stats["fallback_used"],
            "parsing_failures": parse_stats["failed"],
            "perfect_json_rate": round(parse_stats["perfect_json_rate"], 3),
            "fallback_rate": round(parse_stats["fallback_rate"], 3),
            "failure_rate": round(parse_stats["failure_rate"], 3),
            "strategy_breakdown": parse_stats["strategy_breakdown"]
        },
        "per_error_type_performance": per_type_performance,
        "detailed_results": detailed_results
    }

    return report


def _match_error_type(detected: str, actual: str) -> bool:
    """
    Check if detected error type matches actual error type.

    Allows for some flexibility in naming (case, underscores, hyphens).
    """
    if detected == actual:
        return True

    # Normalize names
    detected_norm = detected.lower().replace('_', '').replace('-', '')
    actual_norm = actual.lower().replace('_', '').replace('-', '')

    return detected_norm == actual_norm


def _print_and_capture_metrics_summary(report: Dict[str, Any]) -> List[str]:
    """Print metrics summary to console and return as lines for saving."""
    lines = []

    bc = report['binary_classification']
    et = report['error_type_matching']
    meta = report['metadata']

    # Print and capture
    def add_line(text):
        print(text)
        lines.append(text)

    add_line("")
    add_line("="*70)
    add_line("Benchmark Metrics Summary")
    add_line("="*70)

    # Show evaluation coverage
    if 'total_detections_attempted' in meta and 'parsing_failures' in meta:
        add_line(f"\nEvaluation Coverage:")
        add_line(f"  Total detections: {meta['total_detections_attempted']}")
        add_line(f"  Parsing failures: {meta['parsing_failures']} (excluded from metrics)")
        add_line(f"  Valid evaluations: {meta['valid_evaluations']}")

    add_line(f"\nBinary Classification:")
    add_line(f"  Accuracy:  {bc['accuracy']:.1%}")
    add_line(f"  Precision: {bc['precision']:.1%}")
    add_line(f"  Recall:    {bc['recall']:.1%}")
    add_line(f"  F1 Score:  {bc['f1_score']:.1%}")

    add_line(f"\nConfusion Matrix (n={meta.get('valid_evaluations', 'N/A')}):")
    add_line(f"  TP: {bc['true_positives']:<4}  FP: {bc['false_positives']}")
    add_line(f"  FN: {bc['false_negatives']:<4}  TN: {bc['true_negatives']}")

    add_line(f"\nError Type Identification:")
    add_line(f"  Correct types: {et['correct_type_identified']}/{et['total_errors_detected']} ({et['type_accuracy']:.1%})")

    # Instruction following metrics
    if 'instruction_following' in report:
        inf = report['instruction_following']
        add_line(f"\n📊 Instruction Following (JSON Format Compliance):")
        add_line(f"  Perfect JSON:     {inf['perfect_json_responses']}/{inf['perfect_json_responses'] + inf['fallback_parsing_required'] + inf['parsing_failures']} ({inf['perfect_json_rate']:.1%})")
        add_line(f"  Required fallback: {inf['fallback_parsing_required']}/{inf['perfect_json_responses'] + inf['fallback_parsing_required'] + inf['parsing_failures']} ({inf['fallback_rate']:.1%})")
        add_line(f"  Parse failures:    {inf['parsing_failures']}/{inf['perfect_json_responses'] + inf['fallback_parsing_required'] + inf['parsing_failures']} ({inf['failure_rate']:.1%})")

    if report.get('per_error_type_performance'):
        add_line(f"\nPer-Error-Type Recall:")
        for error_type, stats in sorted(report['per_error_type_performance'].items()):
            add_line(f"  {error_type:<20} {stats['detected']}/{stats['total']} ({stats['recall']:.1%})")

    return lines


def run_batch_benchmark(
    output_dir: Path,
    model: str = "openai/gpt-5",
    context_words: int = 300,
    max_workers: int = 10
) -> Path:
    """
    Benchmark LLM on all papers in output directory and generate aggregate report.

    Args:
        output_dir: Directory containing paper subdirectories
        model: LLM model to use for detection
        context_words: Number of words of context to extract
        max_workers: Number of concurrent API calls

    Returns:
        Path to aggregate report JSON file
    """
    output_dir = Path(output_dir)

    print(f"\n{'='*70}")
    print("Batch Benchmarking - All Papers")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model}")
    print(f"Context words: {context_words}")
    print(f"Max workers: {max_workers}")

    # Find all paper directories (contain _explained.json)
    print(f"\nScanning for processed papers...")
    paper_dirs = []
    for item in output_dir.iterdir():
        if item.is_dir() and item.name != "aggregate_benchmarks":
            # Check if this directory has an explained.json file
            explained_file = item / f"{item.name}_explained.json"
            if explained_file.exists():
                paper_dirs.append(item)

    if not paper_dirs:
        raise FileNotFoundError(f"No processed papers found in {output_dir}")

    print(f"  Found {len(paper_dirs)} paper(s) to benchmark:")
    for paper_dir in paper_dirs:
        print(f"    - {paper_dir.name}")

    # Run benchmark on each paper
    print(f"\n{'='*70}")
    print("Running benchmarks on individual papers")
    print(f"{'='*70}\n")

    results = []
    for i, paper_dir in enumerate(paper_dirs, 1):
        print(f"\n[{i}/{len(paper_dirs)}] Benchmarking: {paper_dir.name}")
        print(f"{'-'*70}")

        try:
            detection_path, report_path = run_benchmark(
                paper_dir=paper_dir,
                model=model,
                context_words=context_words,
                max_workers=max_workers
            )

            # Load the report for aggregation
            if report_path and report_path.exists():
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)

                results.append({
                    "paper_id": paper_dir.name,
                    "paper_dir": str(paper_dir),
                    "detection_path": str(detection_path),
                    "report_path": str(report_path),
                    "report": report,
                    "status": "success"
                })
            else:
                results.append({
                    "paper_id": paper_dir.name,
                    "paper_dir": str(paper_dir),
                    "detection_path": str(detection_path),
                    "status": "no_ground_truth"
                })

            print(f"  ✓ Completed: {paper_dir.name}")

        except Exception as e:
            print(f"  ✗ Failed: {paper_dir.name} - {e}")
            results.append({
                "paper_id": paper_dir.name,
                "paper_dir": str(paper_dir),
                "status": "failed",
                "error": str(e)
            })

    # Generate aggregate report
    print(f"\n{'='*70}")
    print("Generating Aggregate Report")
    print(f"{'='*70}\n")

    aggregate_report = _generate_aggregate_report(results, model)

    # Save aggregate report
    model_dir_name = model.replace('/', '_')
    aggregate_dir = output_dir / "aggregate_benchmarks" / model_dir_name
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregate report JSON
    aggregate_report_path = aggregate_dir / "aggregate_report.json"
    with open(aggregate_report_path, 'w', encoding='utf-8') as f:
        json.dump(aggregate_report, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {aggregate_report_path}")

    # Save per-paper summary JSON
    per_paper_summary_path = aggregate_dir / "per_paper_summary.json"
    per_paper_data = {
        "model": model,
        "total_papers": len(paper_dirs),
        "successful_benchmarks": len([r for r in results if r["status"] == "success"]),
        "papers": [
            {
                "paper_id": r["paper_id"],
                "status": r["status"],
                "metrics": r.get("report", {}).get("binary_classification") if r.get("report") else None
            }
            for r in results
        ]
    }
    with open(per_paper_summary_path, 'w', encoding='utf-8') as f:
        json.dump(per_paper_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {per_paper_summary_path}")

    # Save aggregate summary text
    summary_lines = _format_aggregate_summary(aggregate_report, results)
    summary_path = aggregate_dir / "aggregate_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    print(f"✓ Saved: {summary_path}")

    # Print summary to console
    print(f"\n{'-'*70}\n")
    for line in summary_lines:
        print(line)

    return aggregate_report_path


def _generate_aggregate_report(results: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """
    Generate aggregate metrics across all benchmarked papers.

    Args:
        results: List of benchmark results from individual papers
        model: Model name used

    Returns:
        Aggregate report dictionary
    """
    import statistics

    # Filter successful benchmarks with ground truth
    successful_results = [r for r in results if r["status"] == "success" and "report" in r]

    if not successful_results:
        return {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "total_papers": len(results),
            "successful_benchmarks": 0,
            "error": "No successful benchmarks with ground truth available"
        }

    # Collect metrics from all papers
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_type_accuracies = []

    # Instruction following metrics
    all_perfect_json_rates = []
    all_fallback_rates = []
    all_failure_rates = []

    # Per-error-type performance (aggregate across all papers)
    error_type_stats = {}

    for result in successful_results:
        report = result["report"]
        bc = report.get("binary_classification", {})

        all_accuracies.append(bc.get("accuracy", 0))
        all_precisions.append(bc.get("precision", 0))
        all_recalls.append(bc.get("recall", 0))
        all_f1_scores.append(bc.get("f1_score", 0))

        et = report.get("error_type_matching", {})
        all_type_accuracies.append(et.get("type_accuracy", 0))

        # Instruction following
        inf = report.get("instruction_following", {})
        if inf:
            all_perfect_json_rates.append(inf.get("perfect_json_rate", 0))
            all_fallback_rates.append(inf.get("fallback_rate", 0))
            all_failure_rates.append(inf.get("failure_rate", 0))

        # Aggregate per-error-type performance
        per_type = report.get("per_error_type_performance", {})
        for error_type, stats in per_type.items():
            if error_type not in error_type_stats:
                error_type_stats[error_type] = {"detected": 0, "total": 0}
            error_type_stats[error_type]["detected"] += stats["detected"]
            error_type_stats[error_type]["total"] += stats["total"]

    # Calculate aggregate per-error-type recall
    aggregate_per_type = {}
    for error_type, stats in error_type_stats.items():
        recall = stats["detected"] / stats["total"] if stats["total"] > 0 else 0
        aggregate_per_type[error_type] = {
            "detected": stats["detected"],
            "total": stats["total"],
            "recall": recall
        }

    # Build aggregate report
    aggregate = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "total_papers_in_directory": len(results),
            "successful_benchmarks": len(successful_results),
            "failed_benchmarks": len([r for r in results if r["status"] == "failed"]),
            "no_ground_truth": len([r for r in results if r["status"] == "no_ground_truth"])
        },
        "aggregate_metrics": {
            "binary_classification": {
                "mean_accuracy": statistics.mean(all_accuracies) if all_accuracies else 0,
                "mean_precision": statistics.mean(all_precisions) if all_precisions else 0,
                "mean_recall": statistics.mean(all_recalls) if all_recalls else 0,
                "mean_f1_score": statistics.mean(all_f1_scores) if all_f1_scores else 0,
                "std_accuracy": statistics.stdev(all_accuracies) if len(all_accuracies) > 1 else 0,
                "std_precision": statistics.stdev(all_precisions) if len(all_precisions) > 1 else 0,
                "std_recall": statistics.stdev(all_recalls) if len(all_recalls) > 1 else 0,
                "std_f1_score": statistics.stdev(all_f1_scores) if len(all_f1_scores) > 1 else 0,
                "min_accuracy": min(all_accuracies) if all_accuracies else 0,
                "max_accuracy": max(all_accuracies) if all_accuracies else 0
            },
            "error_type_matching": {
                "mean_type_accuracy": statistics.mean(all_type_accuracies) if all_type_accuracies else 0,
                "std_type_accuracy": statistics.stdev(all_type_accuracies) if len(all_type_accuracies) > 1 else 0
            },
            "instruction_following": {
                "mean_perfect_json_rate": statistics.mean(all_perfect_json_rates) if all_perfect_json_rates else 0,
                "mean_fallback_rate": statistics.mean(all_fallback_rates) if all_fallback_rates else 0,
                "mean_failure_rate": statistics.mean(all_failure_rates) if all_failure_rates else 0
            } if all_perfect_json_rates else None
        },
        "per_error_type_performance": aggregate_per_type,
        "paper_ids": [r["paper_id"] for r in successful_results]
    }

    return aggregate


def _format_aggregate_summary(aggregate: Dict[str, Any], all_results: List[Dict[str, Any]]) -> List[str]:
    """
    Format aggregate report as human-readable text lines.

    Args:
        aggregate: Aggregate report dictionary
        all_results: All benchmark results

    Returns:
        List of text lines for summary
    """
    lines = []

    lines.append("=" * 70)
    lines.append("Aggregate Benchmark Report")
    lines.append("=" * 70)
    lines.append(f"\nModel: {aggregate['model']}")
    lines.append(f"Timestamp: {aggregate['timestamp']}")

    meta = aggregate['metadata']
    lines.append(f"\nPapers Processed:")
    lines.append(f"  Total papers: {meta['total_papers_in_directory']}")
    lines.append(f"  Successful benchmarks: {meta['successful_benchmarks']}")
    lines.append(f"  No ground truth: {meta['no_ground_truth']}")
    lines.append(f"  Failed: {meta['failed_benchmarks']}")

    if meta['successful_benchmarks'] > 0:
        bc = aggregate['aggregate_metrics']['binary_classification']
        lines.append(f"\nBinary Classification (Mean ± Std):")
        lines.append(f"  Accuracy:  {bc['mean_accuracy']:.1%} ± {bc['std_accuracy']:.1%}")
        lines.append(f"  Precision: {bc['mean_precision']:.1%} ± {bc['std_precision']:.1%}")
        lines.append(f"  Recall:    {bc['mean_recall']:.1%} ± {bc['std_recall']:.1%}")
        lines.append(f"  F1 Score:  {bc['mean_f1_score']:.1%} ± {bc['std_f1_score']:.1%}")

        lines.append(f"\nAccuracy Range:")
        lines.append(f"  Min: {bc['min_accuracy']:.1%}")
        lines.append(f"  Max: {bc['max_accuracy']:.1%}")

        et = aggregate['aggregate_metrics']['error_type_matching']
        lines.append(f"\nError Type Identification:")
        lines.append(f"  Mean type accuracy: {et['mean_type_accuracy']:.1%} ± {et['std_type_accuracy']:.1%}")

        # Instruction following
        inf = aggregate['aggregate_metrics'].get('instruction_following')
        if inf:
            lines.append(f"\n📊 Instruction Following (JSON Format Compliance):")
            lines.append(f"  Mean perfect JSON rate: {inf['mean_perfect_json_rate']:.1%}")
            lines.append(f"  Mean fallback rate: {inf['mean_fallback_rate']:.1%}")
            lines.append(f"  Mean failure rate: {inf['mean_failure_rate']:.1%}")

        # Per-error-type performance
        per_type = aggregate.get('per_error_type_performance', {})
        if per_type:
            lines.append(f"\nAggregated Per-Error-Type Recall:")
            for error_type, stats in sorted(per_type.items()):
                lines.append(f"  {error_type:<20} {stats['detected']}/{stats['total']} ({stats['recall']:.1%})")

        # Paper-by-paper summary
        lines.append(f"\nPer-Paper Results:")
        for result in all_results:
            status = result['status']
            paper_id = result['paper_id']

            if status == 'success':
                bc = result['report']['binary_classification']
                lines.append(f"  ✓ {paper_id:<20} Acc: {bc['accuracy']:.1%}, F1: {bc['f1_score']:.1%}")
            elif status == 'no_ground_truth':
                lines.append(f"  ⊘ {paper_id:<20} (no ground truth)")
            else:
                lines.append(f"  ✗ {paper_id:<20} (failed)")

    lines.append("")
    lines.append("=" * 70)

    return lines
