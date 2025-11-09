#!/usr/bin/env python3
"""
Benchmark module for evaluating mathematical formula provers.

This module evaluates how well a prover model can classify formulas as correct or incorrect.
Each formula comes with a flag indicating whether it was purposefully altered.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass, asdict

from .deepseek import DeepseekAgent
import asyncio


@dataclass
class FormulaInput:
    """Structure for formula input with error flag."""
    formula: str
    is_altered: bool
    label: Optional[str] = None
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProverResult:
    """Result from prover evaluation."""
    formula: str
    is_altered: bool  # Ground truth
    detected_as_incorrect: bool  # Model prediction
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    response_time: Optional[float] = None
    error: Optional[str] = None


class ProverBenchmark:
    """Benchmark for evaluating formula provers."""
    
    def __init__(
        self,
        model: str = "openrouter/deepseek/deepseek-prover-v2",
        temperature: float = 0.0,
        max_workers: int = 5
    ):
        """
        Initialize the prover benchmark.
        
        Args:
            model: Model to use for proving/checking formulas
            temperature: Temperature for generation (0.0 for deterministic)
            max_workers: Number of concurrent API calls
        """
        self.model = model
        self.temperature = temperature
        self.max_workers = max_workers
        self.agent = DeepseekAgent()
    
    def check_formula(self, formula_input: FormulaInput) -> ProverResult:
        """
        Check a single formula using the prover model.
        
        Args:
            formula_input: Formula with metadata
            
        Returns:
            ProverResult with classification and reasoning
        """
        start_time = time.time()
        
        try:
            # Use DeepseekAgent
            prompt = f"""Analyze this formula for correctness:

{formula_input.formula}

{"Context: " + formula_input.context if formula_input.context else ""}

Consider:
1. Mathematical syntax and notation
2. Logical consistency
3. Mathematical validity of operations
4. Common mathematical errors (sign errors, incorrect indices, wrong operators, etc.)

Respond in JSON format with the following structure:
{{
    "is_correct": boolean,
    "confidence": float (0.0 to 1.0),
    "reasoning": "Brief explanation of your analysis"
}}"""
            
            # Run async code in sync context
            response = asyncio.run(self.agent.run(prompt))
            response_time = time.time() - start_time
            
            # Parse the response
            try:
                result_data = json.loads(response)
                is_correct = result_data.get("is_correct", True)
                detected_as_incorrect = not is_correct
                confidence = result_data.get("confidence", 0.5)
                reasoning = result_data.get("reasoning", "")
            except json.JSONDecodeError:
                # Fallback parsing if JSON fails
                response_lower = response.lower()
                if "incorrect" in response_lower or "error" in response_lower or "wrong" in response_lower:
                    detected_as_incorrect = True
                else:
                    detected_as_incorrect = False
                confidence = None
                reasoning = response.strip()
            
            return ProverResult(
                formula=formula_input.formula,
                is_altered=formula_input.is_altered,
                detected_as_incorrect=detected_as_incorrect,
                confidence=confidence,
                reasoning=reasoning,
                response_time=response_time
            )
            
        except Exception as e:
            return ProverResult(
                formula=formula_input.formula,
                is_altered=formula_input.is_altered,
                detected_as_incorrect=False,
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    def run_benchmark(
        self,
        formulas: List[FormulaInput],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run benchmark on a list of formulas.
        
        Args:
            formulas: List of formulas to evaluate
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary with benchmark results and metrics
        """
        print(f"\n{'='*70}")
        print("Running Prover Benchmark")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Total formulas: {len(formulas)}")
        print(f"Altered formulas: {sum(1 for f in formulas if f.is_altered)}")
        print(f"Correct formulas: {sum(1 for f in formulas if not f.is_altered)}")
        
        # Process formulas
        results = []
        
        # Process sequentially for DeepseekAgent (since it maintains conversation history)
        for formula in tqdm(formulas, desc="Checking formulas"):
            result = self.check_formula(formula)
            results.append(result)
            # Clear history after each formula to ensure independence
            self.agent.clear_history()
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        # Prepare full report
        report = {
            "metadata": {
                "model": self.model,
                "temperature": self.temperature,
                "total_formulas": len(formulas),
                "timestamp": datetime.now().isoformat(),
                "benchmark_type": "prover_classification"
            },
            "metrics": metrics,
            "results": [asdict(r) for r in results]
        }
        
        # Save results if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save full report
            report_path = output_dir / f"prover_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {report_path}")
        
        # Print summary
        self._print_summary(metrics)
        
        return report
    
    def _calculate_metrics(self, results: List[ProverResult]) -> Dict[str, Any]:
        """Calculate benchmark metrics from results."""
        total = len(results)
        
        # Basic counts
        true_positives = 0   # Correctly identified altered formulas
        true_negatives = 0   # Correctly identified correct formulas
        false_positives = 0  # Incorrectly flagged correct formulas as incorrect
        false_negatives = 0  # Missed altered formulas
        errors = 0
        
        # Calculate confusion matrix
        for result in results:
            if result.error:
                errors += 1
                continue
                
            if result.is_altered and result.detected_as_incorrect:
                true_positives += 1
            elif not result.is_altered and not result.detected_as_incorrect:
                true_negatives += 1
            elif not result.is_altered and result.detected_as_incorrect:
                false_positives += 1
            elif result.is_altered and not result.detected_as_incorrect:
                false_negatives += 1
        
        # Calculate metrics
        accuracy = (true_positives + true_negatives) / (total - errors) if (total - errors) > 0 else 0
        
        # Precision: Of all flagged as incorrect, how many were actually altered?
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        # Recall: Of all altered formulas, how many were detected?
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Response time statistics
        valid_times = [r.response_time for r in results if r.response_time and not r.error]
        avg_response_time = sum(valid_times) / len(valid_times) if valid_times else 0
        
        return {
            "total_formulas": total,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "errors": errors,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "average_response_time": avg_response_time,
            "classification_details": {
                "altered_formulas": {
                    "total": sum(1 for r in results if r.is_altered),
                    "correctly_detected": true_positives,
                    "missed": false_negatives
                },
                "correct_formulas": {
                    "total": sum(1 for r in results if not r.is_altered),
                    "correctly_classified": true_negatives,
                    "incorrectly_flagged": false_positives
                }
            }
        }
    
    def _print_summary(self, metrics: Dict[str, Any]):
        """Print a summary of the benchmark results."""
        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS")
        print(f"{'='*70}")
        print(f"Total formulas evaluated: {metrics['total_formulas']}")
        print(f"Errors during evaluation: {metrics['errors']}")
        print(f"\nAccuracy: {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall: {metrics['recall']:.2%}")
        print(f"F1 Score: {metrics['f1_score']:.2%}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']} (correctly detected altered formulas)")
        print(f"  True Negatives:  {metrics['true_negatives']} (correctly identified correct formulas)")
        print(f"  False Positives: {metrics['false_positives']} (incorrectly flagged correct formulas)")
        print(f"  False Negatives: {metrics['false_negatives']} (missed altered formulas)")
        print(f"\nAverage response time: {metrics['average_response_time']:.2f}s")


def load_formulas_from_json(json_path: Path) -> List[FormulaInput]:
    """
    Load formulas from a JSON file.
    
    Expected format:
    [
        {
            "formula": "E = mc^2",
            "is_altered": false,
            "label": "FORMULA_001",
            "context": "In special relativity..."
        },
        ...
    ]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formulas = []
    for item in data:
        formulas.append(FormulaInput(
            formula=item["formula"],
            is_altered=item["is_altered"],
            label=item.get("label"),
            context=item.get("context"),
            metadata=item.get("metadata")
        ))
    
    return formulas