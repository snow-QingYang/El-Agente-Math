"""OpenReview Crawler package."""

__version__ = "0.1.0"

# Optional crawler imports (may require openreview dependency).
try:
    from .crawler import OpenReviewCrawler, Submission, Review
    from .pipeline import PipelineConfig, run_pipeline
    from .analyze_formula_issues_gpt import FormulaIssueDetector
    from .analyze_iclr_formula_details import FormulaDetailsAnalyzer

    __all__ = [
        "OpenReviewCrawler",
        "Submission",
        "Review",
        "PipelineConfig",
        "run_pipeline",
        "FormulaIssueDetector",
        "FormulaDetailsAnalyzer",
    ]
except Exception:  # pragma: no cover
    __all__: list[str] = []
