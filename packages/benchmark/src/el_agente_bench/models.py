"""Pydantic models for benchmark evaluation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ConsistencyResult(BaseModel):
    """Result of a single consistency check."""

    matched: bool = Field(description="Whether the agent confirmed the reviewer's issue")
    reason: str = Field(description="Explanation of match/mismatch")


class ConsistencyEntry(BaseModel):
    """A single entry in the consistency report."""

    status: str = "unknown"
    paper_id: str | None = None
    issue_index: int | None = None
    matched: bool | None = None
    reason: str | None = None
    evidence: str | None = None
    analysis_snippet: str | None = None
    path: str | None = None


class ConsistencyMetadata(BaseModel):
    """Metadata for a consistency report."""

    conference: str = ""
    model: str = ""
    timestamp: str = ""
    total_reports: int = 0
    valid_comparisons: int = 0
    matches: int = 0
    mismatches: int = 0
    match_rate: float = 0.0
    errors: int = 0
    skipped: int = 0


class ConsistencyReport(BaseModel):
    """Full consistency report."""

    metadata: ConsistencyMetadata = Field(default_factory=ConsistencyMetadata)
    details: list[ConsistencyEntry] = Field(default_factory=list)


class VerdictStats(BaseModel):
    """Statistics about verdicts from benchmark results."""

    total_files: int = 0
    valid_verdicts: int = 0
    unknown_verdicts: int = 0
    formula_issue: int = 0
    no_formula_issue: int = 0

    @property
    def formula_issue_rate(self) -> float:
        if self.valid_verdicts == 0:
            return 0.0
        return self.formula_issue / self.valid_verdicts * 100

    @property
    def no_formula_issue_rate(self) -> float:
        if self.valid_verdicts == 0:
            return 0.0
        return self.no_formula_issue / self.valid_verdicts * 100


class BenchmarkTask(BaseModel):
    """A single benchmark task (paper + issue to evaluate)."""

    paper_id: str
    issue_file: str
    input_md_path: str
    issue_content: str = ""


class BenchmarkResult(BaseModel):
    """Result of a single benchmark evaluation."""

    paper_id: str
    issue_file: str
    status: str = "pending"
    output_path: str = ""
    answer: str = ""
    issue_content: str = ""
    error: str | None = None
    reason: str | None = None
