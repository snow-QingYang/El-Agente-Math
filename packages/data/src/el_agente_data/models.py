"""Pydantic models for data collection, crawling, and processing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# --- Venue Configuration ---


class VenueConfig(BaseModel):
    """Configuration for an academic venue (conference)."""

    venue_id: str = Field(description="Full OpenReview venue ID")
    name: str = Field(description="Display name")
    year: int = Field(description="Conference year")
    submission_invitation: str = Field(description="Invitation pattern for submissions")
    review_invitation: str = Field(description="Invitation pattern for reviews")
    has_reject_tab: bool = Field(default=True, description="Whether public reject tab exists")
    reject_tab_name: str = Field(default="tab-reject", description="Reject tab URL name")


# --- OpenReview Data ---


class Review(BaseModel):
    """A single peer review from OpenReview."""

    id: str
    forum: str
    replyto: str | None = None
    invitation: str | None = None
    content: dict[str, Any] = Field(default_factory=dict)
    signatures: list[str] = Field(default_factory=list)
    cdate: int | None = None
    tcdate: int | None = None


class Submission(BaseModel):
    """A paper submission from OpenReview."""

    id: str
    number: int | None = None
    forum: str | None = None
    invitation: str | None = None
    content: dict[str, Any] = Field(default_factory=dict)
    authors: list[str] = Field(default_factory=list)
    status: str | None = None
    reviews: list[Review] = Field(default_factory=list)
    cdate: int | None = None
    tcdate: int | None = None


class OpenReviewData(BaseModel):
    """Raw data from OpenReview crawl."""

    submissions: list[Submission] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Formula Issue Detection ---


class FormulaCategory(str, Enum):
    """Categories for formula errors."""

    TYPO = "Typo / Symbol misuse"
    MATH_WRONG = "Mathematically wrong"
    NOTATION = "Notational inconsistency"
    REDUNDANCY = "Redundancy"
    UNCLEAR = "Unclear/Confusing notation"
    MISSING_PROOF = "Missing justification/proof"


class ConfidenceLevel(str, Enum):
    """Confidence levels for detected issues."""

    VERY_CERTAIN = "Very certain"
    CONFIDENT = "Confident"
    SUGGESTION = "Suggestion"
    NOT_SURE = "Not sure"


class FormulaError(BaseModel):
    """A single detected formula error."""

    evidence: str = Field(description="Direct quote from the review")
    formula_location: str = Field(description="Standardized location (e.g., 'EQ (3)')")
    category: str = Field(description="Error category")
    confidence: str = Field(description="Confidence level")
    review_id: str | None = None


class DetectionResult(BaseModel):
    """Result of formula issue detection for a single review."""

    has_formula_issue: bool = False
    confidence: str = "low"
    relevant_quotes: list[str] = Field(default_factory=list)
    summary: str = ""
    formula_errors: list[FormulaError] = Field(default_factory=list)


class ReviewIssue(BaseModel):
    """A review that contains formula issues."""

    review_id: str | None = None
    reviewer: str = "Unknown"
    review_fields: dict[str, Any] = Field(default_factory=dict)
    analysis: DetectionResult | None = None
    formula_errors: list[FormulaError] = Field(default_factory=list)


class PaperIssues(BaseModel):
    """All formula issues for a single paper."""

    paper_id: str
    paper_number: int | None = None
    paper_title: str = ""
    paper_pdf: str = ""
    paper_pdf_link: str = ""
    issues: list[ReviewIssue | FormulaError] = Field(default_factory=list)


class FormulaIssuesData(BaseModel):
    """Container for all detected formula issues."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    issues: list[PaperIssues] = Field(default_factory=list)
    papers: list[PaperIssues] = Field(default_factory=list)


# --- Result JSON (curated issues) ---


class ResultIssue(BaseModel):
    """A single curated issue in result.json."""

    evidence: str = ""
    formula_location: str | list[int] = ""
    category: str = ""
    confidence: str = ""
    review_id: str | None = None
    review_url: str | None = None


class ResultPaper(BaseModel):
    """A paper entry in result.json."""

    paper_id: str
    paper_title: str = ""
    paper_pdf_link: str = ""
    issues: list[ResultIssue] = Field(default_factory=list)


class ResultData(BaseModel):
    """The result.json structure."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    papers: list[ResultPaper] = Field(default_factory=list)


# --- Human Review ---


class PaperReviewDecision(BaseModel):
    """A human review decision for a single issue."""

    keep: bool
    comment: str | None = None
    correct_formula_location: str | None = None
    reviewed_at: str | None = None
    decision: str | None = None  # Legacy field


class ReviewLookupEntry(BaseModel):
    """Entry in review_lookup.json."""

    paper_id: str | None = None
    paper_number: int | None = None
    paper_title: str | None = None
    summary: str | None = None
    weaknesses: str | None = None
    questions: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)


# --- Pipeline Config ---


@dataclass
class PipelineConfig:
    """Configuration for the data processing pipeline."""

    conference: str
    output_dir: Path
    model: str = "gpt-5-nano"
    limit: int | None = None
    concurrency: int = 10


# --- Spotlight ---


class SpotlightPaper(BaseModel):
    """A spotlight/accepted paper entry."""

    id: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    pdf_url: str = ""
    venue: str = ""
