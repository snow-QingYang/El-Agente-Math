"""Failure analysis for meta-agent iterations.

Reads benchmark result files and categorizes failures into
false negatives and false positives with concrete examples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - used in function signatures at runtime

VERDICT_RE = re.compile(
    r"^Verdict:\s*(NO_FORMULA_ISSUE|FORMULA_ISSUE)\s*$",
    re.MULTILINE,
)


@dataclass
class FailureExample:
    """A single failure example for analysis."""

    paper_id: str
    issue_file: str
    expected: str  # "FORMULA_ISSUE" or "NO_FORMULA_ISSUE"
    actual: str  # The verdict the agent gave
    issue_content: str  # The issue that was presented
    agent_response: str  # The agent's full response (truncated)


@dataclass
class FailureAnalysis:
    """Structured analysis of benchmark failures."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    fn_examples: list[FailureExample] = field(default_factory=list)
    fp_examples: list[FailureExample] = field(default_factory=list)

    def to_prompt_text(self, max_examples: int = 5) -> str:
        """Format as text for the meta-agent prompt."""
        lines = [
            f"TP={self.tp} FP={self.fp} TN={self.tn} FN={self.fn}",
            f"Precision={self.tp / (self.tp + self.fp):.4f}"
            if (self.tp + self.fp) > 0
            else "Precision=N/A",
            f"Recall={self.tp / (self.tp + self.fn):.4f}"
            if (self.tp + self.fn) > 0
            else "Recall=N/A",
            "",
        ]

        if self.fn_examples:
            lines.append(f"## False Negatives ({self.fn} total — agent missed real issues)")
            for ex in self.fn_examples[:max_examples]:
                lines.append(f"\n### {ex.paper_id}/{ex.issue_file}")
                lines.append(f"**Issue presented:**\n{ex.issue_content[:500]}")
                lines.append(f"**Agent said:** {ex.actual}")
                # Show the relevant part of the agent response
                response_excerpt = ex.agent_response[:800]
                lines.append(f"**Agent reasoning (excerpt):**\n{response_excerpt}")
                lines.append("")

        if self.fp_examples:
            lines.append(
                f"## False Positives ({self.fp} total — agent falsely flagged correct formulas)"
            )
            for ex in self.fp_examples[:max_examples]:
                lines.append(f"\n### {ex.paper_id}/{ex.issue_file}")
                lines.append(f"**Issue presented:**\n{ex.issue_content[:500]}")
                lines.append(f"**Agent said:** {ex.actual}")
                response_excerpt = ex.agent_response[:800]
                lines.append(f"**Agent reasoning (excerpt):**\n{response_excerpt}")
                lines.append("")

        return "\n".join(lines)


def _parse_result_file(result_file: Path) -> tuple[str | None, str, str]:
    """Parse a result file returning (verdict, issue_content, agent_response)."""
    content = result_file.read_text(encoding="utf-8")

    match = VERDICT_RE.search(content)
    verdict = match.group(1) if match else None

    # Extract issue content
    issue_content = ""
    ic_match = re.search(r"\*\*Issue Content:\*\*\n(.+?)(?=\n## )", content, re.DOTALL)
    if ic_match:
        issue_content = ic_match.group(1).strip()

    # Extract agent analysis
    agent_response = ""
    ar_match = re.search(r"## Agentic Reader Analysis\n\n(.+)", content, re.DOTALL)
    if ar_match:
        agent_response = ar_match.group(1).strip()

    return verdict, issue_content, agent_response


def analyze_failures(
    positive_dir: Path,
    negative_dir: Path,
    paper_ids: set[str] | None = None,
) -> FailureAnalysis:
    """Analyze benchmark failures for a set of results.

    Args:
        positive_dir: Directory with results from positive (real issue) samples.
        negative_dir: Directory with results from negative (spotlight) samples.
        paper_ids: Optional set of paper IDs to restrict analysis to.
    """
    analysis = FailureAnalysis()

    # Analyze positives: expect FORMULA_ISSUE
    for result_file in sorted(positive_dir.glob("**/*.result.md")):
        paper_id = result_file.parent.name
        if paper_ids and paper_id not in paper_ids:
            continue

        verdict, issue_content, agent_response = _parse_result_file(result_file)
        if verdict == "FORMULA_ISSUE":
            analysis.tp += 1
        elif verdict == "NO_FORMULA_ISSUE":
            analysis.fn += 1
            analysis.fn_examples.append(
                FailureExample(
                    paper_id=paper_id,
                    issue_file=result_file.stem,
                    expected="FORMULA_ISSUE",
                    actual=verdict,
                    issue_content=issue_content,
                    agent_response=agent_response,
                )
            )

    # Analyze negatives: expect NO_FORMULA_ISSUE
    for result_file in sorted(negative_dir.glob("**/*.result.md")):
        paper_id = result_file.parent.name
        if paper_ids and paper_id not in paper_ids:
            continue

        verdict, issue_content, agent_response = _parse_result_file(result_file)
        if verdict == "NO_FORMULA_ISSUE":
            analysis.tn += 1
        elif verdict == "FORMULA_ISSUE":
            analysis.fp += 1
            analysis.fp_examples.append(
                FailureExample(
                    paper_id=paper_id,
                    issue_file=result_file.stem,
                    expected="NO_FORMULA_ISSUE",
                    actual=verdict,
                    issue_content=issue_content,
                    agent_response=agent_response,
                )
            )

    return analysis
